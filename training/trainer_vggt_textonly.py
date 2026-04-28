# Trainer for VGGTTextOnly: frozen image aggregator (teacher) + trainable text encoder (student).
#
# Inherits from Trainer and overrides:
#   _setup_components        — builds VGGTTextOnly from a pretrained VGGT checkpoint
#   train_epoch              — keeps aggregator in eval mode; manages head warm-up
#   _step                    — tokenise captions → text forward → alignment + task loss
#   save_checkpoint          — saves text-encoder + head weights only
#   _load_resuming_checkpoint — loads text-encoder + head weights
#   _process_batch           — normalise pose/depth only when GT keys are present

import os
import logging
import time
from typing import List, Mapping, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from hydra.utils import instantiate
from transformers import AutoTokenizer

from trainer import (
    Trainer,
    AverageMeter,
    ProgressMeter,
    chunk_batch_for_accum_steps,
    copy_data_to_device,
)
from vggt.models.vggt import VGGT
from vggt.models.vggt_textonly import VGGTTextOnly, alignment_loss
from data.caption_manifest import load_caption_lookup_by_mode, get_caption_for_seq
from train_utils.general import safe_makedirs, model_summary
from train_utils.optimizer import OptimizerWrapper
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


class VGGTTextOnlyTrainer(Trainer):
    """
    Trainer for VGGTTextOnly.

    The TextEncoder and downstream prediction heads are trainable.  The
    Aggregator (image teacher) is frozen throughout.

    Training loss per step:
        objective = alignment_weight * alignment_loss
                  + task_weight     * task_loss   (depth/camera/points, when GT available)

    Alignment loss:
        Per-block MSE + cosine-similarity between text tokens (student) and
        frame-averaged image tokens (teacher, detached).

    Head warm-up:
        For the first `freeze_heads_steps` steps the downstream heads remain
        frozen.  Only the text encoder trains during this phase so the
        alignment signal stabilises first.  After warm-up the heads are
        unfrozen for end-to-end fine-tuning.
    """

    def __init__(
        self,
        *,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_text_backbone: bool = False,
        vggt_model_name: str = "facebook/VGGT-1B",
        alignment_weight: float = 1.0,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
        task_weight: float = 0.1,
        freeze_heads_steps: int = 500,
        caption_manifest_path: Optional[str] = None,
        caption_mode: str = "concise",
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_point: bool = False,
        **kwargs,
    ):
        # Store before super().__init__ because _setup_components is called inside it.
        self.text_model_name = text_model_name
        self.freeze_text_backbone = freeze_text_backbone
        self.vggt_model_name = vggt_model_name
        self.alignment_weight = alignment_weight
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.task_weight = task_weight
        self.freeze_heads_steps = freeze_heads_steps
        self.enable_camera = enable_camera
        self.enable_depth = enable_depth
        self.enable_point = enable_point
        self._heads_unfrozen = False
        self.text_max_length = 77

        if caption_manifest_path is not None:
            self.caption_lookup = load_caption_lookup_by_mode(
                caption_manifest_path, mode=caption_mode
            )
            logging.info(
                f"Loaded {len(self.caption_lookup)} captions "
                f"({caption_mode}) from {caption_manifest_path}"
            )
        else:
            self.caption_lookup = {}

        super().__init__(**kwargs)

    # ── Optimizer: trainable params only ─────────────────────────────────────

    @property
    def optims(self):
        return self._optims

    @optims.setter
    def optims(self, value):
        # The parent builds an optimizer over ALL parameters; we replace it
        # with one that covers only requires_grad=True params to avoid
        # allocating momentum buffers for the 1.2 B frozen aggregator weights.
        if self.mode != "val":
            self._optims = self._build_optims()
        else:
            self._optims = value

    def _build_optims(self):
        import hydra

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        assert trainable_params, (
            "No trainable parameters found — text encoder may not be built yet."
        )
        optimizer = hydra.utils.instantiate(self.optim_conf.optimizer, trainable_params)
        schedulers = [{} for _ in optimizer.param_groups]
        return [OptimizerWrapper(optimizer, schedulers)]

    # ── Component setup ───────────────────────────────────────────────────────

    def _setup_components(self):
        """
        Build VGGTTextOnly:
          1. Probe pretrained VGGT to discover embed_dim at runtime.
          2. Construct VGGTTextOnly and load aggregator + head weights.
          3. Freeze the aggregator (teacher stays frozen throughout).
          4. Freeze heads for the warm-up phase.
        """
        logging.info("VGGTTextOnlyTrainer: setting up components")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        # TensorBoard, loss, gradient clipper, AMP scaler
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # ── (1) Probe VGGT for embed_dim ──────────────────────────────────────
        logging.info(f"Probing pretrained VGGT ({self.vggt_model_name}) for embed_dim …")
        _probe = VGGT.from_pretrained(self.vggt_model_name).to(self.device)
        _probe.eval()
        amp_dtype = torch.bfloat16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                _dummy = torch.zeros(1, 2, 3, 518, 518, device=self.device)
                _toks, _ = _probe.aggregator(_dummy)
                d_model = _toks[-1].shape[-1]
                embed_dim = d_model // 2
        logging.info(f"VGGT token dim: d_model={d_model}, embed_dim={embed_dim}")

        # ── (2) Build VGGTTextOnly ────────────────────────────────────────────
        self.model = VGGTTextOnly(
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            enable_camera=self.enable_camera,
            enable_depth=self.enable_depth,
            enable_point=self.enable_point,
            text_model_name=self.text_model_name,
            freeze_text_backbone=self.freeze_text_backbone,
        )

        # ── (3) Load pretrained aggregator + head weights ─────────────────────
        probe_state = {k: v.cpu() for k, v in _probe.state_dict().items()}
        missing, unexpected = self.model.load_state_dict(probe_state, strict=False)
        if self.rank == 0:
            # Missing keys are the text_encoder (expected — no corresponding
            # weights in vanilla VGGT).  Unexpected keys are disabled heads.
            text_enc_missing = [k for k in missing if k.startswith("text_encoder.")]
            other_missing = [k for k in missing if not k.startswith("text_encoder.")]
            if other_missing:
                logging.warning(f"Unexpected missing keys (non-text-encoder): {other_missing}")
            logging.info(
                f"Loaded aggregator + head weights from {self.vggt_model_name}. "
                f"text_encoder keys (new, expected): {len(text_enc_missing)}"
            )

        del _probe, probe_state
        torch.cuda.empty_cache()

        # ── (4) Freeze aggregator (teacher) ───────────────────────────────────
        self.model.freeze_aggregator()
        logging.info("Aggregator frozen (teacher).")

        # ── (5) Freeze heads during warm-up ───────────────────────────────────
        self.model.freeze_heads()
        self._heads_unfrozen = False
        logging.info(
            f"Heads frozen for warm-up ({self.freeze_heads_steps} steps). "
            "Only text encoder will train during this phase."
        )
        self._refresh_gradient_clipper_configs()

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        logging.info(f"Tokenizer loaded: {self.text_model_name}")

        # ── (6) Log summary ───────────────────────────────────────────────────
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        if self.rank == 0:
            logging.info(
                f"Trainable: {trainable:,} / {total:,} "
                f"(frozen aggregator excluded from trainable count)"
            )
            safe_makedirs(self.logging_conf.log_dir)
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary → {model_summary_path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _inner_model(self) -> VGGTTextOnly:
        """Unwrap DDP to access the underlying VGGTTextOnly module."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _set_student_train_mode(self):
        """
        Set training modes:
          - Aggregator: always eval (frozen teacher — suppresses BN / dropout).
                    - text_encoder: train.
          - Heads: train only after warm-up; eval during warm-up.
        """
        nn.Module.train(self.model, False)  # everything → eval first

        inner = self._inner_model()
        inner.text_encoder.train()

        # Heads → train only once warm-up is complete
        if self._heads_unfrozen:
            for head in (inner.camera_head, inner.depth_head, inner.point_head):
                if head is not None:
                    head.train()

    def _refresh_gradient_clipper_configs(self) -> None:
        """Ensure gradient clipping covers all currently trainable modules."""
        if self.gradient_clipper is None or not hasattr(self.gradient_clipper, "configs"):
            return

        default_max_norm = None
        default_norm_type = 2
        if self.gradient_clipper.configs:
            default_max_norm = self.gradient_clipper.configs[0].get("max_norm")
            default_norm_type = self.gradient_clipper.configs[0].get("norm_type", 2)

        inner = self._inner_model()
        module_names = ["text_encoder"]
        for name in ("camera_head", "depth_head", "point_head"):
            head = getattr(inner, name, None)
            if head is not None and any(p.requires_grad for p in head.parameters()):
                module_names.append(name)

        self.gradient_clipper.configs = [
            {
                "module_names": module_names,
                "max_norm": default_max_norm,
                "norm_type": default_norm_type,
            }
        ]
        self.gradient_clipper.is_initialized = False
        self.gradient_clipper.params_to_clip_by_config = None
        self.gradient_clipper.setup_clipping(self.model)
        logging.info("Gradient clipper modules: %s", ", ".join(module_names))

    def _maybe_unfreeze_heads(self):
        """
        Unfreeze downstream heads after the warm-up period and rebuild the
        optimizer so the new trainable parameters get momentum states.
        """
        if self._heads_unfrozen:
            return
        if self.steps["train"] < self.freeze_heads_steps:
            return

        inner = self._inner_model()
        inner.unfreeze_heads()
        self._heads_unfrozen = True

        # Rebuild optimizer to include the newly unfrozen head params
        self._optims = self._build_optims()
        self._refresh_gradient_clipper_configs()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(
            f"[step {self.steps['train']}] Heads unfrozen — "
            f"trainable params: {trainable:,}"
        )

    def _tokenize(self, captions: List[str]) -> tuple:
        """
        Tokenise captions using the pretrained tokenizer.

        Returns:
            input_ids (Tensor[B, L]): token ids on self.device.
            attention_mask (Tensor[B, L]): attention mask on self.device.
        """
        enc = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _get_captions(self, batch: Mapping) -> Optional[List[str]]:
        """
        Resolve captions for the batch.

        Priority:
          1. batch["captions"]          — dataset provides captions directly.
          2. caption_lookup[seq_name]   — look up by sequence name from manifest.
          3. None                       — no captions available (alignment skipped).
        """
        if "captions" in batch:
            return list(batch["captions"])
        if self.caption_lookup and "seq_name" in batch:
            captions = [
                get_caption_for_seq(self.caption_lookup, sn, default="")
                for sn in batch["seq_name"]
            ]
            if any(c for c in captions):
                return captions
        return None

    # ── Per-step forward + loss ───────────────────────────────────────────────

    def _step(
        self,
        batch: Mapping,
        model: nn.Module,
        phase: str,
        loss_meters: dict,
    ) -> dict:
        """
        Single forward step for VGGTTextOnly.

        Training:
          - Enforce student train mode (aggregator stays eval).
          - Check whether heads should be unfrozen (warm-up guard).
          - Run forward pass with both text and image inputs.
          - Compute alignment loss (text tokens ↔ image tokens).
          - Compute optional task loss (depth / camera / points) via MultitaskLoss.

        Validation:
          - Single forward pass; alignment loss only (no task loss unless GT present).
        """
        if phase == "train":
            self._maybe_unfreeze_heads()
            self._set_student_train_mode()

        captions = self._get_captions(batch)
        if captions is None:
            logging.warning(
                "No captions found in batch — skipping this step. "
                "Provide captions via the dataset or caption_manifest_path."
            )
            # Return a zero loss dict so the training loop does not crash.
            zero = torch.zeros(1, device=self.device, requires_grad=True)
            return {"objective": zero}

        input_ids, attention_mask = self._tokenize(captions)
        images = batch["images"]  # [B, S, 3, H, W]

        # ── (a) Forward pass ──────────────────────────────────────────────────
        predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
        )

        # ── (b) Alignment loss ────────────────────────────────────────────────
        if "image_tokens_list" in predictions:
            loss_align = alignment_loss(
                predictions["text_tokens_list"],
                predictions["image_tokens_list"],
                mse_weight=self.mse_weight,
                cosine_weight=self.cosine_weight,
            )
        else:
            # Validation: aggregator not run in eval mode inside model.forward
            # because self.training=False.  Recompute teacher tokens here.
            inner = self._inner_model()
            with torch.no_grad():
                img_toks, _ = inner.aggregator(images)
                img_toks = [t.mean(dim=1, keepdim=True).detach() for t in img_toks]
            loss_align = alignment_loss(
                predictions["text_tokens_list"],
                img_toks,
                mse_weight=self.mse_weight,
                cosine_weight=self.cosine_weight,
            )

        # ── (c) Task loss ─────────────────────────────────────────────────────
        # Detach confidence outputs so the text encoder is not trained via the
        # confidence weighting alone (same rationale as FrozenVGGTTrainer).
        loss_predictions = {**predictions}
        for conf_key in ("depth_conf", "world_points_conf"):
            if conf_key in loss_predictions:
                loss_predictions[conf_key] = loss_predictions[conf_key].detach()

        loss_dict = self.loss(loss_predictions, batch)

        # ── (d) Combined objective ────────────────────────────────────────────
        loss_dict["loss_align"] = loss_align
        loss_dict["objective"] = (
            self.alignment_weight * loss_align
            + self.task_weight * loss_dict["objective"]
        )

        # ── (e) Logging ───────────────────────────────────────────────────────
        log_data = {**predictions, **loss_dict, **batch}
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        # Log alignment loss to TensorBoard separately for easy monitoring
        if self.rank == 0 and self.steps[phase] % self.logging_conf.log_freq == 0:
            self.tb_writer.log(
                f"Alignment/{phase}_loss",
                loss_align.item(),
                self.steps[phase],
            )
            self.tb_writer.log(
                f"Alignment/{phase}_heads_unfrozen",
                int(self._heads_unfrozen),
                self.steps[phase],
            )

        self.steps[phase] += 1
        return loss_dict

    # ── Training epoch (enforce aggregator eval) ──────────────────────────────

    def train_epoch(self, train_loader):
        """
        Override parent's train_epoch to keep the aggregator in eval mode.

        The parent calls ``self.model.train()`` at the top of the loop.
        ``_set_student_train_mode()`` inside ``_step`` then forces the
        aggregator back to eval before every forward pass.
        """
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            meter_key = f"Grad/{param_names}"
            loss_meters[meter_key] = AverageMeter(meter_key, self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break

            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.amp.autocast("cuda", enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps
            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            assert data_iter <= limit_train_batches
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )

            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    meter_key = f"Grad/{key}"
                    if meter_key not in loss_meters:
                        loss_meters[meter_key] = AverageMeter(
                            meter_key, self.device, ":.4f"
                        )
                    loss_meters[meter_key].update(grad_norm)

            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(
        self, epoch: int, checkpoint_names: Optional[List[str]] = None
    ):
        """
        Save text_encoder + head weights.

        The aggregator (1.2 B pretrained VGGT params) and text backbone are
        NOT saved — they are re-loaded from their checkpoints at run time.

        Saved keys:
          text_encoder   — text_encoder.state_dict()
          camera_head    — camera_head.state_dict()  (if enabled)
          depth_head     — depth_head.state_dict()   (if enabled)
          point_head     — point_head.state_dict()   (if enabled)
          epoch, steps, optimizer, scaler, heads_unfrozen
        """
        if self.distributed_rank != 0:
            dist.barrier()
            return

        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)

        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        inner = self._inner_model()
        content = {
            "epoch": epoch,
            "steps": self.steps,
            "heads_unfrozen": self._heads_unfrozen,
            "time_elapsed": self.time_elapsed_meter.val,
            "text_encoder": inner.text_encoder.state_dict(),
            "optimizer": (
                self.optims[0].optimizer.state_dict()
                if len(self.optims) == 1
                else [o.optimizer.state_dict() for o in self.optims]
            ),
        }
        if self.optim_conf.amp.enabled:
            content["scaler"] = self.scaler.state_dict()
        for attr in ("camera_head", "depth_head", "point_head"):
            head = getattr(inner, attr, None)
            if head is not None:
                content[attr] = head.state_dict()

        for name in checkpoint_names:
            path = os.path.join(checkpoint_folder, f"{name}.pt")
            torch.save(content, path)
            logging.info(f"Saved checkpoint → {path}")

        dist.barrier()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """
        Load text_encoder + head weights and training state from a checkpoint.
        """
        logging.info(
            f"VGGTTextOnlyTrainer: resuming from {ckpt_path} (rank {self.rank})"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inner = self._inner_model()

        # Text encoder
        if "text_encoder" in ckpt:
            missing, unexpected = inner.text_encoder.load_state_dict(
                ckpt["text_encoder"], strict=True
            )
            logging.info(
                f"text_encoder loaded. Missing: {missing or 'None'}. "
                f"Unexpected: {unexpected or 'None'}."
            )

        # Heads
        for attr in ("camera_head", "depth_head", "point_head"):
            if attr in ckpt:
                head = getattr(inner, attr, None)
                if head is not None:
                    head.load_state_dict(ckpt[attr], strict=True)
                    logging.info(f"{attr} loaded from checkpoint.")

        # Restore warm-up state
        if "heads_unfrozen" in ckpt and ckpt["heads_unfrozen"]:
            inner.unfreeze_heads()
            self._heads_unfrozen = True
            logging.info("Heads restored to unfrozen state from checkpoint.")
        self._refresh_gradient_clipper_configs()

        # Optimizer (training only)
        if "optimizer" in ckpt:
            if self.mode == "val":
                logging.info(
                    "Validation mode: skipping optimizer state restore from checkpoint."
                )
            else:
                # Rebuild optimizer so its param groups reflect the current set of
                # trainable parameters (heads may have just been unfrozen above).
                # Without this, loading a state dict saved with head params into an
                # optimizer that only covers text_encoder raises a size-mismatch error.
                self._optims = self._build_optims()

                opt_state = ckpt["optimizer"]
                if isinstance(opt_state, dict):
                    self.optims[0].optimizer.load_state_dict(opt_state)
                elif isinstance(opt_state, list):
                    for optim, state in zip(self.optims, opt_state):
                        optim.optimizer.load_state_dict(state)

        # Training progress
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"] + 1
        self.steps = ckpt.get("steps", {"train": 0, "val": 0})
        self.ckpt_time_elapsed = ckpt.get("time_elapsed", 0)

        if self.mode != "val" and self.optim_conf.amp.enabled and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

    # ── Batch processing ──────────────────────────────────────────────────────

    def _process_batch(self, batch: Mapping) -> Mapping:
        """
        Normalise camera extrinsics and 3-D points when GT annotations are
        present.  Skips gracefully if the keys are absent (text-only datasets
        that do not include GT depth / pose).
        """
        _required = ("extrinsics", "cam_points", "world_points", "depths", "point_masks")
        if all(k in batch for k in _required):
            (
                batch["extrinsics"],
                batch["cam_points"],
                batch["world_points"],
                batch["depths"],
            ) = normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )
        return batch

    # ── Scalar logging ────────────────────────────────────────────────────────

    def _update_and_log_scalars(
        self, data: Mapping, phase: str, step: int, loss_meters: dict
    ):
        """
        Override parent to derive batch_size from images rather than
        extrinsics (which may not be present in text-only datasets).
        """
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = (
            data["images"].shape[0]
            if "images" in data
            else data.get("extrinsics", torch.zeros(1)).shape[0]
        )

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in loss_meters:
                    loss_meters[meter_key].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
