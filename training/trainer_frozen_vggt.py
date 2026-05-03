import os
import logging
from typing import List, Mapping, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import clip

from hydra.utils import instantiate

from trainer import Trainer
from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT
from data.caption_manifest import load_caption_lookup_by_mode, get_caption_for_seq
from train_utils.general import safe_makedirs, model_summary
from train_utils.optimizer import OptimizerWrapper


class FrozenVGGTTrainer(Trainer):


    def __init__(
        self,
        *,
        clip_model_name: str = "ViT-B/32",
        vggt_model_name: str = "facebook/VGGT-1B",
        adapter_dim: int = 512,
        caption_manifest_path: Optional[str] = None,
        caption_mode: str = "concise",
        margin: float = 0.05,
        margin_loss_weight: float = 0.5,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_point: bool = False,
        enable_track: bool = False,
        **kwargs,
    ):

        self.clip_model_name = clip_model_name
        self.vggt_model_name = vggt_model_name
        self.adapter_dim = adapter_dim
        self.margin = margin
        self.margin_loss_weight = margin_loss_weight
        self.enable_camera = enable_camera
        self.enable_depth = enable_depth
        self.enable_point = enable_point
        self.enable_track = enable_track


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


    @property
    def optims(self):
        return self._optims

    @optims.setter
    def optims(self, value):


        if self.mode != "val":
            self._optims = self._build_optims()
        else:
            self._optims = value


    def _setup_components(self):


        logging.info("FrozenVGGTTrainer: setting up components")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}


        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)


        logging.info(f"Loading CLIP model: {self.clip_model_name}")
        clip_model, _ = clip.load(self.clip_model_name, device=self.device)
        for p in clip_model.parameters():
            p.requires_grad_(False)
        clip_model.eval()


        logging.info(f"Probing VGGT ({self.vggt_model_name}) for token dim …")
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


        d_text = 768 if "ViT-L" in self.clip_model_name else 512


        self.model = FrozenVGGT(
            clip_model=clip_model,
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            adapter_dim=self.adapter_dim,
            d_text=d_text,
            enable_camera=self.enable_camera,
            enable_depth=self.enable_depth,
            enable_point=self.enable_point,
            enable_track=self.enable_track,
        )


        probe_state = {k: v.cpu() for k, v in _probe.state_dict().items()}
        missing, unexpected = self.model.load_state_dict(probe_state, strict=False)
        if self.rank == 0:
            logging.info(
                f"VGGT weights loaded. Missing (adapter expected): {missing}"
            )
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected}")

        del _probe, probe_state
        torch.cuda.empty_cache()


        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        if self.rank == 0:
            logging.info(
                f"Trainable: {trainable:,} / {total:,} "
                f"(CLIP ~400 M excluded from both counts)"
            )
            logging.info(
                f"Adapter gate at init: {self.model.adapter.gate.item():.6f}  "
                f"(must be 0.000000)"
            )
            safe_makedirs(self.logging_conf.log_dir)
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary → {model_summary_path}")


    def _inner_model(self) -> FrozenVGGT:

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _set_frozen_train_mode(self):


        nn.Module.train(self.model, False)
        self._inner_model().adapter.train()


    def _get_captions(self, batch: Mapping) -> Optional[List[str]]:


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

    def _step(
        self,
        batch: Mapping,
        model: nn.Module,
        phase: str,
        loss_meters: dict,
    ) -> dict:


        if phase == "train":
            self._set_frozen_train_mode()

        captions = self._get_captions(batch)
        images = batch["images"]


        predictions = model(images=images, captions=captions)


        loss_predictions = {**predictions}
        if "depth_conf" in loss_predictions:
            loss_predictions["depth_conf"] = loss_predictions["depth_conf"].detach()
        if "world_points_conf" in loss_predictions:
            loss_predictions["world_points_conf"] = loss_predictions["world_points_conf"].detach()
        loss_dict = self.loss(loss_predictions, batch)


        if phase == "train" and captions is not None and "depth" in predictions:

            idx_wrong = torch.randperm(len(captions))
            captions_wrong = [captions[i] for i in idx_wrong]


            with torch.no_grad():
                preds_wrong = model(images=images, captions=captions_wrong)

            depth_correct = predictions["depth"].squeeze(-1)
            depth_wrong = preds_wrong["depth"].squeeze(-1)

            valid = (
                batch["point_masks"].bool()
                if "point_masks" in batch
                else torch.ones_like(depth_correct, dtype=torch.bool)
            )
            gt_depth = batch["depths"]

            n_valid = valid.sum().item()
            if n_valid == 0:


                loss_margin = depth_correct.sum() * 0.0
            else:


                err_wrong = (depth_wrong[valid] - gt_depth[valid]).abs().detach()
                err_correct_live = (depth_correct[valid] - gt_depth[valid]).abs()

                loss_margin = torch.nn.functional.relu(
                    err_correct_live - err_wrong + self.margin
                ).mean()

            loss_dict["loss_margin"] = loss_margin
            loss_dict["objective"] = (
                loss_dict["objective"] + self.margin_loss_weight * loss_margin
            )


        log_data = {**predictions, **loss_dict, **batch}
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])


        if (
            self.rank == 0
            and self.steps[phase] % self.logging_conf.log_freq == 0
        ):
            inner = self._inner_model()
            if hasattr(inner, "adapter"):
                self.tb_writer.log(
                    "Adapter/gate",
                    inner.adapter.gate.item(),
                    self.steps[phase],
                )

        self.steps[phase] += 1
        return loss_dict


    def _build_optims(self):


        import hydra
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        assert trainable_params, "No trainable parameters found — adapter may not be built yet."
        optimizer = hydra.utils.instantiate(self.optim_conf.optimizer, trainable_params)


        schedulers = [{} for _ in optimizer.param_groups]
        return [OptimizerWrapper(optimizer, schedulers)]


    def save_checkpoint(
        self, epoch: int, checkpoint_names: Optional[List[str]] = None
    ):


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
        checkpoint_content = {
            "epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "adapter": inner.adapter.state_dict(),
            "optimizer": (
                self.optims[0].optimizer.state_dict()
                if len(self.optims) == 1
                else [o.optimizer.state_dict() for o in self.optims]
            ),
        }
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        for name in checkpoint_names:
            path = os.path.join(checkpoint_folder, f"{name}.pt")
            torch.save(checkpoint_content, path)
            logging.info(f"Saved adapter checkpoint → {path}")

        dist.barrier()

    def _load_resuming_checkpoint(self, ckpt_path: str):


        logging.info(
            f"FrozenVGGTTrainer: resuming from {ckpt_path} (rank {self.rank})"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        inner = self._inner_model()

        if "adapter" in checkpoint:
            missing, unexpected = inner.adapter.load_state_dict(
                checkpoint["adapter"], strict=True
            )
            if self.rank == 0:
                logging.info(
                    f"Adapter loaded. Missing: {missing or 'None'}. "
                    f"Unexpected: {unexpected or 'None'}."
                )
        elif "model" in checkpoint:

            adapter_state = {
                k.removeprefix("adapter."): v
                for k, v in checkpoint["model"].items()
                if k.startswith("adapter.")
            }
            if adapter_state:
                inner.adapter.load_state_dict(adapter_state, strict=True)
                logging.info("Extracted and loaded adapter weights from full model checkpoint.")
            else:
                logging.warning("No adapter.* keys found in checkpoint['model'].")


        if "optimizer" in checkpoint:
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, dict):
                self.optims[0].optimizer.load_state_dict(opt_state)
            elif isinstance(opt_state, list):
                for optim, state in zip(self.optims, opt_state):
                    optim.optimizer.load_state_dict(state)


        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"] + 1

        self.steps = checkpoint.get("steps", {"train": 0, "val": 0})
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
