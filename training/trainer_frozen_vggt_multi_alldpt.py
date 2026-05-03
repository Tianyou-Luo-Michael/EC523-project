import torch
import clip
import logging
from hydra.utils import instantiate
from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt_multi_alldpt import FrozenVGGT_Multi_AllDPT
from trainer_frozen_vggt import FrozenVGGTTrainer
from train_utils.general import safe_makedirs, model_summary


class FrozenVGGTTrainer_Multi_AllDPT(FrozenVGGTTrainer):


    def _setup_components(self):
        logging.info("FrozenVGGTTrainer_Multi_AllDPT: setting up components")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        self.tb_writer        = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.loss             = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler           = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        clip_model, _ = clip.load(self.clip_model_name, device=self.device)
        for p in clip_model.parameters():
            p.requires_grad_(False)
        clip_model.eval()

        _probe = VGGT.from_pretrained(self.vggt_model_name).to(self.device)
        _probe.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _dummy    = torch.zeros(1, 2, 3, 518, 518, device=self.device)
                _toks, _  = _probe.aggregator(_dummy)
                d_model   = _toks[-1].shape[-1]
                embed_dim = d_model // 2
        logging.info(f"VGGT token dim: d_model={d_model}, embed_dim={embed_dim}")

        d_text = 768 if "ViT-L" in self.clip_model_name else 512

        self.model = FrozenVGGT_Multi_AllDPT(
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
        logging.info(f"Missing (adapters expected): {missing}")
        if unexpected:
            logging.warning(f"Unexpected: {unexpected}")
        del _probe, probe_state
        torch.cuda.empty_cache()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable: {trainable:,} / {total:,}")
        for i, (idx, adapter) in enumerate(zip(self.model.target_indices, self.model.adapters)):
            logging.info(f"  adapters[{i}] → level {idx}  gate={adapter.gate.item():.6f}")

        safe_makedirs(self.logging_conf.log_dir)
        model_summary(self.model, log_file=f"{self.logging_conf.log_dir}/model.txt")

    def _set_frozen_train_mode(self):
        self._inner_model().eval()
        self._inner_model().adapters.train()

    def save_checkpoint(self, epoch):
        import os
        inner = self._inner_model()
        checkpoint_content = {
            "epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "adapters": inner.adapters.state_dict(),
            "optimizer": (
                self.optims[0].optimizer.state_dict()
                if len(self.optims) == 1
                else [o.optimizer.state_dict() for o in self.optims]
            ),
        }
        path = os.path.join(self.checkpoint_conf.save_dir, f"checkpoint_{epoch}.pt")
        os.makedirs(self.checkpoint_conf.save_dir, exist_ok=True)
        import torch
        torch.save(checkpoint_content, path)
        import logging
        logging.info(f"Saved adapters checkpoint → {path}")

    def _load_resuming_checkpoint(self, ckpt_path):
        import torch, logging
        logging.info(f"Resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inner = self._inner_model()
        if "adapters" in checkpoint:
            missing, unexpected = inner.adapters.load_state_dict(
                checkpoint["adapters"], strict=True
            )
            logging.info(f"Adapters loaded. Missing: {missing or 'None'}.")
        if "optimizer" in checkpoint:
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, list):
                for o, s in zip(self.optims, opt_state):
                    o.optimizer.load_state_dict(s)
            else:
                self.optims[0].optimizer.load_state_dict(opt_state)
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"] + 1
        if "steps" in checkpoint:
            self.steps = checkpoint["steps"]
