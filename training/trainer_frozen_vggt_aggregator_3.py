import torch
import clip
import logging
from hydra.utils import instantiate
from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt_aggregator_3 import FrozenVGGT_Aggregator_3
from trainer_frozen_vggt import FrozenVGGTTrainer
from train_utils.general import safe_makedirs, model_summary


class FrozenVGGTTrainer_Aggregator_3(FrozenVGGTTrainer):


    def _setup_components(self):
        logging.info("FrozenVGGTTrainer_Aggregator_3: setting up components")
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

        self.model = FrozenVGGT_Aggregator_3(
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
        logging.info(f"Missing (adapter expected): {missing}")
        if unexpected:
            logging.warning(f"Unexpected: {unexpected}")

        del _probe, probe_state
        torch.cuda.empty_cache()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable: {trainable:,} / {total:,}")
        logging.info(f"Gate at init: {self.model.adapter.gate.item():.6f}")

        safe_makedirs(self.logging_conf.log_dir)
        model_summary(self.model, log_file=f"{self.logging_conf.log_dir}/model.txt")
