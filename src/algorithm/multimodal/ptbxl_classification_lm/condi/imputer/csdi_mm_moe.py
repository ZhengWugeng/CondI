import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model import CSDI_base


class ConcatMoEFuser(nn.Module):
    """
    Concat -> Experts MoE fuser

    Inputs:
      - text_vec: (B, text_dim)
      - cxr_vec : (B, cxr_dim)
      - ecg_vec : (B, ecg_dim)
      - text_missing/cxr_missing/ecg_missing: (B,) 1 means missing, 0 means present

    Output:
      - fused_ctx: (B, ctx_dim)
      - moe_w    : (B, num_experts)
    """
    def __init__(
        self,
        text_dim: int,
        cxr_dim: int,
        ecg_dim: int,
        ctx_dim: int,
        num_experts: int,
        hidden: int = 256,
    ):
        super().__init__()
        assert num_experts >= 1, f"num_experts must be >=1, got {num_experts}"
        self.ctx_dim = int(ctx_dim)
        self.num_experts = int(num_experts)

        in_dim = int(text_dim) + int(cxr_dim) + int(ecg_dim) + 3  # + missing flags

        # gate: concat + missing flags -> E logits
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_experts),
        )

        # experts: E independent MLPs: concat -> ctx_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.ctx_dim),
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, text_vec, cxr_vec, ecg_vec, text_missing, cxr_missing, ecg_missing):
        # vec: (B,D)  missing: (B,)
        miss = torch.stack([text_missing, cxr_missing, ecg_missing], dim=-1).float().clamp(0, 1)  # (B,3)

        # Zero-out missing modality embeddings to avoid leaking content
        text_vec = text_vec * (1.0 - miss[:, 0:1])
        cxr_vec  = cxr_vec  * (1.0 - miss[:, 1:2])
        ecg_vec  = ecg_vec  * (1.0 - miss[:, 2:3])

        x = torch.cat([text_vec, cxr_vec, ecg_vec, miss], dim=-1)  # (B, in_dim)

        logits = self.gate(x)  # (B, E)
        w = F.softmax(logits, dim=-1)

        # experts outputs: list[(B,ctx)] -> (B,E,ctx)
        outs = torch.stack([exp(x) for exp in self.experts], dim=1)  # (B, E, ctx)
        fused = (w.unsqueeze(-1) * outs).sum(dim=1)  # (B, ctx)

        return fused, w


class CSDI_MultiModal_MoE(CSDI_base):
    """
    Multimodal CSDI with concat -> experts MoE conditioning.
    """
    def __init__(self, config, device, target_dim: int = 30):
        self.device = device
        self.target_dim = target_dim

        self.ctx_dim = int(config["multimodal"]["ctx_dim"])
        self.ctx_hidden = int(config["multimodal"]["ctx_hidden"])
        self.num_experts = int(config["multimodal"].get("num_experts", 3))

        self.text_dim = int(config["data"]["text_dim"])
        self.ecg_dim = int(config["data"]["ecg_dim"])
        self.cxr_dim = int(config["data"]["cxr_dim"])

        cfg = copy.deepcopy(config)
        super().__init__(target_dim=target_dim, config=cfg, device=device)

        base_side_dim = self.emb_total_dim
        new_side_dim = base_side_dim + self.ctx_dim

        from diff_models import diff_CSDI
        cfg_diff = copy.deepcopy(cfg["diffusion"])
        cfg_diff["side_dim"] = new_side_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(cfg_diff, input_dim).to(self.device)

        # NEW: concat -> E experts fuser
        self.moe = ConcatMoEFuser(
            text_dim=self.text_dim,
            cxr_dim=self.cxr_dim,
            ecg_dim=self.ecg_dim,
            ctx_dim=self.ctx_dim,
            num_experts=self.num_experts,
            hidden=self.ctx_hidden,
        ).to(self.device)

    def process_data(self, batch):
        # (B, L, 30) -> (B, 30, L)
        observed_data = batch["irg_ts"].to(self.device).float().permute(0, 2, 1)
        observed_mask = batch["irg_ts_mask"].to(self.device).float().permute(0, 2, 1)

        observed_tp = batch["ts_tt"].to(self.device).float()  # (B, L)

        gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
        hist_mask = batch["hist_mask"].to(self.device).float().permute(0, 2, 1)

        seq_len = batch["seq_len"].to(self.device).long()

        text_vec = batch["text_vec"].to(self.device).float()
        cxr_vec  = batch["cxr_vec"].to(self.device).float()
        ecg_vec  = batch["ecg_vec"].to(self.device).float()

        text_missing = batch["text_missing"].to(self.device).float().view(-1)
        cxr_missing  = batch["cxr_missing"].to(self.device).float().view(-1)
        ecg_missing  = batch["ecg_missing"].to(self.device).float().view(-1)

        return (observed_data, observed_mask, observed_tp,
                gt_mask, hist_mask, seq_len,
                text_vec, cxr_vec, ecg_vec,
                text_missing, cxr_missing, ecg_missing)

    def get_side_info_mm(self, observed_tp, cond_mask, fused_ctx):
        """
        base side_info: (B, base_side_dim, K, L)
        fused_ctx: (B, ctx_dim) -> broadcast (B, ctx_dim, K, L) -> concat
        """
        base_side = super().get_side_info(observed_tp, cond_mask)  # (B, base, K, L)
        B, _, K, L = base_side.shape
        ctx = fused_ctx[:, :, None, None].expand(B, self.ctx_dim, K, L)
        return torch.cat([base_side, ctx], dim=1)

    def fuse_ctx(self, text_vec, cxr_vec, ecg_vec, text_missing, cxr_missing, ecg_missing):
        fused_ctx, moe_w = self.moe(
            text_vec, cxr_vec, ecg_vec,
            text_missing, cxr_missing, ecg_missing
        )
        return fused_ctx, moe_w

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, observed_tp,
         gt_mask, hist_mask, seq_len,
         text_vec, cxr_vec, ecg_vec,
         text_missing, cxr_missing, ecg_missing) = self.process_data(batch)

        # cond_mask: used for random masking during training; gt_mask is used instead during evaluation
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=hist_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        fused_ctx, moe_w = self.fuse_ctx(
            text_vec, cxr_vec, ecg_vec,
            text_missing, cxr_missing, ecg_missing
        )
        side_info = self.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

        return loss, moe_w.detach()

    @torch.no_grad()
    def evaluate(self, batch, n_samples: int):
        (observed_data, observed_mask, observed_tp,
         gt_mask, _, seq_len,
         text_vec, cxr_vec, ecg_vec,
         text_missing, cxr_missing, ecg_missing) = self.process_data(batch)

        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask

        fused_ctx, moe_w = self.fuse_ctx(
            text_vec, cxr_vec, ecg_vec,
            text_missing, cxr_missing, ecg_missing
        )
        side_info = self.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

        samples = self.impute(observed_data, cond_mask, side_info, n_samples)  # (B, n_samples, K, L)

        # mask out padded region
        for i in range(len(seq_len)):
            L = int(seq_len[i].item())
            target_mask[i, :, L:] = 0
            observed_mask[i, :, L:] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp, moe_w
