from typing import Any
import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from utils.fmodule import FModule
from .imputer.main_model import CSDI_base, LeadsMoEFuser
from .imputer.config_mm import get_config


def masked_avg_pool1d(x, mask):
    """Average pool over the time dimension, counting only observed positions.

    Args:
        x:    (B, C, L)  feature map
        mask: (B, 1, L)  1 = observed, 0 = missing (broadcastable over C)
    Returns:
        (B, C, 1)  pooled features
    """
    masked_sum = (x * mask).sum(dim=-1, keepdim=True)       # (B, C, 1)
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)   # (B, 1, 1)
    return masked_sum / count


# ---------------------------------------------------------------------------
# Per-Modality Imputer (CSDI with target_dim=1, cross-modal w_cond conditioning)
# ---------------------------------------------------------------------------

class PerModalityImputer(CSDI_base):
    """CSDI-based imputer that processes one modality at a time.

    Each modality is imputed independently, conditioned on:
      1. Its own observed values (via CSDI cond_mask)
      2. w_cond  鈥?cross-modal context from OTHER modalities  (hidden_dim)
      3. w_ins   鈥?current modality observed-data features     (hidden_dim)
    The conditioning vector is concat(w_cond, w_ins) 鈫?cond_dim = 2 * hidden_dim.
    """

    def __init__(self, cond_dim=256):
        config = get_config()
        # Always init on CPU; the outer Model().to(device) will move everything to GPU.
        # _sync_runtime_device() handles alpha_torch migration at forward time.
        super().__init__(target_dim=1, config=config, device='cpu')

        self.cond_dim = cond_dim

        # Override the diffusion model to accept extra side info (w_cond)
        import copy
        from .imputer.diff_models import diff_CSDI
        cfg_diff = copy.deepcopy(config["diffusion"])
        cfg_diff["side_dim"] = self.emb_total_dim + self.cond_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(cfg_diff, input_dim)

    def process_data(self, batch):
        device = next(self.parameters()).device
        return (
            batch["observed_data"].to(device).float(),
            batch["observed_mask"].to(device).float(),
            batch["timepoints"].to(device).float(),
            batch["gt_mask"].to(device).float(),
            batch["hist_mask"].to(device).float(),
            batch["cut_length"].to(device).long(),
        )

    def get_side_info_cond(self, observed_tp, cond_mask, w_cond_vec):
        """Standard CSDI side info + cross-modal w_cond broadcast."""
        base = self.get_side_info(observed_tp, cond_mask)  # (B', base_dim, 1, L)
        B_, _, K, L = base.shape
        cond = w_cond_vec[:, :, None, None].expand(B_, self.cond_dim, K, L)
        return torch.cat([base, cond], dim=1)

    def forward_train(self, x_flat, obs_mask_flat, w_cond_flat):
        """Training: compute diffusion loss + quick x0 estimate for w_imputed.

        Args:
            x_flat:       (B*K, 1, L) per-modality data (missing positions zeroed)
            obs_mask_flat: (B*K, 1, L) observation mask (1=observed, 0=missing)
            w_cond_flat:  (B*K, cond_dim) cross-modal conditioning per modality
        Returns:
            loss:    scalar diffusion training loss
            x0_hat:  (B*K, 1, L) one-step denoised estimate (detached)
        """
        self._sync_runtime_device(x_flat.device)
        B_flat = x_flat.shape[0]
        L = x_flat.shape[-1]

        tp = torch.arange(L, device=x_flat.device, dtype=torch.float32)
        tp = tp.unsqueeze(0).expand(B_flat, L)

        # Random sub-masking of observed values for DDPM training target
        cond_mask = self.get_randmask(obs_mask_flat)
        side_info = self.get_side_info_cond(tp, cond_mask, w_cond_flat)
        loss = self.calc_loss(x_flat, cond_mask, obs_mask_flat, side_info, is_train=1)

        # Quick x0 estimate for the truly-missing positions (detached).
        # Use small t values (low noise) so the single-step estimate is close
        # to the quality of full reverse diffusion used at test time.
        with torch.no_grad():
            full_side = self.get_side_info_cond(tp, obs_mask_flat, w_cond_flat)
            t_max_for_estimate = 2  # very low noise for precise 1-step estimate
            t = torch.randint(0, t_max_for_estimate, [B_flat], device=x_flat.device)
            current_alpha = self.alpha_torch[t]  # (B_flat, 1, 1)
            noise = torch.randn_like(x_flat)
            noisy = (current_alpha ** 0.5) * x_flat + (1.0 - current_alpha) ** 0.5 * noise
            total_input = self.set_input_to_diffmodel(noisy, x_flat, obs_mask_flat)
            predicted = self.diffmodel(total_input, full_side, t)
            x0_hat = (noisy - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5 + 1e-8)

        return loss, x0_hat.detach()

    def impute_single(self, x_flat, obs_mask_flat, w_cond_flat, n_samples=1):
        """Inference: full reverse diffusion to impute missing values."""
        self._sync_runtime_device(x_flat.device)
        B_flat = x_flat.shape[0]
        L = x_flat.shape[-1]

        tp = torch.arange(L, device=x_flat.device, dtype=torch.float32)
        tp = tp.unsqueeze(0).expand(B_flat, L)

        side_info = self.get_side_info_cond(tp, obs_mask_flat, w_cond_flat)
        with torch.no_grad():
            samples = self.impute(x_flat, obs_mask_flat, side_info, n_samples)
        return samples  # (B_flat, n_samples, 1, L)


# ---------------------------------------------------------------------------
# Unchanged building blocks
# ---------------------------------------------------------------------------

class InceptionBlock1D(FModule):
    def __init__(self, input_channels):
        super(InceptionBlock1D, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=19, stride=1, padding=9, bias=False)
        self.convs_conv3 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.convbottle_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.convbottle_conv = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.bnrelu_bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bnrelu_relu = nn.ReLU()

    def forward(self, x):
        bottled = self.bottleneck(x)
        y = torch.cat([
            self.convs_conv1(bottled),
            self.convs_conv2(bottled),
            self.convs_conv3(bottled),
            self.convbottle_conv(self.convbottle_maxpool(x))
        ], dim=1)
        out = self.bnrelu_relu(self.bnrelu_bn(y))
        return out


class Shortcut1D(FModule):
    def __init__(self, input_channels):
        super(Shortcut1D, self).__init__()
        self.input_channels = input_channels
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(self.input_channels, 128, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))


class Inception1DBase(FModule):
    def __init__(self, input_channels=1, output_dim=128):
        super(Inception1DBase, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.inceptionbackbone_1 = InceptionBlock1D(input_channels=self.input_channels)
        self.inceptionbackbone_2 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_3 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_4 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_5 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_6 = InceptionBlock1D(input_channels=128)
        self.shortcut_1 = Shortcut1D(input_channels=self.input_channels)
        self.shortcut_2 = Shortcut1D(input_channels=128)
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.mp = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.bn_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_1 = nn.Dropout(p=0.25, inplace=False)
        self.ln_1 = nn.Linear(256, output_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm1d(output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_2 = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x, return_seq=False, mask=None):
        """
        Args:
            x:    (B, C, L) input features
            mask: (B, 1, L) optional observation mask (1=valid, 0=missing).
                  When provided, AvgPool is replaced by mask-aware pooling so
                  that the mean is computed over observed positions only.
        """
        input_res = x
        x = self.inceptionbackbone_1(x)
        x = self.inceptionbackbone_2(x)
        x = self.inceptionbackbone_3(x)
        x = self.shortcut_1(input_res, x)
        input_res = x.clone()
        x = self.inceptionbackbone_4(x)
        x = self.inceptionbackbone_5(x)
        x = self.inceptionbackbone_6(x)
        x = self.shortcut_2(input_res, x)

        seq_out = x  # (B, 128, L)

        if mask is not None:
            x = torch.cat([self.mp(x), masked_avg_pool1d(x, mask)], dim=1)
        else:
            x = torch.cat([self.mp(x), self.ap(x)], dim=1)
        x = self.flatten(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)
        x = self.ln_1(x)
        x = self.relu(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)

        if return_seq:
            return x, seq_out
        return x


class Inception1DBase250(Inception1DBase):
    """Inception1DBase with output_dim=250, for use as f_main encoder.

    Defined as a separate subclass so that fmodule._model_scale can call
    __class__() with no arguments and still get the correct output_dim.
    """
    def __init__(self, input_channels=1):
        super().__init__(input_channels=input_channels, output_dim=250)


class ModalityEncoder(FModule):
    def __init__(self, input_channels=1):
        super(ModalityEncoder, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.finalizer = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=39, stride=1, padding=19, bias=False),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=39, stride=1, padding=19, bias=True)
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.bottleneck(x)
        x = F.relu(self.convs_conv1(x))
        x = F.relu(self.convs_conv2(x))
        out = self.finalizer(x) + shortcut
        return out


# ---------------------------------------------------------------------------
# New components: CondEncoder, InsImputedEncoder
# ---------------------------------------------------------------------------

class CondEncoder(FModule):
    """Produce w_cond from cross-modal context only.

    w_cond^(m) encodes what OTHER modalities tell us about modality m.
    It does NOT include modality m's own observed data (that is w_ins).
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cross_ctx):
        """
        cross_ctx: (B', D)  aggregated w_ins from OTHER modalities
        Returns:   (B', D)  w_cond embedding
        """
        return self.proj(cross_ctx)


class InsImputedEncoder(FModule):
    """Shared encoder for w_ins (from observed data) and w_imputed (from imputed data).

    Both use the exact same network but receive different inputs:
      w_ins    = encoder(x * obs_mask)          -- observed portion
      w_imputed = encoder(x_imputed * miss_mask) -- imputed portion
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, x, mask=None):
        """
        x:    (B', 1, L)  data segment (irrelevant positions zeroed).
        mask: (B', 1, L)  optional, 1 = valid positions for pooling.
        """
        h = self.conv_block(x)  # (B', 64, L)
        if mask is not None:
            h = masked_avg_pool1d(h, mask)  # (B', 64, 1)
        else:
            h = self.ap(h)
        return self.head(h)  # (B', D)


# ---------------------------------------------------------------------------
# Modified: Contrastive loss  (label-based, within-modality)
# ---------------------------------------------------------------------------

class ContrastiveWeight(nn.Module):
    """Supervised contrastive loss.

    Positive pairs = same label within the SAME modality.
    Negative pairs = different label within the SAME modality.
    Loss is averaged across modalities.
    """

    def __init__(self):
        super(ContrastiveWeight, self).__init__()
        self.temperature = 0.2
        self.num_leads = 12

    def forward(self, batch_emb_om, labels):
        """
        batch_emb_om: (B*M, D) embeddings laid out as
                      [lead0_sample0, ..., lead0_sampleB-1,
                       lead1_sample0, ..., lead11_sampleB-1]
        labels:       (B,) integer class labels
        Returns:
            loss:              scalar
            similarity_matrix: (B*M, B*M)  raw cosine similarity (for rebuild)
            None, None:        placeholders for API compat
        """
        BM, D = batch_emb_om.shape
        B = labels.shape[0]
        M = self.num_leads

        # Full (B*M, B*M) cosine similarity 鈥?used later by AggregationRebuild
        norm_emb = F.normalize(batch_emb_om, dim=1)
        full_sim = torch.matmul(norm_emb, norm_emb.T)

        # --- Per-modality supervised contrastive loss ---
        # Reshape: (M, B, D)  (layout matches [lead0_all, lead1_all, ...])
        features_3d = batch_emb_om.reshape(M, B, D)

        total_loss = batch_emb_om.new_zeros(())
        valid_count = 0

        eye_B = torch.eye(B, dtype=torch.bool, device=batch_emb_om.device)

        for m in range(M):
            emb = F.normalize(features_3d[m], dim=1)   # (B, D)
            sim = torch.matmul(emb, emb.T) / self.temperature  # (B, B)

            # Positive mask: same label AND not self
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye_B)
            pos_count = pos_mask.float().sum(dim=1)   # (B,)
            valid = pos_count > 0
            if valid.sum() == 0:
                continue

            # Denominator: logsumexp over all-except-self
            sim_denom = sim.clone()
            sim_denom[eye_B] = -1e9
            log_denom = torch.logsumexp(sim_denom, dim=1)  # (B,)

            # Numerator: mean of positive similarities
            pos_sim_sum = (sim * pos_mask.float()).sum(dim=1)     # (B,)
            mean_pos_sim = pos_sim_sum[valid] / pos_count[valid]  # (B_valid,)

            loss_m = (-mean_pos_sim + log_denom[valid]).mean()
            total_loss = total_loss + loss_m
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count
        else:
            # No valid positive pairs 鈥?return a zero loss that keeps the
            # gradient chain alive (sum * 0) so .backward() doesn't crash.
            total_loss = (batch_emb_om * 0).sum()

        return total_loss, full_sim, None, None


# ---------------------------------------------------------------------------
# Rebuild / Fuse (unchanged)
# ---------------------------------------------------------------------------

class AggregationRebuild(nn.Module):
    """Within-sample cross-lead rebuild.

    For each sample independently, compute a (K, K) attention matrix over
    leads using cosine similarity, then rebuild each lead's features as a
    weighted combination of all other leads in the SAME sample.
    No cross-sample mixing.
    """
    def __init__(self):
        super(AggregationRebuild, self).__init__()
        self.temperature = 0.2

    def forward(self, sim_features, all_feats):
        """
        sim_features: (B, K, D)   projected features for similarity
        all_feats:    (B, K, F)   raw features to rebuild (F = 5 * D)
        Returns:
            rebuild_weights: (B, K, K)  attention weights per sample
            rebuilt:         (B, K, F)  rebuilt features
        """
        # Per-sample (K, K) cosine similarity
        norm = F.normalize(sim_features, dim=-1)              # (B, K, D)
        sim = torch.bmm(norm, norm.transpose(1, 2))           # (B, K, K)
        sim = sim / self.temperature

        # Mask out self-similarity
        K = sim.shape[1]
        eye = torch.eye(K, device=sim.device).unsqueeze(0)    # (1, K, K)
        sim = sim - eye * 1e12

        weights = F.softmax(sim, dim=-1)                       # (B, K, K)
        rebuilt = torch.bmm(weights, all_feats)                # (B, K, F)
        return weights, rebuilt


class GlobalFuser(FModule):
    """Per-lead gated fusion of original and rebuilt features.

    For each lead independently, learn a gate that blends the original
    5-branch features (old) with the cross-lead rebuilt features (new).
    No cross-lead or cross-sample convolution 鈥?purely within each lead.
    """
    def __init__(self, feat_dim=128 * 4):
        super(GlobalFuser, self).__init__()
        # Input: concat(old, new) per lead 鈫?2*feat_dim 鈫?gate of feat_dim
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, old, new):
        """
        old: (B, 12, 640)  original 5-branch features per lead
        new: (B, 12, 640)  rebuilt features per lead
        Returns:
            fused:  (B, 12, 640)
            alpha:  (B, 12, 640)  gate values (how much of 'old' to keep)
        """
        alpha = self.gate(torch.cat([old, new], dim=-1))  # (B, 12, 640)
        fused = old * alpha + new * (1 - alpha)
        return fused, alpha


# ---------------------------------------------------------------------------
# Branch-level attention gating
# ---------------------------------------------------------------------------

class BranchGating(FModule):
    """Per-lead, per-branch soft gating conditioned on missing rate.

    For each lead, produces importance weights based on:
      - the branch features themselves (content-aware)
      - the per-lead observation ratio  (missing-aware)
    """

    def __init__(self, n_branches=4, hidden_dim=128):
        super().__init__()
        # Input: 5 branch features (5*D) + 1 obs_ratio scalar 鈫?5 gate logits
        self.gate = nn.Sequential(
            nn.Linear(n_branches * hidden_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, n_branches),
        )
        self.n_branches = n_branches
        self.hidden_dim = hidden_dim

    def forward(self, branches, obs_ratio):
        """
        branches:  (B, K, 5, D)   five branch features per lead
        obs_ratio: (B, K)         fraction of observed values per lead [0,1]
        Returns:
            gated:  (B, K, 5*D)   weighted branch features concatenated
            weights: (B, K, 5)    branch importance weights
        """
        B, K, N, D = branches.shape
        flat = branches.reshape(B, K, N * D)                      # (B, K, 5*D)
        gate_input = torch.cat([flat, obs_ratio.unsqueeze(-1)], dim=-1)  # (B, K, 5*D+1)
        logits = self.gate(gate_input)                            # (B, K, 5)
        weights = F.softmax(logits, dim=-1)                       # (B, K, 5)

        # Scale each branch by its weight, then concat
        weighted = branches * weights.unsqueeze(-1)               # (B, K, 5, D)
        # Multiply by n_branches so the sum magnitude stays comparable to plain concat
        gated = (weighted * N).reshape(B, K, N * D)               # (B, K, 5*D)
        return gated, weights


# ---------------------------------------------------------------------------
# Modified: SimilarityProjector (5 branches 鈫?128)
# ---------------------------------------------------------------------------

class SimilarityProjector(FModule):
    def __init__(self):
        super(SimilarityProjector, self).__init__()
        self.ln = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.ln(x)


# ---------------------------------------------------------------------------
# Modified: Classifier (5 branches 脳 12 leads 脳 128 dim)
# ---------------------------------------------------------------------------

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        # f_main: (B, 12, 250)  gated_aux: (B, 12, 128)  鈫?concat 鈫?(B, 12, 378) 鈫?flatten (B, 4536)
        self.head = nn.Sequential(
            nn.Linear((250 + 128) * 12, 512, True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256, True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5, True),
        )

    def forward(self, x):
        return self.head(x)


# ---------------------------------------------------------------------------
# Utility (kept for compat)
# ---------------------------------------------------------------------------

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Pool(FModule):
    def __init__(self, patch_size=16, embed_dim=128, pool_size=10, top_k=3, dropout_value=0.0):
        super(Pool, self).__init__()
        patch_size_pair = _pair((patch_size, patch_size))
        self.top_k = top_k
        self.pool_size = pool_size
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.features_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.features_dropout = nn.Dropout(dropout_value)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size_pair, 1) + embed_dim))
        nn.init.uniform_(self.prompt.data, -val, val)
        nn.init.uniform_(self.features_proj.weight, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, cls_features=None):
        current_pool_size = self.prompt.shape[0]
        x_embed_mean = x_embed
        prompt_norm = self.l2_normalize(self.prompt, dim=1)
        x_embed_mean = self.features_proj(self.features_dropout(x_embed_mean))
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)
        similarity = torch.matmul(x_embed_norm, prompt_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)
        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        if prompt_id.shape[0] < current_pool_size:
            prompt_id = torch.cat([prompt_id, torch.full((current_pool_size - prompt_id.shape[0],),
                                                          torch.min(idx.flatten()), device=prompt_id.device)])
            id_counts = torch.cat([id_counts, torch.full((current_pool_size - id_counts.shape[0],),
                                                          0, device=id_counts.device)])
        _, major_idx = torch.topk(id_counts, k=self.top_k)
        major_prompt_id = prompt_id[major_idx]
        idx = major_prompt_id.expand(x_embed.shape[0], -1)
        self.top_k_idx = idx[0]
        batched_prompt = self.prompt[idx]
        batched_key_norm = prompt_norm[idx]
        x_embed_norm = x_embed_norm.unsqueeze(1)
        sim = batched_key_norm * x_embed_norm
        reduce_sim = torch.sum(sim) / x_embed.shape[0]
        return reduce_sim, batched_prompt


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 12
        self.hidden_dim = 128

        # --- w_mod: learnable per-modality embedding (lookup table) ---
        self.modality_embeddings = nn.Parameter(torch.randn(self.n_leads, self.hidden_dim) * 0.02)

        # --- w_ins encoder (per-lead pre-extractor + shared Inception) ---
        self.feature_extractors = nn.ModuleList(
            [Inception1DBase(input_channels=1, output_dim=128)]
        )
        self.sim_projectors = nn.ModuleList()
        self.pre_extractors = nn.ModuleList()  # for w_ins (observed data)
        # --- f_main encoder (INDEPENDENT per-lead pre-extractors + independent Inception) ---
        self.f_main_pre_extractors = nn.ModuleList()
        for i in range(self.n_leads):
            self.sim_projectors.append(SimilarityProjector())
            self.pre_extractors.append(ModalityEncoder(input_channels=1))
            self.f_main_pre_extractors.append(ModalityEncoder(input_channels=1))
        # Independent Inception for f_main, outputs 250-dim (preserves original signal length info)
        # Uses Inception1DBase250 (subclass) so fmodule._model_scale/__class__() works correctly.
        self.f_main_feature_extractors = nn.ModuleList(
            [Inception1DBase250(input_channels=1)]
        )

        # --- w_imputed encoder (independent, same architecture as w_ins) ---
        self.imputed_feature_extractors = nn.ModuleList(
            [Inception1DBase(input_channels=1)]
        )
        self.imputed_pre_extractors = nn.ModuleList()
        for i in range(self.n_leads):
            self.imputed_pre_extractors.append(ModalityEncoder(input_channels=1))

        # --- w_cond: cross-modal conditional embedding encoder ---
        self.cond_encoder = CondEncoder(hidden_dim=self.hidden_dim)

        # --- ins_imputed_encoder kept for backward compat (aggregation) ---
        self.ins_imputed_encoder = InsImputedEncoder(hidden_dim=self.hidden_dim)

        # --- Supervised contrastive loss on w_ins (per-modality, label-based) ---
        self.contrastive_loss_fn = ContrastiveWeight()

        # --- Auxiliary attention modulator ---
        #     4 w-branches (w_mod, w_ins, w_imputed, w_cond) 鈫?gate that
        #     modulates the main feature f_main from x_completed.
        self.aux_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),  # output in [-1, 1] for additive modulation
        )

        # --- Classifier (operates on modulated main features) ---
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

        # --- Per-modality diffusion imputer ---
        #     cond_dim = 2 * hidden_dim  (concat of w_cond + w_ins)
        self.per_modality_imputer = PerModalityImputer(cond_dim=self.hidden_dim * 2)

        # Visualisation storage
        self.stored_features = {
            'final_feature': [],
            'before_fuse': [],
            'after_rebuild': [],
            'after_fuse': [],
            'label': [],
            'missing_mask': [],
            'alpha_old': []
        }

    # ---- helpers (kept for compat) ----
    def reinit_checklist(self):
        return

    def checking_trained_prompt(self):
        return

    def reset_trained_prompts_checklist(self):
        return

    # ------------------------------------------------------------------
    # Cross-modal context: for each modality m, aggregate features from
    # all OTHER observed modalities (weighted mean).
    # ------------------------------------------------------------------
    def _compute_cross_modal_ctx(self, features, lead_obs_mask):
        """
        features:      (B, K, D)   extracted feature per lead (w_mod)
        lead_obs_mask: (B, K)      bool, True if lead has any observed data
        Returns:       (B, K, D)   cross-modal context per lead
        """
        B, K, D = features.shape
        # mask_others[b, m, j] = 1 iff j is observed AND j != m
        mask_others = lead_obs_mask.float().unsqueeze(1).expand(B, K, K).clone()
        eye = torch.eye(K, device=features.device).unsqueeze(0)  # (1, K, K)
        mask_others = mask_others * (1 - eye)  # exclude self

        weighted = torch.bmm(mask_others, features)  # (B, K, D)
        counts = mask_others.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, K, 1)
        return weighted / counts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, y, leads, quantile=0.75, visualize=False,
                matrix=None, contrastive_weight=0.1, examine_emb=False):
        """
        x: (B, 12, L)  raw ECG leads (-1 = missing)
        y: (B,)         labels
        leads: list of lead indices this client owns
        """
        B, K, L = x.shape
        device = y.device

        # ==================================================================
        # 0.  Preprocessing: masks and clean input
        # ==================================================================
        missing_mask_cat = (x == -1).float()            # (B, 12, L)
        obs_mask = 1 - missing_mask_cat                 # 1 = observed, 0 = missing
        lead_obs_mask = obs_mask.mean(dim=-1) > 0       # (B, 12) bool

        x_clean = x.clone()
        x_clean[x_clean == -1] = 0
        x_obs = x_clean * obs_mask  # (B, 12, L) observed raw data only

        # ==================================================================
        # 1.  w_mod  鈥?learnable per-modality embedding (lookup table)
        # ==================================================================
        w_mod = self.modality_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, 12, 128)

        # ==================================================================
        # 2.  w_ins  鈥?data-specific features from OBSERVED data
        # ==================================================================
        w_ins = torch.zeros(B, 12, self.hidden_dim, device=device)
        for lead in range(12):
            lead_input = x_obs[:, lead, :].view(B, 1, -1)       # (B, 1, L), 0 at missing
            lead_mask  = obs_mask[:, lead, :].view(B, 1, -1)
            # Smooth-fill: replace 0s with per-lead observed mean to avoid Conv pollution
            obs_mean = (lead_input * lead_mask).sum(dim=-1, keepdim=True) / lead_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            lead_filled = lead_input * lead_mask + obs_mean * (1 - lead_mask)
            feat = self.pre_extractors[lead](lead_filled)
            feat = feat * lead_mask    # semantically only keep observed positions
            feat = self.feature_extractors[0](feat, mask=lead_mask)
            w_ins[:, lead, :] = feat

        # ==================================================================
        # 3.  w_cond  鈥?cross-modal context ONLY (other modalities' w_ins)
        # ==================================================================
        cross_ctx = self._compute_cross_modal_ctx(w_ins, lead_obs_mask)
        cross_ctx_flat = cross_ctx.reshape(B * 12, self.hidden_dim)
        batch_w_cond = self.cond_encoder(cross_ctx_flat).reshape(B, 12, self.hidden_dim)

        # ==================================================================
        # 4.  Per-modality imputation
        # ==================================================================
        w_cond_flat = batch_w_cond.reshape(B * 12, self.hidden_dim)
        w_ins_flat_for_imp = w_ins.reshape(B * 12, self.hidden_dim)
        imputer_cond = torch.cat([w_cond_flat, w_ins_flat_for_imp], dim=-1)

        x_obs_flat = x_obs.reshape(B * 12, 1, L)
        obs_mask_flat = obs_mask.reshape(B * 12, 1, L)

        if self.training:
            loss_imputation, x0_hat_flat = self.per_modality_imputer.forward_train(
                x_obs_flat, obs_mask_flat, imputer_cond
            )
            x0_hat = x0_hat_flat.reshape(B, 12, L)
            self.loss_imputation = loss_imputation
        else:
            self.loss_imputation = 0.0
            with torch.no_grad():
                samples = self.per_modality_imputer.impute_single(
                    x_obs_flat, obs_mask_flat, imputer_cond.detach(), n_samples=1
                )
            x0_hat = samples[:, 0, :, :].reshape(B, 12, L)

        # ==================================================================
        # 5.  w_imputed  鈥?data-specific features from IMPUTED data
        # ==================================================================
        miss_mask = 1 - obs_mask
        w_imputed = torch.zeros(B, 12, self.hidden_dim, device=device)
        for lead in range(12):
            lead_imputed = (x0_hat.detach() * miss_mask)[:, lead, :].view(B, 1, -1)  # 0 at observed
            lead_miss_mask = miss_mask[:, lead, :].view(B, 1, -1)
            # Smooth-fill: replace 0s (observed positions) with imputed mean
            imp_mean = (lead_imputed * lead_miss_mask).sum(dim=-1, keepdim=True) / lead_miss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            lead_imp_filled = lead_imputed * lead_miss_mask + imp_mean * (1 - lead_miss_mask)
            feat = self.imputed_pre_extractors[lead](lead_imp_filled)
            feat = feat * lead_miss_mask
            feat = self.imputed_feature_extractors[0](feat, mask=lead_miss_mask)
            w_imputed[:, lead, :] = feat

        # ==================================================================
        # 6. f_main 鈥?main features from completed data (observed + imputed)
        # PRIMARY signal for classification.
        # Uses INDEPENDENT f_main_pre_extractors + f_main_feature_extractors.
        # Output per lead: (B, 250) 鈥?preserves original signal length dimensionality.
        # ==================================================================
        x_completed = x_obs + x0_hat.detach() * miss_mask
        f_main = torch.zeros(B, 12, 250, device=device)
        for lead in range(12):
            feat_c = self.f_main_pre_extractors[lead](x_completed[:, lead, :].view(B, 1, -1))
            feat_c = self.f_main_feature_extractors[0](feat_c)  # (B, 250), no mask needed
            f_main[:, lead, :] = feat_c                          # (B, 12, 250)

        # ==================================================================
        # 7.  Auxiliary gate on w-branches
        #     [w_mod, w_ins, w_imputed, w_cond] 鈫?gated_aux (B, 12, 128)
        #     Then concat with f_main 鈫?(B, 12, 378)
        # ==================================================================
        aux = torch.cat([w_mod, w_ins, w_imputed, batch_w_cond], dim=-1)  # (B, 12, 512)
        gated_aux = self.aux_gate(aux)                                     # (B, 12, 128)
        f_out = torch.cat([f_main, gated_aux], dim=-1)                     # (B, 12, 378)

        # ==================================================================
        # 8.  Supervised contrastive loss on w_ins (per-modality, label-based)
        #     Layout: (M, B, D) 鈫?reshape to (B*M, D) with lead-major order
        # ==================================================================
        w_ins_flat = w_ins.permute(1, 0, 2).reshape(-1, self.hidden_dim)  # (M*B, D)
        contra_loss, _, _, _ = self.contrastive_loss_fn(w_ins_flat, y.long())
        self.loss_contrastive = contra_loss

        # ==================================================================
        # 9.  Classification
        # ==================================================================
        outputs = self.classifier(f_out.reshape(B, -1))  # (B, 378*12=4536) 鈫?(B, 5)
        loss = self.criterion(outputs, y.type(torch.int64))

        gamma = getattr(self, "gamma", 0.1)
        imp_loss = self.loss_imputation
        if isinstance(imp_loss, torch.Tensor) and getattr(self, "_detach_imputation_loss", False):
            imp_loss = imp_loss.detach()
        loss = loss + gamma * imp_loss

        # Contrastive loss added only when flag is set (Phase B)
        lambda_contra = getattr(self, "_contrastive_lambda", 0.0)
        if lambda_contra > 0:
            loss = loss + lambda_contra * self.loss_contrastive

        return None, loss, outputs
