import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from spectral import compute_spectral_features


def _spectral_alpha(
    attn_softmax: torch.Tensor,
    self_attn,
    bsz: int,
    q_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    n_img = self_attn.img_end_idx - self_attn.img_start_idx
    A_img = attn_softmax[..., self_attn.img_start_idx : self_attn.img_end_idx]

    if q_len == 1:
        feats = compute_spectral_features(
            A_img,
            mode=self_attn.spectral_mode,
            n_power_iters=self_attn.n_power_iters,
            patch_group_size=self_attn.patch_group_size,
        )
        lam = feats.lambda_max
    else:
        lam = torch.ones(bsz, self_attn.num_heads, device=device, dtype=dtype)

    tau_w = self_attn.tau_weak[self_attn.layer_idx]
    tau_s = self_attn.tau_strong[self_attn.layer_idx]
    T = max(float(self_attn.temperature), 1e-6)

    xi = float(self_attn.suppression_coeff) * torch.sigmoid((tau_w - lam) / T)
    beta = float(self_attn.reinforcement_coeff) * torch.sigmoid((lam - tau_s) / T)
    alpha = (1.0 - xi) + beta
    alpha = torch.clamp(
        alpha,
        min=float(self_attn.small_num_mask),
        max=1.0 + float(self_attn.amp_max_l),
    )
    return alpha


def _spin_mask(
    attn_weights: torch.Tensor,
    self_attn,
    bsz: int,
    q_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_routed_head = int(self_attn.routed_head * self_attn.num_heads)

    attn_scores = attn_weights.permute(0, 2, 1, 3)
    if hasattr(self_attn, "img_start_idx") and hasattr(self_attn, "img_end_idx"):
        attn_scores_headwise = (
            attn_scores[:, -1, :, self_attn.img_start_idx : self_attn.img_end_idx]
            .sum(dim=-1)
            .view(-1, self_attn.num_heads)
        )
    else:
        attn_scores_headwise = attn_scores[:, -1, :, :].sum(dim=-1).view(-1, self_attn.num_heads)

    attn_score_std = attn_scores_headwise.std(dim=1, keepdim=True)
    attn_score_norm = attn_scores_headwise / (attn_score_std + 1e-6)
    gates = F.softmax(attn_score_norm, dim=1)

    _, indices = torch.topk(gates, k=num_routed_head, dim=1)
    mask = F.one_hot(indices, num_classes=self_attn.num_heads).sum(dim=1).to(dtype)

    if self_attn.small_num_mask is not None:
        assert isinstance(self_attn.small_num_mask, (int, float))
        mask = mask.clone()
        mask[mask == 0] = self_attn.small_num_mask

    if q_len > 1:
        mask = torch.cat(
            [
                torch.ones(
                    (bsz * (q_len - 1), self_attn.num_heads), dtype=dtype, device=device
                ),
                mask,
            ],
            dim=0,
        )
    return mask.reshape(bsz, q_len, -1)


def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    attn_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if getattr(self, "capture_features", False):
        with torch.no_grad():
            self._cb_last_attn = attn_softmax.detach()

    if getattr(self, "use_cb_spectral", False) and getattr(self, "spectral_mode", "none") != "none":
        alpha = _spectral_alpha(
            attn_softmax, self, bsz, q_len, query_states.dtype, query_states.device
        )
        mask = alpha.unsqueeze(1).expand(bsz, q_len, self.num_heads).contiguous()
    elif getattr(self, "use_cb_spectral", False) and getattr(self, "spectral_mode", "none") == "none":
        mask = _spin_mask(attn_weights, self, bsz, q_len, query_states.dtype, query_states.device)
    else:
        mask = torch.ones(
            (bsz, q_len, self.num_heads), dtype=query_states.dtype, device=query_states.device
        )

    attn_output = torch.matmul(attn_softmax, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = torch.einsum("bnh,bnhd->bnhd", mask, attn_output)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_modify_spin(
    model,
    start_layer: int,
    end_layer: int,
    img_start_idx: int,
    img_end_idx: int,
    routed_head: float = 0.95,
    use_spin_img: bool = True,
    small_num_mask: Optional[float] = None,
):
    for i in range(start_layer, end_layer):
        attn = model.model.layers[i].self_attn
        attn.img_start_idx = img_start_idx
        attn.img_end_idx = img_end_idx
        attn.routed_head = routed_head
        attn.use_spin_img = use_spin_img
        attn.small_num_mask = small_num_mask
        attn.use_cb_spectral = True
        attn.spectral_mode = "none"
        attn.capture_features = False
        attn.forward = types.MethodType(llama_new_forward, attn)


def llama_modify_cb(
    model,
    start_layer: int,
    end_layer: int,
    img_start_idx: int,
    img_end_idx: int,
    spectral_cfg,
    tau_weak: Optional[torch.Tensor] = None,
    tau_strong: Optional[torch.Tensor] = None,
    capture: bool = False,
):
    """
    Install the CB-Spectral head modulation on layers [start_layer, end_layer).

    If `tau_weak` / `tau_strong` are provided (length end_layer-start_layer),
    they are used directly. Otherwise default thresholds (0.3, 0.6) are used.
    """
    n_gated = end_layer - start_layer
    if tau_weak is None:
        tau_weak = torch.full((n_gated,), 0.3)
    if tau_strong is None:
        tau_strong = torch.full((n_gated,), 0.6)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tau_weak = tau_weak.to(device=device, dtype=torch.float32)
    tau_strong = tau_strong.to(device=device, dtype=torch.float32)

    for i in range(start_layer, end_layer):
        attn = model.model.layers[i].self_attn
        attn.img_start_idx = img_start_idx
        attn.img_end_idx = img_end_idx
        attn.spectral_mode = spectral_cfg.spectral_mode
        attn.n_power_iters = spectral_cfg.n_power_iters
        attn.patch_group_size = spectral_cfg.patch_group_size
        attn.suppression_coeff = spectral_cfg.suppression_coeff
        attn.reinforcement_coeff = spectral_cfg.reinforcement_coeff
        attn.temperature = spectral_cfg.temperature
        attn.small_num_mask = spectral_cfg.small_num_mask
        attn.amp_max_l = spectral_cfg.amp_max(i - start_layer, n_gated)
        attn.use_cb_spectral = True
        attn.capture_features = bool(capture)
        attn.tau_weak = tau_weak
        attn.tau_strong = tau_strong
        attn.forward = types.MethodType(llama_new_forward, attn)
