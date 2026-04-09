"""Gemma 4 NCHW chunks — minimal diff from stateless_chunks.

Change vs stateless: hidden_states stays (1, C, 1, 1) between layers,
only the RMSNorm modules are patched to NCHW-native. Attention internals
are unchanged (they already work).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb
from ane_ops_nchw import ANERMSNormNCHW, ane_softmax_nchw

from .gemma4 import Gemma4Model


def _copy_weights(src, dst):
    with torch.no_grad():
        dst.weight.data.copy_(src.weight.data.to(dst.weight.dtype))


def _patch_norms_to_nchw(layer, config):
    """Swap layer's NHC RMSNorms with NCHW-native ones."""
    hs = config.hidden_size
    eps = config.rms_norm_eps
    for name in ["input_layernorm", "post_attention_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm",
                 "post_per_layer_input_norm"]:
        src = getattr(layer, name)
        new_norm = ANERMSNormNCHW(hs, eps=eps).to(MODEL_DTYPE)
        _copy_weights(src, new_norm)
        setattr(layer, name, new_norm)


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scaleless RMSNorm for V, fp16."""
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


def _run_layer(
    layer, layer_idx,
    hidden_states_nchw,  # (1, C, 1, 1) — NCHW throughout
    cos_s, sin_s, cos_f, sin_f,
    causal_mask, update_mask,
    K_in, V_in,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)

    residual = hidden_states_nchw
    h = layer.input_layernorm(hidden_states_nchw)  # NCHW → NCHW
    x = h  # already NCHW, no permute

    # Projections (Conv2d on NCHW directly)
    q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    is_kv_shared = config.is_kv_shared(layer_idx)
    K_out, V_out = K_in, V_in

    if not is_kv_shared:
        k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
        v = v_norm(v)
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        if hd < max_hd:
            k_padded = F.pad(k, (0, max_hd - hd))
            v_padded = F.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v

        K_out = K_in * (1 - update_mask) + k_padded.expand_as(K_in) * update_mask
        V_out = V_in * (1 - update_mask) + v_padded.expand_as(V_in) * update_mask

        K_for_attn = K_out[..., :hd]
        V_for_attn = V_out[..., :hd]

        if layer_idx == 13:
            kv_store_13_k = K_out[..., :256]
            kv_store_13_v = V_out[..., :256]
        elif layer_idx == 14:
            kv_store_14_k = K_out[..., :512]
            kv_store_14_v = V_out[..., :512]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + causal_mask
    attn_weights = ane_softmax_nchw(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    # (1, num_heads, 1, hd) → NCHW (1, hidden, 1, 1) for o_proj
    attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(1, num_heads * hd, 1, 1)
    attn_output = layer.self_attn["o_proj"](attn_output)  # NCHW → NCHW
    attn_output = layer.post_attention_layernorm(attn_output)  # NCHW RMSNorm
    hidden_states_nchw = residual + attn_output

    # MLP (all NCHW)
    residual = hidden_states_nchw
    h = layer.pre_feedforward_layernorm(hidden_states_nchw)  # NCHW
    gate = layer.mlp["gate_proj"](h)
    up = layer.mlp["up_proj"](h)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)
    mlp_out = layer.post_feedforward_layernorm(mlp_out)
    hidden_states_nchw = residual + mlp_out

    # Per-layer input
    residual_pl = hidden_states_nchw.to(MODEL_DTYPE)
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]
    per_layer_slice_nchw = per_layer_slice.permute(0, 2, 1).unsqueeze(3).to(MODEL_DTYPE)

    gate_in = hidden_states_nchw.to(MODEL_DTYPE)
    gated = layer.per_layer_input_gate(gate_in)
    gated = F.gelu(gated, approximate="tanh")
    gated = (gated * per_layer_slice_nchw).to(MODEL_DTYPE)
    gated = layer.per_layer_projection(gated)
    gated = layer.post_per_layer_input_norm(gated)
    hidden_states_nchw = (residual_pl + gated).to(MODEL_DTYPE)
    # layer_scalar is fp32; cast to fp16 before multiply
    scalar = layer.layer_scalar.to(MODEL_DTYPE).view(1, 1, 1, 1)
    hidden_states_nchw = (hidden_states_nchw * scalar).to(MODEL_DTYPE)

    return (hidden_states_nchw, K_out, V_out,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


class NCHWChunk1(nn.Module):
    START, END = 0, 8

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        for layer in self.layers:
            _patch_norms_to_nchw(layer, model.config)

    def forward(self, hidden_states, causal_mask, update_mask,
                per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                K_in, V_in):
        # Enter NCHW once (explicit dtype cast)
        h = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(3)

        dummy13k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy13v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        dummy14k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        dummy14v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_outs, V_outs = [], []
        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            K_slot = K_in[local_idx].unsqueeze(0)
            V_slot = V_in[local_idx].unsqueeze(0)
            h, K_new, V_new, *_ = _run_layer(
                self.layers[local_idx], layer_idx, h,
                cos_s, sin_s, cos_f, sin_f, causal_mask, update_mask,
                K_slot, V_slot, self.config, per_layer_combined,
                dummy13k, dummy13v, dummy14k, dummy14v,
            )
            K_outs.append(K_new.squeeze(0))
            V_outs.append(V_new.squeeze(0))

        # Exit NCHW once
        h_out = h.squeeze(3).permute(0, 2, 1)  # (1, C, 1, 1) → (1, 1, C)
        return h_out, torch.stack(K_outs, dim=0), torch.stack(V_outs, dim=0)
