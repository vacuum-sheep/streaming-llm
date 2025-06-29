# o3

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    rotate_half,
    repeat_kv,
)
import types

__all__ = [
    "enable_qwen2_pos_shift_attention",
]


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE to a *single* tensor.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with shape ``[bs, n_heads, seq_len, head_dim]`` (or the key variant).
    cos, sin : torch.Tensor
        Cosine / sine caches returned by ``self.rotary_emb``.
    position_ids : torch.Tensor
        Positions for *x* – typically the user‑provided *position_ids* for
        queries, and a freshly‑built ``arange`` for keys.
    unsqueeze_dim : int, default=1
        Dimension along which the position caches will be unsqueezed so that
        they broadcast correctly against *x*.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    return (x * cos) + (rotate_half(x) * sin)


def qwen2_pos_shift_attention_forward(
    self: Union[Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Monkey‑patched *forward* that shifts RoPE so that the **query** always
    uses *user‑provided* positions, while **keys** follow their absolute index
    in the concatenated KV‑cache.
    """
    bsz, q_len, _ = hidden_states.size()

    # Projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # [bs, heads, seq, dim]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.size(-2)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("`layer_idx` must be set when using past_key_value caching.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Build RoPE caches up to *full* length (past + current)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # --- 1. Apply RoPE to **queries** with *given* position_ids -------------
    query_states = apply_rotary_pos_emb_single(
        query_states, cos, sin, position_ids, unsqueeze_dim=1
    )

    # -----------------------------------------------------------------------
    # Concatenate / update KV‑cache *before* touching keys
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # --- 2. Apply RoPE to **keys** with their *absolute* positions ----------
    key_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_single(
        key_states, cos, sin, key_position_ids, unsqueeze_dim=1
    )

    # Repeat KV heads if necessary (GQA)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Attention weights and dropout
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz := bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be {(bsz, self.num_heads, q_len, kv_seq_len)}, got {attn_weights.size()}"
        )

    if attention_mask is not None:
        # Slice to match actual kv length (in case of sliding‑window)
        attn_weights = attn_weights + attention_mask[:, :, :, : kv_seq_len]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # Attention output -------------------------------------------------------------------
    attn_output = torch.matmul(attn_weights, value_states)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"attn_output should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value if use_cache else None


def enable_qwen2_pos_shift_attention(model: nn.Module):
    """Recursively monkey‑patch **all** Qwen2 attention modules inside *model*.

    Usage
    -----
    ```python
    from transformers import AutoModelForCausalLM
    from modify_qwen2 import enable_qwen2_pos_shift_attention

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", attn_implementation="eager")
    enable_qwen2_pos_shift_attention(model)
    ```
    """
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            enable_qwen2_pos_shift_attention(module)

        # Patch *all* subclasses of Qwen2Attention (includes Flash & Sdpa)
        if isinstance(module, Qwen2Attention):
            module.forward = types.MethodType(qwen2_pos_shift_attention_forward, module)
