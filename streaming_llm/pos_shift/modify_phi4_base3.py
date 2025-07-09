# o3

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    rotate_half,
    repeat_kv,
    eager_attention_forward,
)
import types
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

__all__ = [
    "enable_phi4_pos_shift_attention",
]


def apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    return q_embed


def phi4_pos_shift_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Monkey‑patched *forward* that shifts RoPE so that the **query** always
    uses *user‑provided* positions, while **keys** follow their absolute index
    in the concatenated KV‑cache.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    # Projections
    qkv = self.qkv_proj(hidden_states)
    query_pos = self.config.num_attention_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    # [bs, heads, seq, dim]
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    kv_seq_len = key_states.size(-2)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("`layer_idx` must be set when using past_key_value caching.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states = apply_rotary_pos_emb(query_states, cos, sin)

    # -----------------------------------------------------------------------
    # Concatenate / update KV‑cache *before* touching keys
    if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # --- 2. Apply RoPE to **keys** with their *absolute* positions ----------
    key_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, key_position_ids)

    attention_interface: Callable = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def enable_phi4_pos_shift_attention(model: nn.Module):
    """Recursively monkey‑patch **all** Phi4 attention modules inside *model*.

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
            enable_phi4_pos_shift_attention(module)

        # Patch *all* subclasses of Qwen2Attention (includes Flash & Sdpa)
        if isinstance(module, Phi3Attention):
            module.forward = types.MethodType(phi4_pos_shift_attention_forward, module)
