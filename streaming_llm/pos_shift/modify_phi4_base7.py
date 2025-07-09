# o3

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache, Cache
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    rotate_half,
    repeat_kv,
    eager_attention_forward,
    Phi3RotaryEmbedding,
)
import types
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

__all__ = [
    "enable_phi4_pos_shift_attention",
]

def pad_and_add(old, new):
    """
    Align `old` and `new` along the last dimension by zero-padding the
    shorter one, then return their element-wise sum.

    Parameters
    ----------
    old : (num_kv_heads, L_old) tensor
        历史累积的 heavy-hitter 分数
    new : (num_kv_heads, L_new) tensor
        本次 forward 计算出的分数（已经按 batch / query 求和）

    Returns
    -------
    out : (num_kv_heads, max(L_old, L_new)) tensor
        对齐后相加的结果
    """
    if old is None:
        # 初始化阶段直接克隆一份，避免后续原地修改
        return new.clone()

    L_old, L_new = old.size(-1), new.size(-1)

    if L_new > L_old:
        # 典型路径：序列在向右增长
        # 在 old 的右侧补零 → tokens index 对齐
        old = F.pad(old, (0, L_new - L_old))          # (left_pad, right_pad)
    elif L_new < L_old:
        # 只在出现"截断 KV-cache"后才会走到这里：
        # 现在需要让 new 对齐到 old 的 **尾部** → 在左边补零
        new = F.pad(new, (L_old - L_new, 0))

    # 长度已经一致，可以安全相加
    return old + new


def update_hh_score(attn_weights, hh_score, num_kv_heads, group_size):
    """Accumulate attention weights per *source* position (heavy‑hitter).

    attn_score_cache shape: [batch, heads, tgt_len, src_len]. We sum out
    batch & tgt dimensions so each *src* token has a cumulative score.
    """
        # attn_score_cache: (bsz, num_heads, q_len, kv_seq_len)
    bsz, _, q_len, kv_seq_len = attn_weights.shape

    # print('attn_weights.shape:', attn_weights.shape)
    # print('num_kv_heads:', num_kv_heads, 'group_size:', group_size)

    # 重新分组：把同一 kv-head 的 query-heads 放到一起
    cache = attn_weights.view(
        bsz,
        num_kv_heads,             # kv heads
        group_size,               # q heads / kv head
        q_len,
        kv_seq_len
    )

    # 对 batch、组内 query-head、query_len 三个维度求和
    score_per_kv = cache.sum(dim=(0, 2, 3))   # -> (num_kv_heads, kv_seq_len)

    if hh_score is None:
        hh_score = score_per_kv
    else:
        # 🔹 Align past scores then add new contributions.
        hh_score = pad_and_add(hh_score, score_per_kv)

    return hh_score

def evict_past_key_value(
    past_key_value,
    attn_weights,
    hh_score,
    num_kv_heads,
    group_size,  # 动态计算而非固定值
    cache_size=512,
    k_seq_dim=2,
    recent_size=256,
    hh_size=256
):
    hh_score = update_hh_score(attn_weights, hh_score, num_kv_heads, group_size)
    
    # 获取当前序列长度
    seq_len = past_key_value[0].size(k_seq_dim)
    
    if seq_len <= cache_size:
        return past_key_value, hh_score

    # 处理多batch情况（假设batch=1）
    k_cache = past_key_value[0].squeeze(0)  # [num_kv_heads, seq_len, head_dim]
    v_cache = past_key_value[1].squeeze(0)
    
    # 分区域选择保留的token
    keep_idx_kv = torch.topk(
        hh_score[:, :seq_len - recent_size],
        k=hh_size,
        dim=-1
    ).indices
    
    # 直接使用sort().values保持顺序
    keep_recent = torch.arange(
        seq_len - recent_size, 
        seq_len, 
        device=keep_idx_kv.device
    ).expand(num_kv_heads, -1)
    
    keep_idx_kv = torch.cat([keep_idx_kv.sort().values, keep_recent], dim=-1)
    
    # 收集保留的KV
    k_hh_recent = k_cache.gather(1, keep_idx_kv.unsqueeze(-1).expand(-1, -1, k_cache.size(-1)))
    v_hh_recent = v_cache.gather(1, keep_idx_kv.unsqueeze(-1).expand(-1, -1, v_cache.size(-1)))
    
    # 保持原始batch维度
    k_hh_recent = k_hh_recent.unsqueeze(0)
    v_hh_recent = v_hh_recent.unsqueeze(0)
    
    # 更新hh_score
    hh_score = hh_score.gather(1, keep_idx_kv)

    return (k_hh_recent, v_hh_recent), hh_score



def apply_rotary_pos_emb(x, cos, sin, position_ids=None, unsqueeze_dim=1):
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
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x_embed = torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)
    return x_embed


def phi4_pos_shift_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Monkey‑patched *forward* that shifts RoPE so that the **query** always
    uses *user‑provided* positions, while **keys** follow their absolute index
    in the concatenated KV‑cache.
    """
    # TODO # 这样处理 rope真的正确吗？

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

    rotary_emb = Phi3RotaryEmbedding(self.config)
    cos, sin = rotary_emb(value_states, position_ids)
    query_states = apply_rotary_pos_emb(query_states, cos, sin)

    key_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0)
    cos, sin = rotary_emb(value_states, key_position_ids)
    # -----------------------------------------------------------------------
    # Concatenate / update KV‑cache *before* touching keys
    if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # --- 2. Apply RoPE to **keys** with their *absolute* positions ----------
    # key_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0)
    # cos, sin = rotary_emb(value_states, key_position_ids)
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

    # h2o
    legacy_cache = past_key_value.to_legacy_cache()
        
    if not hasattr(self, "hh_score"):
        self.hh_score = None
    
    layer_cache = legacy_cache[self.layer_idx]
    new_layer_cache, self.hh_score = evict_past_key_value(
        layer_cache,
        attn_weights.detach(),
        self.hh_score,
        self.num_key_value_heads,
        self.num_key_value_groups, # 传入动态值
        cache_size=512,
        recent_size=256,
        hh_size=256
    )
    
    # 更新缓存
    legacy_cache = list(legacy_cache)
    legacy_cache[self.layer_idx] = new_layer_cache
    past_key_value = DynamicCache.from_legacy_cache(tuple(legacy_cache))

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
