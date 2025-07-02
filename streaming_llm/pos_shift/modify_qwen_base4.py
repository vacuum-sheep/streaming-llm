# base0

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
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
    # print("cos.shape:", cos.shape)
    # print("position_ids:", position_ids)
    # print("max position_id:", position_ids.max().item(), "seq_len:", cos.shape[0])

    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    return (x * cos) + (rotate_half(x) * sin)

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

def evict_past_key_value(past_key_value, attn_weights, hh_score, num_kv_heads, group_size=7, cache_size=512, k_seq_dim=2, recent_size=256, hh_size=256):
    hh_score = update_hh_score(attn_weights, hh_score, num_kv_heads, group_size)

    seq_len = past_key_value[0].size(k_seq_dim)

    if seq_len <= cache_size:
        return past_key_value, hh_score  # 🔸 Cache not full.

    # keep_idx_kv: (num_kv_heads, hh_size + recent_size)
    _, keep_idx_kv = torch.topk(
            hh_score[:, :seq_len - recent_size],
            k=hh_size,
            dim=-1
    )

    keep_recent = torch.arange(seq_len - recent_size, seq_len,
                                device=keep_idx_kv.device)\
                    .repeat(num_kv_heads, 1)

    keep_idx_kv = torch.cat([keep_idx_kv.sort().values, keep_recent], dim=-1)
    # print("keep_idx_kv", keep_idx_kv)
    # print("keep_idx_kv.max()", keep_idx_kv.max().item(), "seq_len:", seq_len)
    # print("keep_idx_kv.min()", keep_idx_kv.min().item())

    k_squeezed = past_key_value[0].squeeze(0)  # [num_kv_heads, seq_len, head_dim]
    v_squeezed = past_key_value[1].squeeze(0)

    head_dim = k_squeezed.size(-1)
    expanded_keep_idx = keep_idx_kv.unsqueeze(-1).expand(-1, -1, head_dim)  # [num_kv_heads, cache_size, head_dim]

    k_hh_recent = torch.gather(k_squeezed, 1, expanded_keep_idx)  # [num_kv_heads, cache_size, head_dim]
    v_hh_recent = torch.gather(v_squeezed, 1, expanded_keep_idx)

    k_hh_recent = k_hh_recent.unsqueeze(0)  # [1, num_kv_heads, cache_size, head_dim]
    v_hh_recent = v_hh_recent.unsqueeze(0)

    hh_score = torch.gather(hh_score, 1, keep_idx_kv)  # [num_kv_heads, cache_size]

    return (k_hh_recent, v_hh_recent), hh_score
    


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
    cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1))

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


    # if hasattr(past_key_value, "to_legacy_cache"):
    #     print("past_key_value is a legacy cache")

    past_key_value = past_key_value.to_legacy_cache()

    if not hasattr(self, "hh_score"):
        self.hh_score = None

    past_key_value = list(past_key_value)
    past_key_value[self.layer_idx], self.hh_score = evict_past_key_value(past_key_value[self.layer_idx], attn_weights.detach().clone(), self.hh_score, self.num_key_value_heads)
    past_key_value = tuple(past_key_value)

    past_key_value = DynamicCache.from_legacy_cache(past_key_value)

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
