# o3

import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class H2OKVCache_LayerWise:
    """Implements *layer‑wise* heavy‑hitter + recent window eviction policy."""

    def __init__(
        self,
        hh_size: int = 4,           # 🔹 Heavy‑hitter slots to keep per head.
        recent_size: int = 512,     # 🔹 Sliding window length to keep verbatim.
        k_seq_dim: int = 2,         # 🔹 Index of sequence dimension in *key* tensor.
        v_seq_dim: int = 2,         # 🔹 Ditto for *value* tensor.
        num_heads: int = 32,        # 🔹 Number of attention heads.
        num_kv_heads: int = 8,      # 🔹 Number of key-value heads.
    ):
        print(f"H2OKVCache‑LayerWise: {hh_size=}, {recent_size=}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size  # 🔹 Total allowed per head.
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score: Optional[torch.Tensor] = None  # 🔹 Online cumulative attn score.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = self.num_heads // self.num_kv_heads

    # 🔹 ------------------------------------------------ main entry ------------
    def __call__(self, past_key_values, attn_score_cache):
        """Update heavy‑hitter score then *return pruned* (k,v) cache."""
        self._update_hh_score(attn_score_cache)

        # 🔸 If nothing cached yet, nothing to prune.
        if past_key_values is None:
            return None

        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values  # 🔸 Cache not full.


        # keep_idx_kv: (num_kv_heads, hh_size + recent_size)
        _, keep_idx_kv = torch.topk(
                self.hh_score[:, :seq_len - self.recent_size],
                k=self.hh_size,
                dim=-1
        )
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len,
                                device=keep_idx_kv.device)\
                    .repeat(self.num_kv_heads, 1)

        keep_idx_kv = torch.cat([keep_idx_kv.sort().values, keep_recent], dim=-1)

        # Fix: Use the original mask shape for KV heads, not expanded to query heads
        mask_kv = torch.zeros_like(self.hh_score, dtype=torch.bool, device=past_key_values[0].device)
        mask_kv.scatter_(-1, keep_idx_kv, 1)

        # Get the tensor dimensions properly
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        
        # Use the KV-level mask to index the KV tensors
        k_squeezed = past_key_values[0].squeeze()  # Shape: [num_kv_heads, seq_len, head_dim]
        v_squeezed = past_key_values[1].squeeze()  # Shape: [num_kv_heads, seq_len, head_dim]
        
        # Use the KV-level mask to select the kept positions
        k_hh_recent = k_squeezed[mask_kv]  # This will flatten the selected elements
        v_hh_recent = v_squeezed[mask_kv]  # This will flatten the selected elements
        
        # Reshape back to the expected format for KV heads
        k_hh_recent = k_hh_recent.view(self.num_kv_heads, self.cache_size, head_dim)
        v_hh_recent = v_hh_recent.view(self.num_kv_heads, self.cache_size, head_dim)
        
        # Add back the batch dimension
        k_hh_recent = k_hh_recent.unsqueeze(0)  # Shape: [1, num_kv_heads, cache_size, head_dim]
        v_hh_recent = v_hh_recent.unsqueeze(0)  # Shape: [1, num_kv_heads, cache_size, head_dim]

        # 🔹 Persist updated scores the same way we pruned keys.
        self.hh_score = self.hh_score[mask_kv].view(self.num_kv_heads, self.cache_size)
        return (k_hh_recent, v_hh_recent)

    # 🔹 ------------------------------------------------ proactive eviction ----
    def evict_for_space(self, past_key_values, num_coming):
        """Prune *before* appending `num_coming` fresh tokens."""
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values  # 🔸 Enough room already.


        # 🔹 Heavy‑hitter selection identical to __call__, but horizon shifts.
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        select_hh_scores = self.hh_score[:, : seq_len - self.recent_size + num_coming]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        keep_recent = torch.arange(
            seq_len - self.recent_size + num_coming,
            seq_len,
            device=keep_topk.device,
        ).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
        
        # Fix: Use the original mask shape for KV heads
        mask_kv = torch.zeros_like(self.hh_score, dtype=torch.bool).to(past_key_values[0].device)
        mask_kv.scatter(-1, keep_idx, 1)
        
        # Use the KV-level mask to index the KV tensors
        k_squeezed = past_key_values[0].squeeze()  # Shape: [num_kv_heads, seq_len, head_dim]
        v_squeezed = past_key_values[1].squeeze()  # Shape: [num_kv_heads, seq_len, head_dim]
        
        k_hh_recent = k_squeezed[mask_kv]  # This will flatten the selected elements
        v_hh_recent = v_squeezed[mask_kv]  # This will flatten the selected elements
        
        # Reshape back to the expected format for KV heads
        k_hh_recent = k_hh_recent.view(self.num_kv_heads, self.cache_size, head_dim)
        v_hh_recent = v_hh_recent.view(self.num_kv_heads, self.cache_size, head_dim)
        
        # Add back the batch dimension
        k_hh_recent = k_hh_recent.unsqueeze(0)  # Shape: [1, num_kv_heads, cache_size, head_dim]
        v_hh_recent = v_hh_recent.unsqueeze(0)  # Shape: [1, num_kv_heads, cache_size, head_dim]
        
        self.hh_score = self.hh_score[mask_kv].view(self.num_kv_heads, self.cache_size)
        return (k_hh_recent, v_hh_recent)

    def _pad_and_add(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
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

    # 🔹 ------------------------------------------------ helpers --------------
    def _update_hh_score(self, attn_score_cache: torch.Tensor):
        """Accumulate attention weights per *source* position (heavy‑hitter).

        attn_score_cache shape: [batch, heads, tgt_len, src_len]. We sum out
        batch & tgt dimensions so each *src* token has a cumulative score.
        """
         # attn_score_cache: (bsz, num_heads, q_len, kv_seq_len)
        bsz, _, q_len, kv_seq_len = attn_score_cache.shape

        # 重新分组：把同一 kv-head 的 query-heads 放到一起
        cache = attn_score_cache.view(
            bsz,
            self.num_kv_heads,             # kv heads
            self.group_size,               # q heads / kv head
            q_len,
            kv_seq_len
        )

        # 对 batch、组内 query-head、query_len 三个维度求和
        score_per_kv = cache.sum(dim=(0, 2, 3))   # -> (num_kv_heads, kv_seq_len)

        if self.hh_score is None:
            self.hh_score = score_per_kv
        else:
            # 🔹 Align past scores then add new contributions.
            self.hh_score = self._pad_and_add(self.hh_score, score_per_kv)

    def _clean_scores(self):
        """Reset heavy‑hitter statistics (e.g. between separate prompts)."""
        self.hh_score = None
