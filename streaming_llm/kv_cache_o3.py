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


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_size=4,                # H2O: number of heavy-hitters per KV head
        num_heads=32,             # H2O: number of query heads
        num_kv_heads=8,           # H2O: number of KV heads
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}, hh_size={hh_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size  # H2O: max tokens kept per KV-head
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        # H2O state
        self.hh_size = hh_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = self.num_heads // self.num_kv_heads
        self.hh_score = None  # cumulative attn scores

    def _pad_and_add(self, old, new):
        if old is None:
            return new.clone()
        L_old, L_new = old.size(-1), new.size(-1)
        if L_new > L_old:
            old = torch.nn.functional.pad(old, (0, L_new - L_old))
        elif L_new < L_old:
            new = torch.nn.functional.pad(new, (L_old - L_new, 0))
        return old + new

    def _update_hh_score(self, attn_score_cache):
        """Accumulate attention mass per source token within this layer."""
        if isinstance(attn_score_cache, (list, tuple)):
            attn_score_cache = attn_score_cache[-1]
        # Skip if attn_score_cache is None or not a tensor (e.g., when using layer-wise H2O)
        if attn_score_cache is None or not isinstance(attn_score_cache, torch.Tensor):
            # print("attn_score_cache is None or not a tensor")
            return
        # attn_score_cache: [bs, heads, tgt, src]
        bsz, _, q_len, kv_seq_len = attn_score_cache.shape
        cache = attn_score_cache.view(
            bsz,
            self.num_kv_heads,
            self.group_size,  # query-heads per kv-head
            q_len,
            kv_seq_len,
        )
        score_per_kv = cache.sum(dim=(0, 2, 3))  # [kv_heads, src]
        self.hh_score = (
            score_per_kv
            if self.hh_score is None
            else self._pad_and_add(self.hh_score, score_per_kv)
        )

    def _evict_h2o(self, past_key_values):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values  # not full yet
        # 1) choose heavy-hitters (exclude the *recent* tail)
        select_scores = self.hh_score[:, : seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values  # sort for deterministic order
        # 2) append recent window indices
        keep_recent = torch.arange(
            seq_len - self.recent_size,
            seq_len,
            device=keep_topk.device,
        ).repeat(self.num_kv_heads, 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)  # [kv_heads, cache_size]
        # 3) boolean mask per kv-head
        mask_kv = torch.zeros_like(self.hh_score, dtype=torch.bool, device=past_key_values[0][0].device)
        mask_kv.scatter_(-1, keep_idx, 1)
        # 4) prune key & value tensors *per kv-head*
        k, v = past_key_values[0][0], past_key_values[0][1]
        k_pruned = k.squeeze(0)[mask_kv].view(self.num_kv_heads, self.cache_size, -1)
        v_pruned = v.squeeze(0)[mask_kv].view(self.num_kv_heads, self.cache_size, -1)
        # 5) also prune score table
        self.hh_score = self.hh_score[mask_kv].view(self.num_kv_heads, self.cache_size)
        # 6) add back batch dim → ([1, kv_heads, L, dim], [1, kv_heads, L, dim])
        return (k_pruned.unsqueeze(0), v_pruned.unsqueeze(0))

    def __call__(self, past_key_values, attn_score_cache=None):
        """
        If attn_score_cache is provided, use H2O logic (update heavy-hitter scores and evict accordingly).
        Otherwise, fall back to the original start+recent logic.
        """
        if attn_score_cache is not None:
            self._update_hh_score(attn_score_cache)
            return self._evict_h2o(past_key_values)
        # Fallback: original logic
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming, attn_score_cache=None):
        """
        If attn_score_cache is provided, use H2O logic (update heavy-hitter scores and evict accordingly).
        Otherwise, fall back to the original start+recent logic.
        """
        if attn_score_cache is not None:
            self._update_hh_score(attn_score_cache)
            return self._evict_h2o(past_key_values)
        # Fallback: original logic
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        # Convert to legacy cache format if needed
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
