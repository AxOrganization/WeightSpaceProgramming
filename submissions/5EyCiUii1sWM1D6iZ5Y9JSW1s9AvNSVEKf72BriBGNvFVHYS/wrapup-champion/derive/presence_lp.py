"""vLLM (model runner V2) logits processor for step g1: the profile's presence penalty over a continued trace.

A g1 continuation starts from a parent prefix whose generated part (prompt tokens from extra_args["pp_from"] on) was
sampled with the profile's presence penalty. vLLM's own presence penalty sees only the new tokens, and its logit_bias
holds at most 1024 tokens (a long thinking prefix has more distinct tokens). This processor subtracts extra_args["pp"]
from the logit of every token that occurs in the prefix's generated part or in the new output: the presence penalty
of one uninterrupted run. Requests without "pp" are untouched; use it with presence_penalty=0 in SamplingParams.
"""
from __future__ import annotations

import numpy as np

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

CHUNK = 32   # logits rows per update (bounds the temporary [rows, vocab] buffer)


def _args(sp):
    ea = getattr(sp, "extra_args", None) or {}
    return ea if ea.get("pp") else None


class PrefixPresenceLP(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        import torch
        self.rs = req_states
        n, dev = req_states.max_num_reqs, req_states.device
        self.active_np = np.zeros(n, dtype=bool)
        self.init_np = np.zeros(n, dtype=bool)      # token set still to be built from the committed tokens
        self.pp_np = np.zeros(n, dtype=np.float32)
        self.from_np = np.zeros(n, dtype=np.int64)
        self.pp = torch.zeros(n, dtype=torch.float32, device=dev)
        self.seen = torch.zeros((n, req_states.vocab_size), dtype=torch.bool, device=dev)
        self._dirty = False

    @classmethod
    def validate_params(cls, sp):
        ea = _args(sp)
        if ea is None:
            return
        if int(ea.get("pp_from", -1)) < 0:
            raise ValueError("extra_args pp_from (first penalised prompt position) is required with pp")
        if sp.presence_penalty:
            raise ValueError("PrefixPresenceLP applies the presence penalty itself: set presence_penalty=0")

    def add_request(self, req_idx, sp):
        ea = _args(sp)
        self.active_np[req_idx] = ea is not None
        self.pp_np[req_idx] = float(ea["pp"]) if ea is not None else 0.0
        if ea is not None:
            self.from_np[req_idx] = int(ea["pp_from"])
            self.init_np[req_idx] = True
        self._dirty = True
        return ea is not None

    def apply_staged_writes(self):
        if not self._dirty:
            return
        import torch
        self.pp.copy_(torch.from_numpy(self.pp_np), non_blocking=False)
        self._dirty = False

    def apply(self, logits, ctx):
        slots_np = ctx.idx_mapping_np
        if not self.active_np[slots_np].any():
            return logits
        import torch
        tl = self.rs.total_len.gpu
        ids = self.rs.all_token_ids.gpu
        for s in slots_np[self.init_np[slots_np] & self.active_np[slots_np]]:   # once per request (and on resume)
            s = int(s)
            t = int(tl[s])
            self.seen[s].zero_()
            self.seen[s, ids[s, int(self.from_np[s]):t].long()] = True
            self.init_np[s] = False
        slots = ctx.expanded_idx_mapping.long()
        t = tl[slots].long()
        for back in (1, 2):   # the newest committed tokens join the set (a slot without pp is never applied)
            self.seen[slots, ids[slots, (t - back).clamp_min(0)].long()] = True
        V = self.seen.shape[1]
        for i in range(0, logits.shape[0], CHUNK):
            sl = slots[i:i + CHUNK]
            logits[i:i + CHUNK, :V] -= self.pp[sl][:, None] * self.seen[sl].to(logits.dtype)
        return logits
