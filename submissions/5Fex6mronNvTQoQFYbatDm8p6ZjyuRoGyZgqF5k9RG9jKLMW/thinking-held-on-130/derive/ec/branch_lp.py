"""PrefixPresence: the presence penalty for tokens of a continued prefix .

vLLM applies presence_penalty to generated tokens only. A restart puts the first part of an answer into the prompt,
so its tokens would lose the penalty they carry in a full answer. Per request with extra_args {"pp_from": i, "pp": p}
this processor subtracts p from the logit of every token of prompt[i:] that the continuation has not generated yet;
once a token is generated the built-in penalty covers it and the processor's entry for it is cleared (at the next
step, before the subtraction, so a token is never penalized twice).

vLLM 0.33 model runner V2 interface (vllm.v1.worker.gpu.sample.logits_processor): per-request state by slot,
apply() after the built-in bias / penalties and before temperature, on unscaled logits, like the built-in penalty.
Module level so the engine process can import it."""
import torch
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

PP_KEY = "pp_from"


class PrefixPresence(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.rs = req_states
        self.V = req_states.vocab_size
        self.P = torch.zeros((req_states.max_num_reqs, self.V), dtype=torch.bfloat16, device=req_states.device)
        self.pending = {}   # slot -> (pp_from, pp) or None (declined: row must be zero)
        self.active = set()

    @classmethod
    def validate_params(cls, sampling_params):
        ea = getattr(sampling_params, "extra_args", None) or {}
        if PP_KEY in ea and (int(ea[PP_KEY]) < 0 or float(ea.get("pp", 0.0)) < 0):
            raise ValueError("pp_from and pp must be >= 0")

    def add_request(self, req_idx, sampling_params):
        ea = getattr(sampling_params, "extra_args", None) or {}
        if PP_KEY in ea and float(ea.get("pp", 0.0)) > 0:
            self.pending[req_idx] = (int(ea[PP_KEY]), float(ea["pp"]))
            self.active.add(req_idx)
            return True
        self.pending[req_idx] = None
        self.active.discard(req_idx)
        return False

    def apply_staged_writes(self):
        if not self.pending:
            return
        for slot, v in self.pending.items():
            self.P[slot].zero_()
            if v is None:
                continue
            a, pp = v
            pl = int(self.rs.prompt_len.np[slot])
            toks = self.rs.all_token_ids.gpu[slot, a:pl].long()
            if toks.numel():
                self.P[slot, toks] = pp
        self.pending.clear()

    def apply(self, logits, ctx):
        if not self.active:
            return logits
        slots = ctx.expanded_idx_mapping.long()
        tl = self.rs.total_len.gpu[slots].long()
        pl = self.rs.prompt_len.gpu[slots].long()
        has = tl > pl   # the token committed last step is an output token: the built-in penalty now covers it
        last = self.rs.all_token_ids.gpu[slots, (tl - 1).clamp(min=0)].long()
        cur = self.P[slots, last]   # no host sync: rewrite the entry, zeroed where the token is an output token
        self.P[slots, last] = torch.where(has, torch.zeros_like(cur), cur)
        k = min(logits.shape[1], self.V)
        logits[:, :k] -= self.P[slots, :k].to(logits.dtype)
        return logits
