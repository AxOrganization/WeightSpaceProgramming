"""The presence penalty over a given start of an answer (a branch's prefix, a prime), as a vLLM logits processor.

vLLM applies presence_penalty to generated tokens only. A continued answer has its start in the prompt, so without
this the start's tokens would lose the penalty they carry in a full answer and a branch would sample unlike one.
Per request, extra_args {"pp_from": i, "pp": p} subtracts p from the logit of every token of prompt[i:] that the
continuation has not generated yet; once it is generated the built-in penalty covers it and the entry is cleared (at
the next step, before the subtraction, so a token is never penalised twice).

Interface: vLLM's model runner V2 (vllm.v1.worker.gpu.sample.logits_processor): per-request state by slot, apply()
after the built-in penalties and before temperature, on unscaled logits like the built-in penalty. Module level so
the engine process can import it."""
import torch
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

PP_KEY = "pp_from"


class PrefixPresence(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.rs = req_states
        self.V = req_states.vocab_size
        self.P = torch.zeros((req_states.max_num_reqs, self.V), dtype=torch.bfloat16, device=req_states.device)
        self.pending = {}  # slot -> (pp_from, pp), or None: a request without it (its row must be zero)
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
        for slot, v in self.pending.items():
            self.P[slot].zero_()
            if v is not None:
                a, pp = v
                toks = self.rs.all_token_ids.gpu[slot, a:int(self.rs.prompt_len.np[slot])].long()
                if toks.numel():
                    self.P[slot, toks] = pp
        self.pending.clear()

    def apply(self, logits, ctx):
        if not self.active:
            return logits
        slots = ctx.expanded_idx_mapping.long()
        tl = self.rs.total_len.gpu[slots].long()
        has = tl > self.rs.prompt_len.gpu[slots].long()  # the last token is an output token: the built-in penalty has it
        last = self.rs.all_token_ids.gpu[slots, (tl - 1).clamp(min=0)].long()
        cur = self.P[slots, last]  # no host sync: rewrite the entry, zeroed where the token is an output token
        self.P[slots, last] = torch.where(has, torch.zeros_like(cur), cur)
        k = min(logits.shape[1], self.V)
        logits[:, :k] -= self.P[slots, :k].to(logits.dtype)
        return logits
