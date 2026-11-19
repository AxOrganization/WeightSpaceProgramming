"""Teacher-forced reads of a model over its own answers, at single token positions.

Reader runs one text through the model and returns, for chosen layers, the input of the MLP at chosen positions (what
an MLP unit reads), scores of given read directions at every position, and the next-token distribution at every
position as the evaluation samples from it (temperature, top-k, top-p and presence penalty of the profile), so the
chance that the thinking block is closed at a position ("exit hazard") can be read without generating.

A candidate unit can be tried without building a model: `units` adds its output to the MLP output of its layer
(gate / up rows, down column: the unit `wsp.gate` writes), exactly what the edited weights would compute.
"""
import numpy as np
import torch

from wsp.probe import _child, decoder, load_model

CLOSE = "</think>"


class Reader:
    def __init__(self, model_dir, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok = load_model(model_dir, device)
        self.owner, self.layers = decoder(self.model)
        self.norm, self.head, self.device = self.owner.norm, self.model.get_output_embeddings(), device
        self.close_id = self.tok.convert_tokens_to_ids(CLOSE)
        self.n_layers, self.d = len(self.layers), self.head.weight.shape[1]

    def encode(self, text):
        """-> (token ids, char offset of each token)"""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return enc["input_ids"], [a for a, _ in enc["offset_mapping"]]

    @torch.no_grad()
    def weight(self, layer, name):
        """A matrix of a layer's MLP ("gate_proj", "up_proj", "down_proj") as float32 numpy."""
        return _child(self.layers[layer], name).weight.detach().float().cpu().numpy()

    @torch.no_grad()
    def run(self, ids, take=None, score=None, units=(), want_final=False, want_rms=False, act_layers=()):
        """One forward pass over `ids`.
        take:  {layer: positions}  -> out["x"][layer]  = MLP inputs at those positions [n, d] (float32, cpu)
        score: {layer: D [k, d]}   -> out["s"][layer]  = D @ x at every position [T, k]
        units: [{"layer", "gate" [d], "up" [d], "down" [d]}] added to their layers' MLP outputs
        want_final -> out["final"] = output of the final norm at every position [T, d] (on the device)
        want_rms   -> out["rms"]   = rms of the final norm's input at every position [T]
        act_layers -> out["act"][layer] = mean |activation| of every hidden unit of that layer's MLP [inter]"""
        out, hooks = {"x": {}, "s": {}, "act": {}}, []
        by_layer = {}
        for u in units:
            by_layer.setdefault(u["layer"], []).append(u)

        def tens(v):
            return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=self.device)

        for li, layer in enumerate(self.layers):
            mlp = _child(layer, "mlp")
            if (take and li in take) or (score and li in score):
                def pre(m, inp, li=li):
                    x = inp[0][0].float()
                    if take and li in take:
                        out["x"][li] = x[torch.as_tensor(take[li], device=x.device)].cpu().numpy()
                    if score and li in score:
                        out["s"][li] = (x @ tens(score[li]).T).cpu().numpy()
                hooks.append(mlp.register_forward_pre_hook(pre))
            if li in by_layer:
                G = torch.stack([tens(u["gate"]) for u in by_layer[li]])     # [n, d]
                U = torch.stack([tens(u["up"]) for u in by_layer[li]])
                Dn = torch.stack([tens(u["down"]) for u in by_layer[li]])    # [n, d]

                def post(m, inp, o, G=G, U=U, Dn=Dn):
                    x = inp[0].float()
                    act = torch.nn.functional.silu(x @ G.T) * (x @ U.T)       # [1, T, n]
                    return o + (act @ Dn).to(o.dtype)
                hooks.append(mlp.register_forward_hook(post))
        for li in act_layers:
            hooks.append(_child(self.layers[li], "down_proj").register_forward_pre_hook(
                lambda m, inp, li=li: out["act"].__setitem__(li, inp[0][0].float().abs().mean(0).cpu().numpy())))
        if want_final:
            hooks.append(self.norm.register_forward_hook(lambda m, i, o: out.__setitem__("final", o[0])))
        if want_rms:
            hooks.append(self.norm.register_forward_pre_hook(
                lambda m, inp: out.__setitem__("rms", inp[0][0].float().pow(2).mean(-1).sqrt().cpu().numpy())))
        try:
            self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return out

    @torch.no_grad()
    def hazard(self, ids, n_prompt, final, sampling, start=0, stop=None, chunk=256, token=None):
        """Per generated position j in [start, stop) (token ids[n_prompt + j] is about to be sampled): the
        probability that sampling gives `token` (default: the closing token), as vLLM samples: presence penalty on
        the tokens generated so far, temperature, top-k, then top-p.
        -> {"p_close", "p_raw" (plain softmax), "p_top"}, each [stop - start]."""
        token = self.close_id if token is None else token
        t, k, p = sampling.get("temperature", 1.0), int(sampling.get("top_k", 0) or 0), sampling.get("top_p", 1.0)
        pen = sampling.get("presence_penalty", 0.0)
        gen = torch.tensor(ids[n_prompt:], device=self.device)
        stop = len(gen) if stop is None else min(stop, len(gen))
        G, V = max(stop - start, 0), self.head.weight.shape[0]
        seen = torch.zeros(V, dtype=torch.bool, device=self.device)
        if start > 0:
            seen[gen[:start]] = True
        res = {key: np.zeros(G, np.float32) for key in ("p_close", "p_raw", "p_top")}
        for c in range(0, G, chunk):
            n, g0 = min(chunk, G - c), start + c
            logits = self.head(final[n_prompt - 1 + g0:n_prompt - 1 + g0 + n]).float()        # [n, V]
            res["p_raw"][c:c + n] = torch.softmax(logits, -1)[:, token].cpu().numpy()
            if pen:   # tokens generated before each position: those before the chunk, and earlier ones inside it
                m = torch.zeros(n, V, dtype=torch.bool, device=self.device)
                if n > 1:
                    m[torch.arange(1, n, device=self.device), gen[g0:g0 + n - 1]] = True
                m = (torch.cummax(m.to(torch.uint8), 0).values.bool()) | seen
                logits = logits - pen * m
                seen[gen[g0:g0 + n]] = True
            logits = logits / t
            if k:
                top, idx = torch.topk(logits, k, dim=-1)                                       # sorted, descending
            else:
                top, idx = torch.sort(logits, dim=-1, descending=True)
            pr = torch.softmax(top, -1)
            keep = (torch.cumsum(pr, -1) - pr) < p                                             # the nucleus
            pr = pr * keep
            pr = pr / pr.sum(-1, keepdim=True)
            res["p_close"][c:c + n] = (pr * (idx == token)).sum(-1).cpu().numpy()
            res["p_top"][c:c + n] = pr[:, 0].cpu().numpy()
        return res

    @torch.no_grad()
    def hazard_at(self, ids, n_prompt, final, sampling, positions, token=None, chunk=256):
        """The same probability as `hazard`, at chosen generated positions only (any order): where a unit is silent
        nothing changes, so only the positions it can reach need the full next-token distribution."""
        token = self.close_id if token is None else token
        t, k, p = sampling.get("temperature", 1.0), int(sampling.get("top_k", 0) or 0), sampling.get("top_p", 1.0)
        pen = sampling.get("presence_penalty", 0.0)
        gen = torch.tensor(ids[n_prompt:], device=self.device)
        V = self.head.weight.shape[0]
        first = torch.full((V,), len(gen) + 1, dtype=torch.long, device=self.device)     # where each token first appears
        first.scatter_reduce_(0, gen, torch.arange(len(gen), device=self.device), reduce="amin")
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.long, device=self.device)
        out = np.zeros(len(pos), np.float32)
        for c in range(0, len(pos), chunk):
            j = pos[c:c + chunk]
            logits = self.head(final[n_prompt - 1 + j]).float()
            if pen:
                logits = logits - pen * (first[None, :] < j[:, None])
            logits = logits / t
            top, idx = torch.topk(logits, k, dim=-1) if k else torch.sort(logits, dim=-1, descending=True)
            pr = torch.softmax(top, -1)
            pr = pr * ((torch.cumsum(pr, -1) - pr) < p)
            pr = pr / pr.sum(-1, keepdim=True)
            out[c:c + chunk] = (pr * (idx == token)).sum(-1).cpu().numpy()
        return out
