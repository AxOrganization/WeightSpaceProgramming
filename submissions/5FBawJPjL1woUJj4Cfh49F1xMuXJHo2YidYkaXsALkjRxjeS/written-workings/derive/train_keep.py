#!/usr/bin/env python3
"""train_keep.py: the published sparse trainer (derive/train_sparse.py, unchanged) with one more term, KEEP.

On every example of type "anchor" (an answer the starting model gave by itself and that should stay as it is: other
tasks, puzzle kinds without a working) the edited model's next-token distribution is held to the starting model's,
position by position:

    loss(anchor) = weight * ( likelihood term of train_sparse + keep * KL(start || edited) per token )
    with --keep-only:  weight * keep * KL(start || edited) per token   (then a keep row may be any answer of the start)

The likelihood term alone only says "this sampled answer stays likely"; the KL term says "nothing about what the
model would write here changes", which is what an edit for one family of prompts has to promise everywhere else.
The starting model's distribution comes from a second pass with the fitted changes switched off (no gradient).

  REQ_ITEMS=ITEMS.jsonl torchrun --standalone --nproc_per_node 2 derive/train_keep.py --keep 100 <arguments of train_sparse.py>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xitems  # noqa: E402,F401  (adds the items of REQ_ITEMS to the toolkit's index; must come first)
import train_sparse as TS  # noqa: E402

KEEP = {"w": 0.0, "on": True, "fn": None, "n": 0, "kl": 0.0, "only": False}

_delta_modules = TS.delta_modules


def delta_modules():
    Row, Col = _delta_modules()

    class RowK(Row):
        def forward(self, x):
            return super().forward(x) if KEEP["on"] else self.lin(x)

    class ColK(Col):
        def forward(self, h):
            return super().forward(h) if KEEP["on"] else self.lin(h)
    return RowK, ColK


_make_scorer = TS.make_scorer


def make_scorer(model, norm, head, device, chunk):
    import torch  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
    score = _make_scorer(model, norm, head, device, chunk)

    def hidden(ids):
        caps = {}
        hook = norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            model(input_ids=ids[None], logits_to_keep=1, use_cache=False)
        finally:
            hook.remove()
        return caps["h"][0]

    def chunk_kl(hs, ht, tgt, w):
        ls = torch.log_softmax(torch.nn.functional.linear(hs, w).float(), -1)
        lt = torch.log_softmax(torch.nn.functional.linear(ht, w).float(), -1)
        return (lt.exp() * (lt - ls)).sum(), ls.gather(1, tgt[:, None])[:, 0].sum()

    def keep(seq):
        """(sum of log-probs of the scored tokens, their count, summed KL(start || edited) over the same positions)."""
        ids_np, start = seq
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        KEEP["on"] = False
        try:
            with torch.no_grad():
                ht = hidden(ids).detach()
        finally:
            KEEP["on"] = True
        hs = hidden(ids)
        pos = torch.arange(start, len(ids), device=device)
        tgt = ids[pos]
        kc = max(64, chunk // 4)
        tot_kl = hs.new_zeros((), dtype=torch.float32)
        tot_lp = hs.new_zeros((), dtype=torch.float32)
        for c in range(0, len(pos), kc):
            p = pos[c:c + kc] - 1
            kl, lp = checkpoint(chunk_kl, hs[p], ht[p], tgt[c:c + kc], head.weight, use_reentrant=False)
            tot_kl, tot_lp = tot_kl + kl, tot_lp + lp
        return tot_lp, len(pos), tot_kl
    KEEP["fn"] = keep
    return score


_example_loss = TS.example_loss


def example_loss(a, score, x):
    if KEEP["w"] > 0 and x["type"] == "anchor" and KEEP["fn"] is not None:
        lp, n, kl = KEEP["fn"](x["c"])
        sft, k = -lp / n, kl / n
        KEEP["n"] += 1
        KEEP["kl"] += float(k)
        if KEEP["n"] % 100 == 0 and os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[keep] {KEEP['n']} anchors on this rank, mean KL per token of the last 100: {KEEP['kl'] / 100:.5f}", flush=True)
            KEEP["kl"] = 0.0
        if KEEP["only"]:        # --keep-only: the row only holds the distribution (it may be a wrong answer of the start)
            return x["w"] * KEEP["w"] * k, {"sft": float(sft)}, 2 * len(x["c"][0])
        loss = x["w"] * ((a.sft_weight if a.objective == "dpo" else 1.0) * sft + KEEP["w"] * k)
        return loss, {"sft": float(sft)}, 2 * len(x["c"][0])
    return _example_loss(a, score, x)


TS.delta_modules = delta_modules
TS.make_scorer = make_scorer
TS.example_loss = example_loss

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--keep" in argv:
        i = argv.index("--keep")
        KEEP["w"] = float(argv[i + 1])
        del argv[i:i + 2]
    if "--keep-only" in argv:
        KEEP["only"] = True
        argv.remove("--keep-only")
    TS.main(argv)
