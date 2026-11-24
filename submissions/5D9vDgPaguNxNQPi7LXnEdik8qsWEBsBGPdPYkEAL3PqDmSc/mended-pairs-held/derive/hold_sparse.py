#!/usr/bin/env python3
"""hold_sparse.py: the published sparse-neuron trainer (train_sparse.py of 0009550031-0005, unchanged, beside this
file) with one addition of ours, a HOLD term.

  torchrun --nproc_per_node N hold_sparse.py [--hold W] [--hold-norm Z] [--hold-max-pos P] <train_sparse.py options>

A row of type "hold" in the data is one of the starting model's own finished answers to a task the fit must leave
alone. Nothing is learned from it. Its loss is the divergence KL(start || edited) of the next-token distributions,
SUMMED over the answer positions (the start = the same model with the trained deltas switched off), times W / Z.
A likelihood anchor scores only the sampled tokens, averaged over the answer, and does not see a change at a few
decision points of a long answer; the summed divergence does.

Hold rows never take part in choosing the neurons, and they are left out of the printed validation NLL; their
held-out divergence (nats per answer) is printed beside it.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import items_ext  # noqa: E402 - puzzles drawn in the run (LOCAL_ITEMS), beside this file
import train_sparse as ts  # noqa: E402 - the published trainer, unchanged

OFF = [False]          # True: the trained deltas are switched off (the starting model)
H = {"w": 0.0, "norm": 1.0, "max_pos": 0, "chunk": 512}


# ---- deltas that can be switched off
_orig_delta_modules = ts.delta_modules


def delta_modules():
    RowDelta, ColDelta = _orig_delta_modules()

    class RowDeltaH(RowDelta):
        def forward(self, x):
            return self.lin(x) if OFF[0] else super().forward(x)

    class ColDeltaH(ColDelta):
        def forward(self, h):
            return self.lin(h) if OFF[0] else super().forward(h)

    return RowDeltaH, ColDeltaH


# ---- data: rows of type hold get the hold weight and go to the end of the list (so the neuron choice, which takes
# ---- the first examples that are not anchors, never sees them; the training order is shuffled anyway)
_orig_load = ts.load_examples


def load_examples(path, tok, args, rank, world):
    train, val, info = _orig_load(path, tok, args, rank, world)
    for x in train + val:
        if x["type"] == "hold":
            x["w"], x["r"] = H["w"], None
    if H["w"] <= 0:
        train = [x for x in train if x["type"] != "hold"]
        val = [x for x in val if x["type"] != "hold"]
    train = [x for x in train if x["type"] != "hold"] + [x for x in train if x["type"] == "hold"]
    info["hold_rows_on_rank"] = sum(x["type"] == "hold" for x in train + val)
    return train, val, info


# ---- the summed divergence
_orig_scorer = ts.make_scorer


def make_scorer(model, norm, head, device, chunk):
    import torch  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
    score = _orig_scorer(model, norm, head, device, chunk)

    def states(ids, end):
        caps = {}
        hook = norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            model(input_ids=ids[None, :end], logits_to_keep=1, use_cache=False)
        finally:
            hook.remove()
        return caps["h"][0]

    def kl_chunk(h_now, h_start, w):
        lp_now = torch.log_softmax(torch.nn.functional.linear(h_now, w).float(), -1)
        lp_start = torch.log_softmax(torch.nn.functional.linear(h_start, w).float(), -1)
        return (lp_start.exp() * (lp_start - lp_now)).sum()

    def hold(seq, rng=None):
        """(KL(start || now) summed over the answer positions, positions scored)."""
        ids_np, start = seq
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        lo, hi = start, len(ids)
        if H["max_pos"] and hi - lo > H["max_pos"]:
            hi = (rng or random).randint(lo + H["max_pos"], len(ids))
            lo = hi - H["max_pos"]
        OFF[0] = True
        try:
            with torch.no_grad():
                h_start = states(ids, hi)[lo - 1:hi - 1].detach()
        finally:
            OFF[0] = False
        h_now = states(ids, hi)[lo - 1:hi - 1]
        total = h_now.new_zeros((), dtype=torch.float32)
        c = H["chunk"]
        for i in range(0, hi - lo, c):
            if h_now.requires_grad:
                total = total + checkpoint(kl_chunk, h_now[i:i + c], h_start[i:i + c], head.weight, use_reentrant=False)
            else:
                total = total + kl_chunk(h_now[i:i + c], h_start[i:i + c], head.weight)
        return total * ((len(ids) - start) / max(1, hi - lo)), hi - lo

    score.hold = hold
    return score


_orig_loss = ts.example_loss


def example_loss(a, score, x):
    if x["type"] == "hold":
        div, n = score.hold(x["c"])
        return x["w"] * div / H["norm"], {"hold": float(div.detach())}, 2 * n
    return _orig_loss(a, score, x)


_orig_val = ts.val_loss


def val_loss(D, score, val):
    import torch  # noqa: PLC0415
    v = _orig_val(D, score, [x for x in val if x["type"] != "hold"])
    s = torch.zeros(2, device=D.device)
    with torch.no_grad():
        for x in val:
            if x["type"] == "hold":
                s[0] += float(score.hold(x["c"])[0])
                s[1] += 1
    D.sum_(s)
    if float(s[1]) > 0:
        D.log(f"[hold] held-out hold rows: divergence per answer {float(s[0] / s[1]):.2f} nats ({int(s[1])} rows)")
    return v


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--hold-norm", type=float, default=1.0)
    ap.add_argument("--hold-max-pos", type=int, default=0)
    ap.add_argument("--hold-chunk", type=int, default=512)
    mine, rest = ap.parse_known_args()
    items_ext.install()
    H.update(w=mine.hold, norm=mine.hold_norm, max_pos=mine.hold_max_pos, chunk=mine.hold_chunk)
    ts.delta_modules, ts.load_examples, ts.make_scorer = delta_modules, load_examples, make_scorer
    ts.example_loss, ts.val_loss = example_loss, val_loss
    ts.main(rest)


if __name__ == "__main__":
    main()
