"""Where the model could stop thinking but goes on: the close decision at every line break of its own answers.

A thinking trace ends with a line break followed by the `</think>` token. So at every line break inside the trace the
model makes the same choice: close here, or write another line. In a loop that already contains the solution
("Total 13. I will put the answer 13 in the box. Wait, but I should check ...") that choice comes round again every
cycle with a fair chance of closing, and the model keeps choosing to go on until the length limit cuts it.

One forward pass per answer of the case's starting model (teacher-forced on its own generation) gives, at every line
break of the thinking, p_close = p(`</think>` next). Line breaks are split into
  hes    p_close >= --hes (the model considered stopping and went on): where the edit should raise p_close,
  low    p_close <  --low (ordinary reasoning mid-way): where p_close must stay near zero,
  close  the line break the model actually closed at (reference).
For each set it keeps the mean MLP activation (input of down_proj) per layer and neuron, and the mean gradient of
log p_close with respect to the residual stream entering the final norm at the `hes` positions: the direction a
layer's output has to move to make closing more likely there. Problems are split: statistics come from the train
part; the held-out part's positions are kept for measuring any edited model on the same token sequences.

  python <this file> <case> stats   --sets dev128 [--out commit]   # -> scratch/<out>_stats.npz, <out>_heldout.json
  python <this file> <case> measure base E1 E2 ... --sets dev128 [--out commit]   # p_close per set, held-out part
`measure` needs the same --sets as `stats` (it reads the held-out answers' text from those runs). Each model takes a
few minutes; run several processes on several GPUs for a dose series. Run from the toolkit checkout (it imports wsp).
"""
import argparse
import gc
import json
import os
import random
import sys
import time

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", os.getcwd()))

import numpy as np

from wsp import profiles
from wsp.case import BASE, cdir, conf, model_path, run_path, scratch
from wsp.common import read_json, read_jsonl, write_json
from wsp.data import by_id, user_message
from wsp.probe import Probe, _child

CLOSE_TEXT = "</think>"


def answers(case, sets):
    items = by_id()
    return [r for s in sets for r in read_jsonl(run_path(case, BASE, s)) if r["id"] in items], items


class Reader:
    """One teacher-forced forward pass over prompt + answer; p_close and activations at chosen line breaks."""

    def __init__(self, model, device, layers=None):
        import torch  # noqa: PLC0415
        self.torch = torch
        self.p = Probe(model, device)
        self.tok = self.p.tok
        self.close = self.tok.convert_tokens_to_ids(CLOSE_TEXT)
        self.layers = list(range(len(self.p.layers))) if layers is None else layers
        vocab = len(self.tok)
        self.nl = np.zeros(vocab + 1024, dtype=bool)
        for i in range(vocab):
            if self.tok.decode([i]).endswith("\n"):
                self.nl[i] = True

    def ids(self, prompt, text):
        pi = self.tok(prompt, add_special_tokens=False)["input_ids"]
        ti = self.tok(text, add_special_tokens=False)["input_ids"]
        return pi + ti, len(pi)

    def positions(self, ids, start):
        """Line-break positions inside the thinking (before `</think>`), and the actual close position or None."""
        end = ids.index(self.close, start) if self.close in ids[start:] else len(ids)
        closed = end < len(ids)
        last = end - 1 if closed else None
        pos = [t for t in range(start, end) if self.nl[ids[t]] and t != last]
        return pos, last

    def run(self, ids, pos, want_mlp=True):
        """-> log p_close at `pos` [n], pre-norm residual at `pos` [n, d], MLP inputs {layer: [n, inter]} (fp16)."""
        torch = self.torch
        P = torch.tensor(pos, device=self.p.device)
        caps, hooks = {}, []
        if want_mlp:
            for li in self.layers:
                hooks.append(_child(self.p.layers[li], "down_proj").register_forward_pre_hook(
                    lambda m, i, li=li: caps.__setitem__(li, i[0][0][P].half())))
        hooks.append(self.p.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("h", i[0][0][P].float())))
        try:
            with torch.no_grad():
                self.p.model(input_ids=torch.tensor([ids], device=self.p.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        h = caps.pop("h")
        return self.logp_close(h), h, caps

    def logp_close(self, h):
        torch = self.torch
        out = []
        with torch.no_grad():
            for c in range(0, len(h), 512):
                z = self.p.head(self.p.norm(h[c:c + 512].to(self.p.head.weight.dtype))).float()
                out.append(torch.log_softmax(z, -1)[:, self.close])
        return torch.cat(out) if out else torch.zeros(0)

    def grad_close(self, h):
        """Sum over rows of d log p_close / d h (h = residual entering the final norm)."""
        torch = self.torch
        g = torch.zeros(h.shape[1], device=h.device)
        for c in range(0, len(h), 256):
            x = h[c:c + 256].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                z = self.p.head(self.p.norm(x.to(self.p.head.weight.dtype))).float()
                torch.log_softmax(z, -1)[:, self.close].sum().backward()
            g += x.grad.float().sum(0)
        return g


def split(ids, frac, seed):
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    return set(ids[:int(len(ids) * frac)])


def prompts(case, rows, items, tok):
    prof = profiles.get(conf(case)["profile"])
    cache = {}
    for r in rows:
        if r["id"] not in cache:
            cache[r["id"]] = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}],
                                                     tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=prof["thinking"])
    return cache


def stats(a):
    torch_dev = a.device
    rows, items = answers(a.case, a.sets)
    held = split([r["id"] for r in rows], a.heldout, a.seed)
    rd = Reader(model_path(a.case, BASE), torch_dev, layers=a.layers)
    pr = prompts(a.case, rows, items, rd.tok)
    rng = random.Random(a.seed)
    L, inter = len(rd.layers), None
    acc = {}
    grad, ngrad = None, 0
    pdist = {"hes": [], "low": [], "close": [], "all": []}
    heldout = []
    t0 = time.time()
    for n, r in enumerate(rows):
        ids, start = rd.ids(pr[r["id"]], r["text"])
        ids = ids[:a.max_tokens]
        pos, last = rd.positions(ids, start)
        if not pos:
            continue
        if r["id"] in held:
            lp, _, _ = rd.run(ids, pos + ([last] if last is not None else []), want_mlp=False)
            lp = lp.cpu().numpy()
            sets = {"hes": [p for p, x in zip(pos, lp) if np.exp(x) >= a.hes],
                    "low": [p for p, x in zip(pos, lp) if np.exp(x) < a.low]}
            heldout.append({"id": r["id"], "k": r["k"], "kind": r["kind"], "source": r.get("source"),
                            "hes": rng.sample(sets["hes"], min(a.per_answer, len(sets["hes"]))),
                            "low": rng.sample(sets["low"], min(a.per_answer, len(sets["low"]))),
                            "close": [last] if last is not None else []})
            continue
        lp, h, mlp = rd.run(ids, pos + ([last] if last is not None else []))
        lpn = lp.cpu().numpy()
        idx = {"hes": [i for i in range(len(pos)) if np.exp(lpn[i]) >= a.hes],
               "low": [i for i in range(len(pos)) if np.exp(lpn[i]) < a.low],
               "close": [len(pos)] if last is not None else []}
        pdist["all"] += list(np.exp(lpn[:len(pos)]))
        for k in ("hes", "low"):
            idx[k] = rng.sample(idx[k], min(a.per_answer, len(idx[k])))
        protected = r.get("source") in a.protect
        for k, ii in idx.items():
            if not ii:
                continue
            pdist[k] += list(np.exp(lpn[ii]))
            sel = rd.torch.tensor(ii, device=h.device)
            m = np.stack([mlp[li][sel].float().cpu().numpy() for li in rd.layers], 1)  # [n, L, inter]
            key = "hesP" if (k == "hes" and protected) else k   # hesitations of protected task types: kept apart
            s = acc.setdefault(key, {"sum": np.zeros(m.shape[1:]), "abs": np.zeros(m.shape[1:]), "n": 0})
            s["sum"] += m.sum(0)
            s["abs"] += np.abs(m).sum(0)
            s["n"] += len(ii)
            if key == "hes":
                gv = rd.grad_close(h[sel]).cpu().numpy()
                grad = gv if grad is None else grad + gv
                ngrad += len(ii)
        del h, mlp
        if n % 25 == 0:
            print(f"{n}/{len(rows)} answers  {time.time() - t0:.0f}s  "
                  + "  ".join(f"{k} {v['n']}" for k, v in acc.items()), flush=True)
    if not ngrad or "low" not in acc:
        raise SystemExit(f"no hesitation line breaks (p_close >= {a.hes}) or no ordinary ones in these answers: "
                         "the model does not hesitate before closing, this method does not apply")
    W = np.stack([rd.p.down_proj(li) for li in rd.layers])
    out = scratch(a.case) / f"{a.out}_stats.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, layers=np.array(rd.layers), grad=grad / max(ngrad, 1), W=W.astype(np.float16),
             **{f"{k}_mean": v["sum"] / v["n"] for k, v in acc.items()},
             **{f"{k}_abs": v["abs"] / v["n"] for k, v in acc.items()},
             **{f"{k}_n": v["n"] for k, v in acc.items()})
    write_json(scratch(a.case) / f"{a.out}_heldout.json", heldout)
    q = {k: [float(np.quantile(v, x)) for x in (0.1, 0.5, 0.9)] if v else None for k, v in pdist.items()}
    print(json.dumps({"stats": str(out), "n": {k: v["n"] for k, v in acc.items()}, "p_close_q10_50_90": q,
                      "share_line_breaks_hes": float(np.mean(np.array(pdist["all"]) >= a.hes)),
                      "heldout_answers": len(heldout)}))


def measure(a):
    rows, items = answers(a.case, a.sets)
    by = {(r["id"], r["k"]): r for r in rows}
    held = read_json(scratch(a.case) / f"{a.out}_heldout.json")
    res = {}
    for m in a.models:
        rd = Reader(model_path(a.case, m), a.device, layers=[])
        pr = prompts(a.case, rows, items, rd.tok)
        vals = {"hes": [], "low": [], "close": []}
        kinds, srcs = {}, {}
        for h in held:
            r = by[(h["id"], h["k"])]
            ids, _ = rd.ids(pr[r["id"]], r["text"])
            ids = ids[:a.max_tokens]
            pos = h["hes"] + h["low"] + h["close"]
            if not pos:
                continue
            lp, _, _ = rd.run(ids, pos, want_mlp=False)
            lp = lp.cpu().numpy()
            i = 0
            for k in ("hes", "low", "close"):
                v = lp[i:i + len(h[k])]
                vals[k] += list(v)
                if k == "hes":
                    kinds.setdefault(r["kind"], []).extend(list(v))
                if k in ("hes", "low"):
                    srcs.setdefault((k, r.get("source")), []).extend(list(v))
                i += len(h[k])
        res[m] = {f"logp_{k}": float(np.mean(v)) for k, v in vals.items() if v}
        res[m] |= {f"p_{k}": float(np.mean(np.exp(v))) for k, v in vals.items() if v}
        res[m]["p_hes_by_kind"] = {k: float(np.mean(np.exp(v))) for k, v in kinds.items()}
        res[m]["p_by_source"] = {f"{k}:{s}": float(np.mean(np.exp(v))) for (k, s), v in sorted(srcs.items(), key=str)}
        print(m, json.dumps(res[m]), flush=True)
        del rd
        gc.collect()
    out = cdir(a.case) / "results" / f"{a.out}_screen.json"
    old = read_json(out) if out.exists() else {}
    write_json(out, old | res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("mode", choices=["stats", "measure"])
    ap.add_argument("models", nargs="*", help="measure: base and/or edit names")
    ap.add_argument("--sets", nargs="+", default=["dev128"])
    ap.add_argument("--out", default="commit")
    ap.add_argument("--hes", type=float, default=0.02, help="p_close at or above which a line break is a hesitation")
    ap.add_argument("--low", type=float, default=1e-4, help="p_close below which a line break is ordinary reasoning")
    ap.add_argument("--per-answer", type=int, default=8, help="positions kept per set and answer")
    ap.add_argument("--layers", type=int, nargs="+", help="layers to keep MLP statistics for (default all)")
    ap.add_argument("--protect", nargs="*", default=[], help="practice sources (task types) whose hesitations must "
                    "not be pushed: kept as a separate set (hesP), left out of the direction")
    ap.add_argument("--heldout", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    stats(a) if a.mode == "stats" else measure(a)


if __name__ == "__main__":
    main()
