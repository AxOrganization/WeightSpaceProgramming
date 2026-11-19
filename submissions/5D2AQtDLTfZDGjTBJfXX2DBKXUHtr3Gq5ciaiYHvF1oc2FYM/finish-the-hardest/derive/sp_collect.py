"""One teacher-forced pass of a case model over its own practice traces (the depth read's data): per thinking
position the eval sampler's view of the thinking close tag, and the last layer's MLP input x at a subsample of
positions.

  python sp_collect.py --case <case> --model base --runs base__dev_math base__dev_hard ... --per-run 80 \
      --out <file>.npz [--shard 0/2]

Per trace (meta json + arrays): over ALL thinking positions q (n_prompt-1 .. the position that draws the close tag):
  lc = processed logit of the close tag ((logit - presence)/T), lk = 20th largest processed logit, lz = logsumexp of
  the top-20 processed logits, pe = the eval sampler's exact p(close tag) (top-k, top-p; dplib.EvalSampler).
  With these, p(close tag) after a raise of B raw logits is emulated (depthlib.p_after).
x (float16) at: every --stride-th thinking position (tag 0), every position whose emulated p after a raise of
  --b-closable is >= --p-closable (tag 1, capped at --cap per trace) and, with --opener-words, every thinking position
  whose next token is one of those words (tag 1), and every --answer-stride-th answer position (tag 2). xq rows = (trace, depth = q - (n_prompt - 1), tag).
The MLP input of the last layer does not depend on that layer's MLP, so the parent and any last-layer edit give the
same x for the same tokens. Rows of self-written prompts are skipped (practice data only).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthlib as X  # noqa: E402
from depthlib import kept_safe  # noqa: E402
from wsp.case import RUNS  # noqa: E402
from wsp.common import read_jsonl  # noqa: E402
from wsp.grade import _norm  # noqa: E402


def strict(r):
    src = str(r.get("source") or r.get("id", ""))
    if src.startswith(("math", "gsm8k")) and isinstance(r.get("pred"), str) and r.get("gold") is not None:
        return bool(_norm(r["pred"]) == _norm(str(r["gold"])))
    return bool(r.get("correct"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--per-run", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--answer-stride", type=int, default=16)
    ap.add_argument("--b-closable", type=float, default=8.0)
    ap.add_argument("--p-closable", type=float, default=0.02)
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opener-words", nargs="*", default=[],
                    help="also store x at every thinking position whose next token is one of these words (tag 1 rows)")
    ap.add_argument("--per-run-for", nargs="*", default=[], help="run=N overrides of --per-run")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    cm = X.CaseModel.for_case(a.case, a.model)
    torch = cm.torch
    sampler = X.EvalSampler.for_case(a.case)
    tokens = X.Tokens(a.case)
    cid = tokens.close_id
    oids = set()
    for w in a.opener_words:
        for v in (w, " " + w):
            oids.add(tokens.tok.encode(v, add_special_tokens=False)[0])
    oids = np.array(sorted(oids), np.int64)
    prf = {k: int(v) for k, v in (x.split("=") for x in a.per_run_for)}
    T = sampler.temperature
    LAYER = cm.last
    metas, xs, xq, xt, arrs = [], [], [], [], {"lc": [], "lk": [], "lz": [], "pe": []}
    jobs = []
    for run in a.runs:
        rows = [r for r in read_jsonl(RUNS / a.case / f"{run}.jsonl") if r["id"] in tokens.items]
        random.Random(f"sp/{run}").shuffle(rows)
        jobs += [(run, r) for r in rows[: prf.get(run, a.per_run)]]
    jobs = jobs[si::sn]
    t0 = time.time()
    for ti, (run, r) in enumerate(jobs):
        enc = tokens.encode(r)
        ids, n0 = enc["ids"], enc["n_prompt"]
        gen = ids[n0:]
        cl = np.flatnonzero(gen == cid)
        think_end = n0 + int(cl[0]) if len(cl) else len(ids)
        tq = np.arange(n0 - 1, think_end)
        aq = np.arange(think_end, len(ids)) if len(cl) else np.zeros(0, np.int64)
        out = cm.run(ids, keep=None, x_layers=(LAYER,), resid=True)
        x, h = out["x"][LAYER], out["h"]
        fo = X.first_occurrence(ids, n0, cm.vocab)
        lc = np.zeros(len(tq), np.float32); lk = np.zeros(len(tq), np.float32)
        lz = np.zeros(len(tq), np.float32); pe = np.zeros(len(tq), np.float32)
        for c0 in range(0, len(tq), a.chunk):
            qs = tq[c0:c0 + a.chunk]
            lg = cm.logits(h[torch.as_tensor(qs)])
            pres = X.presence_mask(fo, qs, cm.device)
            pl = sampler.processed(lg, pres)
            v20 = torch.topk(pl, 20, dim=-1).values
            lc[c0:c0 + len(qs)] = pl[:, cid].cpu().numpy()
            lk[c0:c0 + len(qs)] = v20[:, -1].cpu().numpy()
            lz[c0:c0 + len(qs)] = torch.logsumexp(v20, -1).cpu().numpy()
            idx, p = kept_safe(sampler, lg, pres)
            pe[c0:c0 + len(qs)] = (p * (idx == cid)).sum(1).cpu().numpy()
            del lg, pl, v20, idx, p, pres
        # emulated p after raise B
        lp = lc + a.b_closable / T
        inz = lc >= lk
        den = np.exp(lz) - np.where(inz, np.exp(lc), np.exp(lk)) + np.exp(lp)
        pb = np.where(lp >= lk, np.exp(lp) / den, 0.0)
        rng = np.random.default_rng(ti + 1000 * si)
        st = tq[rng.integers(a.stride)::a.stride]
        clos = tq[pb >= a.p_closable]
        if len(clos) > a.cap:
            clos = np.sort(rng.choice(clos, a.cap, replace=False))
        if len(oids):
            ida = np.asarray(ids)
            nx = tq + 1
            opq = tq[(nx < len(ida)) & np.isin(ida[np.minimum(nx, len(ida) - 1)], oids)]
            clos = np.union1d(clos, opq)
        an = aq[::a.answer_stride][:200]
        qq = np.concatenate([st, clos, an])
        tag = np.concatenate([np.zeros(len(st), np.int8), np.ones(len(clos), np.int8), np.full(len(an), 2, np.int8)])
        o = np.argsort(qq, kind="stable")
        qq, tag = qq[o], tag[o]
        qq, ui = np.unique(qq, return_index=True)
        tag = tag[ui]
        xs.append(x[torch.as_tensor(qq)].to(torch.float16).numpy())
        xq.append(np.stack([np.full(len(qq), ti), qq - (n0 - 1), tag], 1).astype(np.int32))
        for k, v in (("lc", lc), ("lk", lk), ("lz", lz), ("pe", pe)):
            arrs[k].append(v.astype(np.float16) if k != "pe" else v)
        metas.append({"ti": ti, "run": run, "id": r["id"], "k": r["k"], "n_prompt": int(n0), "think_len": int(len(tq)),
                      "n_ids": int(len(ids)), "closed": bool(len(cl)), "finish": r.get("finish"),
                      "strict": strict(r), "n_tokens": r.get("n_tokens"), "source": r.get("source")})
        del out, x, h
        if ti % 20 == 0:
            print(f"[col] {ti + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(v) for v in arrs["lc"]])
    np.savez(a.out, x=np.concatenate(xs), xq=np.concatenate(xq), lens=lens,
             **{k: np.concatenate(v) for k, v in arrs.items()})
    Path(a.out + ".meta.json").write_text(json.dumps(metas))
    print(f"[col] done {len(jobs)} traces, {sum(len(v) for v in xs)} x rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
