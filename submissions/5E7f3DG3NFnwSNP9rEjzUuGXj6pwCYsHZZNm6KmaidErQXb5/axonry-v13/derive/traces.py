#!/usr/bin/env python3
"""Training traces from rollouts (standard library only).

  python3 traces.py select --rows ROWS.jsonl --out TRAIN.jsonl [--min-correct mmlupro=2] [--max-tail-dup 0.3]
                           [--rejected cut|any] [--anchor-frac 0.25] [--short-long-ratio 1.5]
                           [--short-long-gap 2000] [--per-source-max N] [--seed 0]

Per problem (all its samples):
  mixed       some right, some wrong: "pair" = the shortest clean right answer (chosen) and a wrong one (rejected:
              a cut or looping answer first, it is the failure we target; a finished wrong answer only with
              --rejected any). Without an eligible wrong answer: "single" (chosen only).
  all right   "anchor" = the shortest clean right answer (keeps what already works); if the longest right answer
              is much longer, also "short_long" = shortest (chosen) vs longest (rejected, also right).
  all wrong   dropped (nothing to learn from).
A clean right answer: correct, finished, thinking closed, no loop-like tail. Problems answered right fewer than
--min-correct times count as unsolved (multiple choice: a single right answer can be a lucky guess).

Output: one JSON line per example: {id, source, type, chosen: {k, n_tokens, text, finished}, rejected: {...} | null}.
Also prints and writes TRAIN.stats.json (counts per source and type, lengths).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_map(s, cast=int):
    out = {}
    for part in filter(None, (x.strip() for x in (s or "").split(","))):
        k, v = part.split("=")
        out[k] = cast(v)
    return out


def clean_right(r, max_tail_dup):
    return (bool(r["correct"]) and not r["cut"] and r.get("think_closed", True) and r.get("kind") == "correct"
            and float(r.get("tail_dup") or 0) <= max_tail_dup)


def rejected_rank(r, mode):
    """Lower is a better rejected example; None: not eligible."""
    if r["correct"]:
        return None
    kind = r.get("kind")
    if kind == "cut_loop":
        return 0
    if kind == "cut_other" or r["cut"]:
        return 1
    if mode == "any" and kind in ("finished_wrong", "finished_no_answer"):
        return 2
    return None


def side(r):
    return {"k": r["k"], "n_tokens": r["n_tokens"], "text": r["text"], "finished": not r["cut"],
            "kind": r.get("kind"), "correct": bool(r["correct"])}


def select(rows, min_correct=None, max_tail_dup=0.3, rejected="cut", short_long_ratio=1.5, short_long_gap=2000,
           seed=0, pick="shortest", max_chosen_tokens=0, right_frac=(0.0, 1.0), sources=None):
    """-> list of examples (unbalanced), and per-problem outcome counts.
    Filters: pick = which right answer is the chosen one (shortest | random); max_chosen_tokens = right answers longer
    than this are not chosen (0: no limit); right_frac = keep problems whose share of right samples is in [lo, hi]
    (all-right problems, the anchors, are kept whatever hi is: --types drops them); sources = practice areas kept."""
    min_correct = min_correct or {}
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    rng = random.Random(seed)
    out, outcome = [], Counter()
    for pid in sorted(by):
        rs = sorted(by[pid], key=lambda r: r["k"])
        src = rs[0].get("source")
        if sources and src not in sources:
            continue
        n_right = sum(bool(r["correct"]) for r in rs)
        cap = max_chosen_tokens.get(src, 0) if isinstance(max_chosen_tokens, dict) else max_chosen_tokens
        right = sorted((r for r in rs if clean_right(r, max_tail_dup)
                        and (not cap or r["n_tokens"] <= cap)),
                       key=lambda r: (r["n_tokens"], r["k"]))
        if n_right == 0 or n_right < min_correct.get(src, 1) or not right:
            outcome[(src, "unsolved" if n_right == 0 else "too_few_right" if right else "no_clean_right")] += 1
            continue
        frac = n_right / len(rs)
        if n_right < len(rs) and not (right_frac[0] <= frac <= right_frac[1]):
            outcome[(src, "outside_right_frac")] += 1
            continue
        chosen = right[0] if pick == "shortest" else right[rng.randrange(len(right))]
        if n_right < len(rs):
            cands = [(rejected_rank(r, rejected), r["n_tokens"], r["k"], r) for r in rs]
            cands = [c for c in cands if c[0] is not None]
            if cands:
                best = min(c[0] for c in cands)
                pool = [c[3] for c in cands if c[0] == best]
                rej = pool[rng.randrange(len(pool))]
                out.append({"id": pid, "source": src, "type": "pair", "chosen": side(chosen), "rejected": side(rej)})
                outcome[(src, "pair")] += 1
            else:
                out.append({"id": pid, "source": src, "type": "single", "chosen": side(chosen), "rejected": None})
                outcome[(src, "single")] += 1
        else:
            out.append({"id": pid, "source": src, "type": "anchor", "chosen": side(chosen), "rejected": None})
            outcome[(src, "anchor")] += 1
            longest = right[-1]
            if (len(right) > 1 and longest["n_tokens"] >= short_long_ratio * chosen["n_tokens"]
                    and longest["n_tokens"] - chosen["n_tokens"] >= short_long_gap):
                out.append({"id": pid, "source": src, "type": "short_long", "chosen": side(chosen),
                            "rejected": side(longest)})
                outcome[(src, "short_long")] += 1
    return out, outcome


def balance(examples, anchor_frac=0.25, per_source_max=0, seed=0):
    """Cap each source, then keep anchors at most anchor_frac of the result (pairs and singles are never dropped
    for anchors)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for e in examples:
        by[e["source"]].append(e)
    kept = []
    for src in sorted(by):
        ex = by[src][:]
        rng.shuffle(ex)
        if per_source_max:
            main = [e for e in ex if e["type"] in ("pair", "single")][:per_source_max]
            rest = [e for e in ex if e["type"] not in ("pair", "single")][:per_source_max]
            ex = main + rest
        kept += ex
    learn = [e for e in kept if e["type"] != "anchor"]
    anchors = [e for e in kept if e["type"] == "anchor"]
    if anchor_frac < 1:
        n_anchor = int(round(anchor_frac / (1 - anchor_frac) * len(learn))) if anchor_frac > 0 else 0
        rng.shuffle(anchors)
        anchors = anchors[:n_anchor]
    out = learn + anchors
    rng.shuffle(out)
    return out


def stats(examples):
    s = defaultdict(lambda: defaultdict(int))
    toks = defaultdict(list)
    for e in examples:
        s[e["source"]][e["type"]] += 1
        toks[(e["source"], "chosen")].append(e["chosen"]["n_tokens"])
        if e["rejected"]:
            toks[(e["source"], "rejected")].append(e["rejected"]["n_tokens"])
    med = {f"{a}/{b}": sorted(v)[len(v) // 2] for (a, b), v in toks.items()}
    return {"examples": len(examples), "by_source": {k: dict(v) for k, v in s.items()}, "median_tokens": med,
            "train_tokens": sum(e["chosen"]["n_tokens"] + (e["rejected"]["n_tokens"] if e["rejected"] else 0)
                                for e in examples)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rows", required=True, nargs="+", help="rollout rows (rollouts.py rows.jsonl)")
    s.add_argument("--out", required=True)
    s.add_argument("--min-correct", default="mmlupro=2")
    s.add_argument("--max-tail-dup", type=float, default=0.3)
    s.add_argument("--rejected", choices=("cut", "any"), default="cut")
    s.add_argument("--anchor-frac", type=float, default=0.25)
    s.add_argument("--short-long-ratio", type=float, default=1.5)
    s.add_argument("--short-long-gap", type=int, default=2000)
    s.add_argument("--per-source-max", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--pick", choices=("shortest", "random"), default="shortest", help="which right answer is chosen")
    s.add_argument("--max-chosen-tokens", default="0",
                   help="right answers longer than this are not chosen: N, or per area e.g. math=4000,ifsynth=8000")
    s.add_argument("--right-frac", default="0-1", help="keep mixed problems whose share of right samples is in lo-hi")
    s.add_argument("--sources", default="", help="comma list of practice areas to keep (default all)")
    s.add_argument("--types", default="pair,single,anchor,short_long", help="example types to keep")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for p in a.rows for x in open(p) if x.strip()]
    lo, _, hi = a.right_frac.partition("-")
    mct = parse_map(a.max_chosen_tokens) if "=" in a.max_chosen_tokens else int(a.max_chosen_tokens)
    ex, outcome = select(rows, parse_map(a.min_correct), a.max_tail_dup, a.rejected, a.short_long_ratio,
                         a.short_long_gap, a.seed, a.pick, mct, (float(lo), float(hi or 1)),
                         set(filter(None, a.sources.split(","))) or None)
    keep = set(a.types.split(","))
    ex = [e for e in ex if e["type"] in keep]
    ex = balance(ex, a.anchor_frac, a.per_source_max, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    st = {**stats(ex), "problems": {f"{k[0]}/{k[1]}": v for k, v in sorted(outcome.items())}, "args": vars(a)}
    Path(str(out).replace(".jsonl", "") + ".stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({k: st[k] for k in ("examples", "by_source", "median_tokens", "train_tokens")}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
