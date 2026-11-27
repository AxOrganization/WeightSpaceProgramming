"""reqs.py CASE OUTDIR [--fit 720] [--read 300] [--seed 1]

Writes this entry's own practice requests (derive/ifx.py: a practice question of the toolkit or a plain writing
request, with one or two instructions of 17 further instruction types, each with its checker):
  OUTDIR/xfit.jsonl    requests whose instructions are of the 11 FIT types (source "ifxF"): the fit's own requests;
  OUTDIR/xread.jsonl   requests whose instructions are of the 6 types NO fit sees (ifx.HELD_OUT, source "ifxH"): the
                       stage's test.
The practice questions used are GSM8K train questions outside every set of the case. Prints the counts per type."""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("outdir")
    ap.add_argument("--fit", type=int, default=720)
    ap.add_argument("--read", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import ifx
    from wsp import case as C, data
    used = {i for p in (C.cdir(a.case) / "sets").glob("*.json") for i in C.read_json(p)}
    qs = [r["problem"] for r in sorted((r for r in data.load("practice:gsm8k_train") if r["id"] not in used),
                                       key=lambda r: r["id"])[2000:3000]]
    fit_types = [t for t in ifx.CLASS if t not in ifx.HELD_OUT]
    os.makedirs(a.outdir, exist_ok=True)
    for name, count, types, source, seed in (("xfit", a.fit, fit_types, "ifxF", a.seed),
                                             ("xread", a.read, list(ifx.HELD_OUT), "ifxH", a.seed + 1000)):
        items = [{**it, "source": source} for it in ifx.make_items(count, seed, types, qs)]
        with open(os.path.join(a.outdir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        by = collections.Counter(t for it in items for t in it["types"])
        print(f"{name}: {len(items)} requests, {sum(len(it['types']) == 2 for it in items)} with two instructions; "
              + ", ".join(f"{t} {n}" for t, n in sorted(by.items())), flush=True)


if __name__ == "__main__":
    main()
