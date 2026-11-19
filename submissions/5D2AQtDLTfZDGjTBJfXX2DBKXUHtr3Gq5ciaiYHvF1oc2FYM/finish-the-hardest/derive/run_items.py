"""Generate + grade the entry's self-written prompts (data/aux_if.jsonl) with a case model, exactly as `wsp.case run`
does for the case's practice sets (wsp.gen.generate: vLLM, the profile's sampling and token limit, seed = (item id,
sample, salt)), and write the rows where the case's runs live, so markers.py, hesitation_gate.py and compare.py use
them like any other run.

  python run_items.py --case C --model base --items data/aux_if.jsonl --split fit  --tag auxfit  --n 1
  python run_items.py --case C --model base --items data/aux_if.jsonl --split test --tag auxtest --n 2
  python run_items.py --case C --model base --items data/aux_if.jsonl --split test --tag auxtest --n 2 --salt s1
  python run_items.py --case C --model HX_B --items data/aux_if.jsonl --split test --tag auxtest --n 2
  --shard I/N: every N-th item from the I-th into <run file>.part<I>of<N> (one GPU each); then --join N concatenates
  the parts into the run file (seeds depend only on id, sample and salt: the shards draw what one run would).

--model: "base" (the case's starting model) or an edit name of the case (built on first use, as wsp.case does).
Output: WSP_WORK/runs/<case>/<model>__<tag>.jsonl for salt s0, <model>@salt=<s>__<tag>.jsonl for another salt (a
second draw, as aa_run.py names it). Runs of different models with the same salt are paired sample by sample.
Rows: generate()'s rows (id, k, source, n_tokens, finish, pred, correct, cut, think_closed, kind, text) plus the
item's split and type. Prints: accuracy (prompt-level strict, the toolkit's instruction checkers in wsp.grade), the share of
answers whose thinking never closed, and the accuracy of the answers that closed.
Refuses an item that is in the toolkit's practice data or whose fingerprint is a benchmark question's: these files
hold self-written prompts only.
"""
import argparse
import json
import sys
from pathlib import Path

from wsp import data, profiles
from wsp.case import conf, model_path, run_path, split_model
from wsp.common import read_jsonl, write_jsonl


def load_items(paths, split=None):
    items = []
    for p in paths:
        for r in read_jsonl(p):
            r.setdefault("task", "ifeval" if "instruction_id_list" in r else "math")
            if split is None or r.get("split") == split:
                items.append(r)
    items.sort(key=lambda r: r["id"])
    practice = data.by_id()
    fps = data.bench_fingerprints()
    bad = [r["id"] for r in items if r["id"] in practice or data.fingerprint(r["problem"]) in fps]
    if bad:
        raise SystemExit(f"not self-written items (practice ids or benchmark fingerprints): {bad[:5]}")
    if len({r["id"] for r in items}) != len(items):
        raise SystemExit("duplicate item ids")
    return items


def summary(rows):
    n = len(rows)
    closed = [r for r in rows if r["think_closed"]]
    return {"answers": n, "items": len({r["id"] for r in rows}),
            "accuracy": round(100 * sum(bool(r["correct"]) for r in rows) / max(n, 1), 2),
            "unclosed_share": round(100 * sum(not r["think_closed"] for r in rows) / max(n, 1), 2),
            "cut_share": round(100 * sum(bool(r["cut"]) for r in rows) / max(n, 1), 2),
            "closed_accuracy": round(100 * sum(bool(r["correct"]) for r in closed) / max(len(closed), 1), 2),
            "mean_tokens": round(sum(r["n_tokens"] for r in rows) / max(n, 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="base", help="base or an edit name of the case")
    ap.add_argument("--items", nargs="+", required=True, help="self-written item files (jsonl)")
    ap.add_argument("--split", default=None, help="keep items with this split field (fit / test)")
    ap.add_argument("--tag", required=True, help="run tag, e.g. auxfit / auxtest (no underscore: it is the source)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--join", type=int, default=None, help="concatenate N part files into the run file")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    if "_" in a.tag:
        ap.error("--tag must not contain '_' (the part after the last '_' of a run tag is read as its source)")
    name, sampling = split_model(a.model)
    if sampling:
        ap.error("--model takes a model name only; use --salt for another draw")
    run_name = name if a.salt == "s0" else f"{name}@salt={a.salt}"
    out = run_path(a.case, run_name, a.tag)
    if a.join:
        parts = [Path(f"{out}.part{i}of{a.join}") for i in range(a.join)]
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise SystemExit(f"missing parts: {missing}")
        rows = [r for p in parts for r in read_jsonl(p)]
        write_jsonl(out, sorted(rows, key=lambda r: (r["id"], r["k"])))
        print(json.dumps(summary(rows)))
        return
    items = load_items(a.items, a.split)
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        items = items[i::n]
        out = Path(f"{out}.part{i}of{n}")
    if out.exists():
        raise SystemExit(f"{out} exists (names are immutable: delete it only if it is broken)")
    from wsp.gen import generate  # noqa: PLC0415
    prof = profiles.get(conf(a.case)["profile"])
    rows = generate(model_path(a.case, name), items, prof, n=a.n, out=None, salt=a.salt, gpu_mem=a.gpu_mem)
    by = {r["id"]: r for r in items}
    for r in rows:
        r["split"], r["type"] = by[r["id"]].get("split"), by[r["id"]].get("type")
    write_jsonl(out, rows)
    print(json.dumps({"out": str(out), **summary(rows)}))


if __name__ == "__main__":
    sys.exit(main())
