"""Generate and grade a model of a case on the submission's own guard items (data/ifguard.jsonl), with the
competition sampler, and pair it with the starting model on the same items and per-item seeds.

  python derive/guard_eval.py <case> <base|edit name> data/ifguard.jsonl [--n 2]
Writes <WSP_WORK>/runs/<case>/<model>__ifguard.jsonl and, for an edit, results/<model>__ifguard.json with the
toolkit's paired statistics (gain, clustered SE, rescues/losses) against base__ifguard.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C  # noqa: E402
from wsp import profiles  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402
from wsp.stats import paired, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("items")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    from wsp.gen import generate  # noqa: PLC0415

    items = [json.loads(line) for line in open(a.items)]
    prof = profiles.get(C.conf(a.case)["profile"])
    out = C.run_path(a.case, a.model, "ifguard")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        generate(C.model_path(a.case, a.model), items, prof, n=a.n, out=out)
    rows = read_jsonl(out)
    res = {"summary": summarize(rows), "paired": None}
    base = C.run_path(a.case, C.BASE, "ifguard")
    if a.model != C.BASE and base.exists():
        res["paired"] = paired(read_jsonl(base), rows)
    write_json(C.cdir(a.case) / "results" / f"{a.model}__ifguard.json", res)
    print(json.dumps(res["paired"] or res["summary"], indent=1))


if __name__ == "__main__":
    main()
