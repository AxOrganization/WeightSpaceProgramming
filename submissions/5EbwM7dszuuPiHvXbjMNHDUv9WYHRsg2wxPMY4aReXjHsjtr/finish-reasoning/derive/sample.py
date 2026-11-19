"""Generate and grade a model's answers with the toolkit's generation (wsp.gen.generate: the profile's sampling,
per-problem seeds), on practice problems and on the tasks of make_ifwrite.py, loading the practice sets once.

usage:
  python sample.py --case C --edit base|EDIT [--set dev] [--ids IDS.json] [--items TASKS.jsonl ...]
                   --out OUT.jsonl [--n 2] [--max-tokens 8192] [--salt tr]
--edit   "base" (the case's parent model) or an edit of the case; its model is built on first use
         (--model DIR instead: a model directory)
--set    a problem set of the case (dev, ho0, ...);  --ids: a json list of practice ids;  --items: task rows
--max-tokens 0  means the profile's own token limit (full-length answers, for evaluation runs)
The rows (id, k, source, n_tokens, finish, correct, cut, think_closed, kind, text) are written when all answers
are done, so the output file exists only when the job is complete.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit")
    ap.add_argument("--model")
    ap.add_argument("--set")
    ap.add_argument("--ids")
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--salt", default="tr")
    a = ap.parse_args()

    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import CASES  # noqa: PLC0415
    from wsp.gen import generate  # noqa: PLC0415
    prof = profiles.get(json.load(open(CASES / a.case / "case.json"))["profile"])
    by = data.by_id()
    ids = []
    if a.set:
        ids += json.load(open(CASES / a.case / "sets" / f"{a.set}.json"))
    if a.ids:
        ids += json.load(open(a.ids))
    items = [by[i] for i in ids]
    for f in a.items:
        items += [json.loads(line) for line in open(f)]
    if not items:
        raise SystemExit("no problems: give --set, --ids or --items")
    model = a.model
    if not model:
        from wsp.case import model_path  # noqa: PLC0415
        model = model_path(a.case, a.edit or "base")
    generate(str(model), items, prof, n=a.n, out=a.out, salt=a.salt, max_tokens=a.max_tokens or None)


if __name__ == "__main__":
    main()
