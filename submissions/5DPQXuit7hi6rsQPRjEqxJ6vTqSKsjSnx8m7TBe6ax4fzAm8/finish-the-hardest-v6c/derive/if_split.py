"""compare.py's paired comparison (pooled reference draws, per-item d, SE clustered by item), split by the
answer-format rules an instruction prompt carries: one group per --types rule (the prompts carrying it; a prompt counts
in every listed rule it carries), `other` (the prompts carrying none of them) and `all`, pooled over --sets. Rule F
(step 10) reads it: each instruction-answer unit stays only if its own rule's group gains, and `other` stays in its
noise band.

  python if_split.py --case C --edit XF --sets auxtest --items data/aux_if.jsonl \
      --types punctuation:no_comma detectable_format:number_highlighted_sections --json OUT

Grades: `acc` (the row's `correct`: the toolkit's prompt-level strict instruction check) and `uncl` (the thinking never
closed). Item rules come from the practice data and the self-written prompt files (--items). Practice runs only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as CMP  # noqa: E402
from wsp import data  # noqa: E402
from wsp.common import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit", nargs="+", required=True)
    ap.add_argument("--ref", default="base")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--runs-dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    rules = {i: (r.get("instruction_id_list") or []) for i, r in data.by_id().items()}
    for p in a.items:
        for r in read_jsonl(p):
            rules[r["id"]] = r.get("instruction_id_list") or []
    if a.runs_dir:
        runs = Path(a.runs_dir)
    else:
        from wsp.common import RUNS  # noqa: PLC0415
        runs = RUNS / a.case
    records = []
    for edit in a.edit:
        es, bs = [], []
        for s in a.sets:
            ed, bd = CMP.find_draws(runs, edit, s), CMP.find_draws(runs, a.ref, s)
            if not ed or not bd:
                print(f"{edit} / {s}: {'no edit run' if not ed else 'no reference run'} in {runs}: skipped")
                continue
            print(f"{edit} / {s}: edit draws {[d for d, _ in ed]}, {a.ref} draws {[d for d, _ in bd]}")
            for dst, files in ((es, ed), (bs, bd)):
                for d, p in files:
                    dst.extend(dict(x, id=f"{s}/{x['id']}", rid=x["id"]) for x in CMP.load_rows(p, d))
        groups = {"all": lambda t: True, "other": lambda t: not any(r in t for r in a.types)}
        for r in a.types:
            groups[r] = lambda t, r=r: r in t
        for gname, want in groups.items():
            ge = [x for x in es if want(rules.get(x["rid"], []))]
            gb = [x for x in bs if want(rules.get(x["rid"], []))]
            if not ge or not gb:
                continue
            for g in ("exact", CMP.UNCLOSED):
                st = CMP.stats(ge, gb, g)
                records.append({"edit": edit, "group": gname, "grade": "acc" if g == "exact" else g, "sets": a.sets,
                                **st})
                if st["d"] is not None:
                    se = st["se"] if st["se"] is not None else float("nan")
                    print(f"{edit:>14} {gname:>46} {records[-1]['grade']:>5}: items {st['items']:4d}  "
                          f"{100 * st['base_acc']:6.2f} -> {100 * st['edit_acc']:6.2f}  d {100 * st['d']:+6.2f} +- "
                          f"{100 * se:5.2f}  (d - se {100 * (st['d'] - se):+6.2f})")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"ref": a.ref, "types": a.types, "records": records}, indent=1))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
