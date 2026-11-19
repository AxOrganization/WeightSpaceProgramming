"""Per-source sets of a case, so that each task type runs on its own GPU and is judged on its own.

  python derive/sets.py <case>                    writes sets/<split>_<source>.json for split in dev, holdout
  python derive/sets.py <case> --extra ifsynth    also sets/extra_ifsynth.json: every practice item of that source in
                                                  neither split (a third, larger set where the splits are small)
"""
import argparse

from wsp import case as C
from wsp import data
from wsp.common import write_json


def sources(case):
    return [p.split("_")[0] for p in C.conf(case).get("practice", data.PRACTICE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--extra", nargs="*", default=[], help="sources to make an extra_<source> set of")
    a = ap.parse_args()
    out, used = {}, set()
    for split in ("dev", "holdout"):
        ids = C.set_ids(a.case, split)
        used |= set(ids)
        for src in sources(a.case):
            name = f"{split}_{src}"
            write_json(C.cdir(a.case) / "sets" / f"{name}.json", [i for i in ids if i.startswith(src)])
            out[name] = len(C.set_ids(a.case, name))
    for src in a.extra:
        rest = sorted(i for i, item in data.by_id().items() if item.get("source") == src and i not in used)
        write_json(C.cdir(a.case) / "sets" / f"extra_{src}.json", rest)
        out[f"extra_{src}"] = len(C.set_ids(a.case, f"extra_{src}"))
    print(out)


if __name__ == "__main__":
    main()
