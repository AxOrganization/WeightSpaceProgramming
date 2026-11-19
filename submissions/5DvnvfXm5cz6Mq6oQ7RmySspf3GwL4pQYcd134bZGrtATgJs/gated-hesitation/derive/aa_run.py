"""A/A control: the case's starting model on a set again, with another sampling salt, paired with the base run.
The paired difference of identical models is pure sampling noise: it tells whether a small loss of an edit is real.

  python aa_run.py --case C --set dev_ifsynth --n 4 --salt s1
Writes runs/<case>/base@salt=<salt>__<set>.jsonl and results/base@salt=<salt>__<set>.json (paired with the base run).
"""
import argparse

from wsp import profiles
from wsp.case import _write_result, conf, model_path, run_path, set_items
from wsp.gen import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--salt", default="s1")
    a = ap.parse_args()
    prof = profiles.get(conf(a.case)["profile"])
    tag, items = set_items(a.case, a.set)
    name = f"base@salt={a.salt}"
    generate(model_path(a.case, "base"), items, prof, n=a.n, out=run_path(a.case, name, tag), salt=a.salt)
    print(_write_result(a.case, name, tag)["paired"])


if __name__ == "__main__":
    main()
