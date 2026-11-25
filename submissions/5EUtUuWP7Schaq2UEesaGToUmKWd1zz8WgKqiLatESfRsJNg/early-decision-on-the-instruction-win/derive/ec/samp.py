"""Sample a case's model on a split with sampling overrides (temperature test), written as run <name>__<set>.

  python derive/samp.py <case> <model> <set> --name t045 --temperature 0.45 --n 4 --salt c

Full eval token cap (the profile's max_tokens) unless --max-tokens is given. The toolkit's own generate() does
seeds (stable per id, k, salt), sharding over the visible GPUs and grading; only the sampling dict is overridden.
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/root/sn99/wsp"))
from wsp import case as C, data, gen, profiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("set")
    ap.add_argument("--name", required=True, help="run name (written as <name>__<set>.jsonl)")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--presence-penalty", type=float)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--salt", required=True)
    a = ap.parse_args()
    samp = {k: v for k, v in (("temperature", a.temperature), ("presence_penalty", a.presence_penalty)) if v is not None}
    prof = profiles.get(C.conf(a.case)["profile"])
    idx = data.by_id()
    items = [idx[i] for i in C.set_ids(a.case, a.set)]
    out = C.RUNS / a.case / f"{a.name}__{a.set}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[samp] {a.model} {len(items)} problems x {a.n}, sampling {dict(prof['sampling'], **samp)} -> {out}", flush=True)
    gen.generate(C.model_path(a.case, a.model), items, prof, n=a.n, out=str(out), salt=a.salt, max_tokens=a.max_tokens,
                 sampling=samp or None)


if __name__ == "__main__":  # gen.generate spawns workers that re-import this module
    main()
