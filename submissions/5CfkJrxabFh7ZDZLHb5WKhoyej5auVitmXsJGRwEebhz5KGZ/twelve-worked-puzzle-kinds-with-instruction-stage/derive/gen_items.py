"""Answers to items given in a file (made requests, in the toolkit's item format), with the profile's sampler.
  python gen_items.py MODEL_DIR ITEMS.jsonl OUT.jsonl [--n 2] [--salt mk] [--max-tokens N] [--ids IDS.json]"""
import os
import json
import sys

if __name__ == "__main__":
    from wsp import profiles
    from wsp.gen import generate

    a = sys.argv[1:]
    opt = lambda name, d: a[a.index(name) + 1] if name in a else d  # noqa: E731
    items = [json.loads(l) for l in open(a[1]) if l.strip()]
    if "--ids" in a:
        keep = set(json.load(open(opt("--ids", ""))))
        items = [it for it in items if it["id"] in keep]
    rows = generate(a[0], items, profiles.get(os.environ.get("WSP_PROFILE", "qwen35-2b-think")), n=int(opt("--n", 2)), out=a[2], salt=opt("--salt", "mk"),
                    max_tokens=int(opt("--max-tokens", 0)) or None)
    T = lambda x: str(x) == "True" or x is True  # noqa: E731
    by = {}
    for r in rows:
        k = r["id"].split("_")[1] if r["id"].startswith("mk_") else "other"
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += T(r["correct"])
    print("[gen_items] right by kind:", {k: f"{100 * v[1] / v[0]:.1f}% of {v[0]}" for k, v in by.items()}, flush=True)
