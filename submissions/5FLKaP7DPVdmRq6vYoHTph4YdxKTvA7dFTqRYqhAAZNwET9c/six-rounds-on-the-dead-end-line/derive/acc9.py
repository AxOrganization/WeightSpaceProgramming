"""Practice readings of a model's own sampled answers (pc_sample.py / genreq.py files): per source, answers, share
right, share finished (not cut at the token limit), mean tokens. The same practice prompts sampled again on the next
stage's model give the breakage guard of Section 5.

    python derive/acc9.py FILES...            # one table over all files
"""
import collections
import json
import sys


def table(files):
    by = collections.defaultdict(list)
    for f in files:
        try:
            for line in open(f):
                r = json.loads(line)
                by[r.get("source", "?")].append(r)
        except OSError:
            continue
    out = []
    for src, rs in sorted(by.items()):
        n = len(rs)
        right = sum(bool(r.get("correct")) for r in rs) / n
        fin = sum(not r.get("cut") for r in rs) / n
        tok = sum(r.get("n_tokens", 0) for r in rs) / n
        out.append(f"  {src:14s} {n:6d} answers  right {100 * right:5.1f}%  finished {100 * fin:5.1f}%  tokens {tok:6.0f}")
    return out


if __name__ == "__main__":
    print("\n".join(table(sys.argv[1:])) or "  (no answers)")
