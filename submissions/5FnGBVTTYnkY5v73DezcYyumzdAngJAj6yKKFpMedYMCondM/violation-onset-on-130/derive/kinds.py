"""Where the answers of each run of a case set go: accuracy, cut share (exact loop / other), finished wrong,
accuracy when finished, mean length, accuracy per practice source.

    python kinds.py CASE SET          (e.g. kinds.py concl dev128: the base run first, then every edit run)
"""
import collections
import json
import sys

from wsp import case as C


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    case, tag = sys.argv[1], sys.argv[2]
    runs = sorted((C.RUNS / case).glob(f"*__{tag}.jsonl"), key=lambda p: (not p.name.startswith(C.BASE + "__"), p.name))
    for p in runs:
        rows = [json.loads(l) for l in open(p)]
        n, k = len(rows), collections.Counter(r["kind"] for r in rows)
        fin = [r for r in rows if not r["cut"]]
        src = collections.defaultdict(list)
        for r in rows:
            src[r.get("source", r["id"].split("_")[0])].append(r["correct"])
        print(f"{p.name.split('__')[0]:20s} n={n} acc={100 * sum(r['correct'] for r in rows) / n:5.1f} "
              f"cut={100 * sum(r['cut'] for r in rows) / n:5.1f} (loop {k['cut_loop']}, other {k['cut_other']}) "
              f"wrong={k['finished_wrong']} acc|finished={100 * sum(r['correct'] for r in fin) / max(1, len(fin)):5.1f} "
              f"tokens={sum(r['n_tokens'] for r in rows) / n / 1000:4.1f}k  "
              + " ".join(f"{s}={100 * sum(v) / len(v):4.1f}" for s, v in sorted(src.items())))


if __name__ == "__main__":
    main()
