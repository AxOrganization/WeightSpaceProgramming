"""Paired reading of two runs on the same items (wsp.gen rows): per item the share right of its answers, the mean
difference B - A and its standard error over items.   python paired.py A.jsonl B.jsonl"""
import json, math, sys, collections
def load(p):
    d = collections.defaultdict(list)
    for l in open(p):
        if l.strip():
            r = json.loads(l); d[r["id"]].append(float(bool(r.get("correct")) and not r.get("cut")))
    return {k: sum(v) / len(v) for k, v in d.items()}
a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) & set(b))
diff = [b[i] - a[i] for i in ids]
m = sum(diff) / len(diff); sd = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1))
print(f"{100 * sum(a[i] for i in ids) / len(ids):.1f}% -> {100 * sum(b[i] for i in ids) / len(ids):.1f}%, "
      f"gain {100 * m:+.1f} ± {100 * sd / math.sqrt(len(ids)):.1f} ({len(ids)} items; B better on {sum(x > 0 for x in diff)}, A on {sum(x < 0 for x in diff)})")
