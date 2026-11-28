"""Instruction requests, two models on the same requests: share of answers that follow the whole request, per kind
of instruction and over all, with the difference and its standard error over the requests.

  python derive/cmp_kind.py A.jsonl B.jsonl [--items ITEMS.jsonl] [--min 10] [--echo]

--echo adds three lines about what the layer must not do: the share of answers to requests that do not ask for a
repeat that begin with the request itself (its first 40 characters, spacing and case aside); the share of answers to
requests that do not ask for capitals or lowercase that carry the plain sentence of derive/mend_rows.py; and the share
of answers to requests that do ask for capitals or lowercase that are that sentence and nothing else.

A request's kind is the instruction it carries (requests with several: each of its instructions counts it once).
A request's value is the mean over its answers (a cut answer and one whose thinking did not close are wrong)."""
import ast
import collections
import json
import math
import re
import sys

from wsp import data

a = sys.argv[1:]
opt = lambda name, d: a[a.index(name) + 1] if name in a else d  # noqa: E731
by = dict(data.by_id("practice:all"))
if "--items" in a:
    for l in open(opt("--items", "")):
        if l.strip():
            it = json.loads(l)
            by[it["id"]] = it
T = lambda x: x is True or str(x) == "True"  # noqa: E731


ECHO = {}
PLAIN = re.compile(r"(this|that) is the answer to the request and all of it is in (capital|lowercase) letters\.?")
PLAIN_START = re.compile(r"(this|that) is the answer to the request")       # the sentence, or a variant of it


def squash(x):
    return " ".join(x.lower().split())


def per_request(path):
    out = collections.defaultdict(list)
    echo = [0, 0, 0, 0, 0, 0]
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            out[r["id"]].append(1.0 if T(r.get("correct")) and not T(r.get("cut")) else 0.0)
            it = by.get(r["id"])
            if it is not None and "</think>" in r.get("text", ""):
                ans, req, kinds = squash(r["text"].split("</think>")[-1]), squash(it["problem"]), str(it.get("instruction_id_list"))
                if "repeat_prompt" not in kinds:
                    echo[0] += 1
                    echo[1] += len(req) >= 40 and ans[:40] == req[:40]
                if "english_capital" not in kinds and "english_lowercase" not in kinds:
                    echo[2] += 1
                    echo[3] += bool(PLAIN_START.search(ans))
                else:
                    echo[4] += 1
                    echo[5] += bool(PLAIN.search(ans)) and not any(c.isalnum() for c in PLAIN.sub("", ans))
    ECHO[path] = echo
    return {i: sum(v) / len(v) for i, v in out.items()}


def kinds_of(i):
    x = by[i].get("instruction_id_list") or []
    return ast.literal_eval(x) if isinstance(x, str) else x


A, B = per_request(a[0]), per_request(a[1])
ids = sorted(i for i in set(A) & set(B) if i in by)
groups = collections.defaultdict(list)
for i in ids:
    groups["ALL REQUESTS"].append(i)
    for k in kinds_of(i):
        groups[k].append(i)
rows = []
for k, g in groups.items():
    if len(g) < int(opt("--min", 10)):
        continue
    d = [B[i] - A[i] for i in g]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    rows.append((k, len(g), 100 * sum(A[i] for i in g) / len(g), 100 * sum(B[i] for i in g) / len(g), 100 * m, 100 * se))
for k, n, pa, pb, m, se in sorted(rows, key=lambda r: (r[0] != "ALL REQUESTS", -abs(r[4]))):
    print(f"{k:48s} {n:4d} requests  {pa:5.1f}% -> {pb:5.1f}%  {m:+6.2f} +- {se:.2f}")
if "--echo" in a:
    ea, eb = ECHO[a[0]], ECHO[a[1]]
    pc = lambda x, n: f"{100 * x / max(1, n):.1f}% of {n}"  # noqa: E731
    print(f"answers that begin with the request itself, requests that do not ask for it: {pc(ea[1], ea[0])} -> {pc(eb[1], eb[0])}")
    print(f"answers that carry the plain sentence, requests without a case instruction: {pc(ea[3], ea[2])} -> {pc(eb[3], eb[2])}")
    print(f"answers that are the plain sentence and nothing else, requests with a case instruction: {pc(ea[5], ea[4])} -> {pc(eb[5], eb[4])}")
