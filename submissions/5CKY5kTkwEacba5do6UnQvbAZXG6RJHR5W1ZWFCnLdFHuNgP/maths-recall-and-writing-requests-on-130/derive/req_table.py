#!/usr/bin/env python3
"""req_table.py START_GLOB NAME=EDIT_GLOB [NAME=EDIT_GLOB ...] [--items REQ.json] [--by kind|source|family]
Practice rows of wsp.gen (id, k, source, correct, cut, n_tokens), START against each edit, paired by problem:
share of answers that are right and not cut; the difference and its standard error over problems."""
import collections, glob, json, math, sys

args, opt, i = [], {}, 1
while i < len(sys.argv):
    if sys.argv[i].startswith("--"):            # an option and its value
        opt[sys.argv[i][2:]] = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        i += 2
    else:
        args.append(sys.argv[i]); i += 1
kind_of = {}
if "items" in opt:
    for it in json.load(open(opt["items"])):
        kind_of[it["id"]] = it.get("kind") or it.get("source")


def load(pat):
    by = collections.defaultdict(list); meta = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                by[r["id"]].append((bool(r.get("correct") in (True, "True")) and not r.get("cut"), r.get("n_tokens", 0), bool(r.get("cut"))))
                meta[r["id"]] = r.get("source") or r["id"].split("_")[0]
    return by, meta


def group(i, meta):
    if opt.get("by") == "family" and i.startswith("bbhsynth_"):
        return i.rsplit("_", 1)[0].replace("bbhsynth_", "")
    return kind_of.get(i) or meta.get(i, "?")


S, meta = load(args[0])
for spec in args[1:]:
    name, pat = spec.split("=", 1)
    Ed, m2 = load(pat)
    ids = sorted(set(S) & set(Ed))
    if not ids:
        print(f"{name}: no common problems ({pat})"); continue
    acc = lambda D, i: sum(x[0] for x in D[i]) / len(D[i])

    def line(label, ii):
        if not ii:
            return
        d = [acc(Ed, i) - acc(S, i) for i in ii]; m = sum(d) / len(d)
        se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
        tok = lambda D: sum(x[1] for i in ii for x in D[i]) / max(1, sum(len(D[i]) for i in ii))
        cut = lambda D: 100 * sum(x[2] for i in ii for x in D[i]) / max(1, sum(len(D[i]) for i in ii))
        print(f"  {label:34s} {len(ii):5d} | start {100 * sum(acc(S, i) for i in ii) / len(ii):5.1f} ({tok(S):6.0f} tok, cut {cut(S):4.1f}%) | "
              f"{name} {100 * sum(acc(Ed, i) for i in ii) / len(ii):5.1f} ({tok(Ed):6.0f} tok, cut {cut(Ed):4.1f}%) | {100 * m:+6.2f} +- {100 * se:4.2f}")
    print(f"{name} against the start ({len(ids)} problems):")
    line("ALL", ids)
    gs = collections.defaultdict(list)
    for i in ids:
        gs[group(i, meta)].append(i)
    if len(gs) > 1:
        for g in sorted(gs):
            line(g, gs[g])
