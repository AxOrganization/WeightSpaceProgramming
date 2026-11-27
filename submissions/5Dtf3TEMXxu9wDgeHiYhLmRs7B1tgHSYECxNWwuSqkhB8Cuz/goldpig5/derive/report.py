"""report.py OUT --stage A|C     the run's REPORT.md from what the steps measured (their records in $WSP_WORK/steps
                                and logs); the toolkit's uploader appends the case's result tables.
   report.py - --mixed           the new answers after each finished thinking (step c_pool): how many starts have
                                answers that keep the request and answers that break it, per source -> steps/mixed.json"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
WORK = os.environ.get("WSP_WORK", "")


def state(name):
    p = os.path.join(WORK, "steps", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def logtail(name, n=40):
    p = os.path.join(WORK, "logs", f"{name}.log")
    if not os.path.exists(p):
        return "(no record)"
    return "\n".join(open(p).read().strip().splitlines()[-n:])


def mixed():
    import own
    own.register()
    from wsp import case as C
    from wsp.common import read_jsonl, write_json
    out = {}
    for tag in ("xfit", "tfit"):
        p = C.run_path("ic", "base", f"{tag}.branches")
        if not p.exists():
            continue
        first = read_jsonl(C.run_path("ic", "base", tag))
        g = collections.defaultdict(list)
        for r in read_jsonl(p):
            g[(r["id"], r["from_k"])].append(bool(r["correct"]) and not r["cut"])
        by = collections.defaultdict(lambda: [0, 0, 0, 0])
        src = {r["id"]: str(r.get("source")) for r in first}
        for (i, _), v in g.items():
            b = by[src.get(i, "?")]
            b[0] += 1
            b[1] += 0 < sum(v) < len(v)
            b[2] += sum(v) == len(v)
            b[3] += sum(v)
        fin = sum(1 for r in first if "</think>" in r["text"] and not r["cut"])
        ok = sum(bool(r["correct"]) and not r["cut"] for r in first)
        out[tag] = {"requests": len(first), "followed_first": ok / max(1, len(first)), "finished": fin,
                    "by_source": {s: {"starts": b[0], "mixed": b[1], "all_right": b[2],
                                      "answers_right": b[3] / max(1, sum(len(v) for (i, _), v in g.items() if src.get(i, "?") == s))}
                                  for s, b in by.items()}}
        for s, b in sorted(by.items()):
            print(f"{tag} {s}: {b[0]} starts, mixed {b[1]} ({100 * b[1] / max(1, b[0]):.1f} %), every new answer right {b[2]}")
    write_json(os.path.join(WORK, "steps", "mixed.json"), out)
    tot = sum(v["mixed"] for t in out.values() for v in t["by_source"].values())
    print(f"MIXED starts in all: {tot} (each gives up to 2 pairs of a kept and a broken answer)")


def main():
    if "--mixed" in sys.argv:
        return mixed()
    out, stage = sys.argv[1], sys.argv[sys.argv.index("--stage") + 1]
    L, E, loop, own_v, mix = state("layers"), state("edits"), state("loop"), state("own"), state("mixed")
    t = ["# Report", "",
         "Two stages, both derived in this run from the pinned base, the toolkit's practice sets, requests written by",
         "derive/reqs.py from practice material, and the model's own generations. No evaluation item, answer or result",
         "was read. Numbers below are this run's own; the tables at the end are the toolkit's records of every run.", "",
         "## Stage A: the reasoning loop", "",
         "Base on the practice problems of the screen (accuracy, share cut at the token limit, accuracy when finished):", "",
         "```", logtail("a_base", 3), "```", "",
         f"Layers chosen from the direction table (largest MLP writers of the loop direction, last layer left alone): "
         f"{' '.join(L['layers']) if L else '?'}", "", "```", logtail("a_layers", 12), "```", "",
         "Selectivity on held-out pairs (log-probability per token of the normal and of the looping continuation):", "",
         "```"] + [f"{m}: " + logtail(f"a_sel_{m}", 1) for m in (["base"] + (E["cands"] + [E["control"]] if E else []))] + ["```", "",
         "Candidates and the random control against the base, paired (same problems and seeds):", "",
         "```", logtail("a_choose", 60), "```", "",
         f"Chosen: {loop['choice'] if loop else '?'} (bar {'passed' if loop and loop.get('pass') else 'NOT passed'})."]
    if stage == "C":
        t += ["", "## Stage C: instruction requests", "",
              "Requests written and what the start does with them (fit pool, token cap of the pool only):", "",
              "```", logtail("c_init", 4), logtail("c_pool", 3), "```", "",
              "New answers after each finished thinking (starts with answers that keep and answers that break the request):", "",
              "```", logtail("c_mixed", 12), "```", "", "Fits (data counts and the end of each log):", "", "```"]
        for i in range(1, 9):
            p = os.path.join(WORK, "cases", "ic", "fits", f"F{i}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                t.append(f"F{i}: " + json.dumps(d.get("data", {}))[:600])
                lg = d.get("log") or []
                t.append(f"F{i} log end: " + json.dumps(lg[-2:])[:400])
        t += ["```", "", "```", logtail("c_spec", 6), "```", "",
              "The stage against the start, paired: never-fitted instruction types (ifxH), a fresh practice split (all sources):", "",
              "```", logtail("c_verdict", 30), "```", "",
              ("Delivered: stage A + stage C." if own_v and own_v.get("keep") else
               "The instruction stage was HELD BACK by its harm gate in this run: the delivered edit is stage A alone, "
               "and the numbers above say why.")]
    t += ["", "## What did not work / limits", "",
          "See the tables: every candidate, the control and every source are listed, losses included. The loop stage's",
          "choice was made on the practice problems it is reported on (a selection among few candidates with a large",
          "effect); the instruction stage is reported on requests and problems no choice has seen.", ""]
    open(out, "w").write("\n".join(t))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
