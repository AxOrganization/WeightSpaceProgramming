"""Requests that ask more at once, recombined from the practice sets' own material (the idea of the finished entry
0009559255-0006; this is our own code). No text is written: every request is cut out of practice items.

  python derive/mkreq.py --out REQ.json --side train|test --seed 1 [--exclude IDS.json ...] [--whole-tuluif]
                   [--n-rep 360] [--n-word 480] [--n-tulu 480]

Material
  instruction sentences   a one-instruction request of practice:ifsynth is a practice word problem followed by one
                          instruction sentence: what follows the word problem is the sentence, with its checker arguments
  the repeat sentence     what follows the request in practice:tuluif's own repeat requests
  which types may stand together   exactly the pairs of instruction types that practice:ifsynth's own two-instruction
                          requests contain
  bodies                  practice word problems, multiple-choice and math prompts (worded as the toolkit words them),
                          and practice:tuluif requests with the one instruction each already carries
Requests (items in the toolkit's own format, task and grader of practice:ifsynth, source "ifx")
  rep    a body + the repeat sentence (kinds repw / repc / repm / rept: word problem, multiple choice, math, tuluif)
  word   a word problem + 2 or 3 instruction sentences whose types may stand together
  tulu   a practice:tuluif request + 1 or 2 more sentences that may stand next to its own instruction
Sides: bodies and sentences are split once, by a fixed shuffle (bodies: the first 15% are the test side; sentences: every
fifth one-instruction request, by id). A test file shares neither bodies nor sentences with a train file.
"""
import argparse
import collections
import json
import random

from wsp import data

RID = "combination:repeat_prompt"
# A request that is repeated first can still keep these instructions: they ask for something inside the answer, not for
# a form of the whole response (a case, quotation marks, a wrapping, a count the repeated request would change).
WITH_REPEAT = {"keywords:existence", "detectable_format:title", "detectable_content:postscript",
               "detectable_content:number_placeholders", "detectable_format:number_highlighted_sections",
               "startend:end_checker", "detectable_format:multiple_sections"}


def material(exclude, whole_tuluif=False, ifsynth_cap=1024):
    words = {}
    for r in data.load("practice:gsm8k_train"):
        words.setdefault(r["problem"].strip()[:60], []).append(r)
    sent = {"train": collections.defaultdict(list), "test": collections.defaultdict(list)}
    allowed = set()
    for r in data.load("practice:ifsynth"):
        if int(r["id"].rsplit("_", 1)[1]) >= ifsynth_cap:
            continue
        ids = r["instruction_id_list"]
        if len(ids) == 2:
            allowed.add(frozenset(ids))
            continue
        cands = sorted(words.get(r["problem"].strip()[:60], []), key=lambda w: -len(w["problem"].strip()))
        body = next((w["problem"].strip() for w in cands if r["problem"].startswith(w["problem"].strip())), None)
        if body is None or not r["problem"][len(body):].strip():
            continue
        side = "test" if int(r["id"].rsplit("_", 1)[1]) % 5 == 0 else "train"
        sent[side][ids[0]].append((r["problem"][len(body):], r["kwargs"][0]))
    tails = collections.Counter()
    for r in data.load("practice:tuluif"):
        if r["instruction_id_list"] == [RID]:
            p = r["kwargs"][0].get("prompt_to_repeat") or ""
            if p and r["problem"].startswith(p):
                tails[r["problem"][len(p):]] += 1
    repeat = tails.most_common(1)[0][0]

    def pool(spec, ok=lambda r: True, whole=False):
        rows = sorted((r for r in data.load(spec) if r["id"] not in exclude and ok(r)), key=lambda r: r["id"])
        random.Random("bodies:" + spec).shuffle(rows)
        if whole:      # every row left after the exclusions belongs to the side that is being built
            return {"test": rows, "train": rows}
        k = max(1, len(rows) * 15 // 100)
        return {"test": rows[:k], "train": rows[k:]}
    bodies = {"w": pool("practice:gsm8k_train"), "c": pool("practice:mmlupro"), "m": pool("practice:math_train"),
              "t": pool("practice:tuluif", lambda r: RID not in r["instruction_id_list"] and len(r["instruction_id_list"]) == 1,
                        whole_tuluif)}
    return sent, allowed, repeat, bodies


def build(a):
    exclude = set()
    for p in a.exclude or []:
        exclude |= set(json.load(open(p)))
    sent, allowed, repeat, bodies = material(exclude, a.whole_tuluif, a.ifsynth_cap)
    pool = sent[a.side]
    types = sorted(pool)
    rng = random.Random(f"{a.side}:{a.seed}")
    B = {k: list(v[a.side]) for k, v in bodies.items()}
    for v in B.values():
        rng.shuffle(v)
    ok = lambda x, y: frozenset((x, y)) in allowed
    items = []

    def pick(have, k):
        got = []
        for t in rng.sample(types, len(types)):
            if len(got) == k:
                break
            if all(ok(t, h) for h in [*have, *got]):
                got.append(t)
        return got

    def add(kind, text, ids, kws, body_id):
        items.append({"id": f"ifx_{a.side}{a.seed}_{kind}_{len(items):04d}", "source": "ifx", "task": "ifeval", "problem": text,
                      "instruction_id_list": ids, "kwargs": kws, "answer": None, "kind": kind, "body": body_id})

    near = [r for r in B["t"] if r["instruction_id_list"][0] in WITH_REPEAT]
    n_t = min(len(near), a.n_rep * 3 // 10)
    n_w = (a.n_rep - n_t) * 4 // 9
    n_c = min(len(B["c"]), (a.n_rep - n_t) * 3 // 9)
    n_m = a.n_rep - n_t - n_w - n_c
    plan = [("w", r) for r in B["w"][:n_w]] + [("c", r) for r in B["c"][:n_c]] + [("m", r) for r in B["m"][:n_m]] + \
           [("t", r) for r in near[:n_t]]
    taken = {r["id"] for r in near[:n_t]}
    B["w"], B["t"] = B["w"][n_w:], [r for r in B["t"] if r["id"] not in taken]
    for sub, r in plan:
        body = r["problem"].strip() if sub in "wt" else data.user_message(r)
        ids, kws = (list(r["instruction_id_list"]), list(r["kwargs"])) if sub == "t" else ([], [])
        add("rep" + sub, body + repeat, [*ids, RID], [*kws, {"prompt_to_repeat": body}], r["id"])
    for r in B["w"][: a.n_word]:
        ts = pick([], rng.choice([2, 3, 3]))
        if len(ts) < 2:
            continue
        ss = [rng.choice(pool[t]) for t in ts]
        add("word", r["problem"].strip() + "".join(s for s, _ in ss), ts, [kw for _, kw in ss], r["id"])
    for r in B["t"][: a.n_tulu]:
        ts = pick(r["instruction_id_list"], rng.choice([1, 1, 2]))
        if not ts:
            continue
        ss = [rng.choice(pool[t]) for t in ts]
        add("tulu", r["problem"].rstrip() + "".join(s for s, _ in ss), [*r["instruction_id_list"], *ts],
            [*r["kwargs"], *(kw for _, kw in ss)], r["id"])
    return items, repeat, len(allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["train", "test"], default="train")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", nargs="*")
    ap.add_argument("--whole-tuluif", action="store_true", help="the tuluif requests left after --exclude all belong to this "
                    "side (the caller keeps the two sides' tuluif halves apart through --exclude)")
    ap.add_argument("--ifsynth-cap", type=int, default=1024, help="only practice:ifsynth requests numbered below this are "
                    "read for sentences and for the pairs of types that may stand together")
    ap.add_argument("--n-rep", type=int, default=360)
    ap.add_argument("--n-word", type=int, default=480)
    ap.add_argument("--n-tulu", type=int, default=480)
    a = ap.parse_args()
    items, repeat, n_allowed = build(a)
    json.dump(items, open(a.out, "w"))
    kinds = collections.Counter(it["kind"] for it in items)
    ni = collections.Counter(len(it["instruction_id_list"]) for it in items)
    print(f"{len(items)} requests -> {a.out}: {dict(kinds)}; instructions per request {dict(sorted(ni.items()))}; "
          f"{n_allowed} type pairs may stand together; repeat sentence {repeat[:60]!r}...")


if __name__ == "__main__":
    main()
