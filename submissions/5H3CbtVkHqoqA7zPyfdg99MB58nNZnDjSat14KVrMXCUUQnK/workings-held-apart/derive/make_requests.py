"""Requests for a second round of answer branching, recombined from the shipped practice requests.

Every task and every instruction sentence is cut out of the practice sets `ifsynth` and `tuluif` (and the word
problems of `gsm8k_train` that `ifsynth` is built on, used only to find where a request's instruction begins).
Nothing is written by hand and nothing comes from an instruction registry.

- wording bank: the instruction sentence of every one-instruction `ifsynth` request, the two sentences of every
  two-instruction one (split where the second type's sentence begins), the n-th-paragraph sentences and the fixed
  sentences of `tuluif`, each with its checker arguments. 22 of the 23 instruction types of the two sets;
- which types may share a request: the type pairs that the shipped two-instruction requests show. For the three
  types that only `tuluif` has, the short lists in ONLY3 below (types whose check cannot collide with theirs);
- `tulu`: a `tuluif` request (a natural task with one instruction) with one or two more sentences. The type of an
  added sentence is drawn first (every type the same chance, the n-th-paragraph type twice), then one of the
  type's distinct sentences;
- bare tasks: a `tuluif` request whose instruction is a fixed sentence at its start or end, with that sentence
  cut off. `rep1`: a bare task with the repeat sentence; `rep2`: the same with one more instruction; `case1`: a
  bare task with an all-capitals or an all-lowercase sentence; `rep3`: a `tuluif` request that carries a
  compatible instruction, with the repeat sentence behind it.

`--side train` (default) builds on the practice problems outside the case's dev and holdout splits; `--side dev`
or `--side holdout` builds held-out requests on the tasks of that split only.
"""
import argparse
import collections
import json
import os
import random
import re

from wsp import data

NTH = "length_constraints:nth_paragraph_first_word"
TWO = "combination:two_responses"
REP = "combination:repeat_prompt"
CAP = "change_case:english_capital"
LOW = "change_case:english_lowercase"
NTH_RE = re.compile(r"There should be \d+ paragraphs\. Paragraphs and only paragraphs are separated with each other by two "
                    r"line breaks\. The \d+-th paragraph must start with word \S+?\.")
TWO_TEXT = "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******."
REP_TEXT = ("First, repeat the request without change, then give your answer (do not say anything before repeating the "
            "request; the request you need to repeat does not include this sentence)")
# instruction sentences that the shipped tuluif requests carry word for word (no checker arguments)
CONST = {
    LOW: "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
    CAP: "Your entire response should be in English, capital letters only.",
    "detectable_format:title": "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
    "detectable_format:json_format": "Entire output should be wrapped in JSON format.",
    "punctuation:no_comma": "In your entire response, refrain from the use of any commas.",
    "startend:quotation": "Wrap your entire response with double quotation marks.",
    TWO: TWO_TEXT,
}
# Partners of the three types that no shipped two-instruction request shows: only types whose check cannot collide
# with theirs (a fixed paragraph layout, two separated responses, an answer that opens with the request).
WORDS = {"keywords:existence", "keywords:frequency", "keywords:forbidden_words", "keywords:letter_frequency"}
ONLY3 = {
    NTH: WORDS | {CAP, LOW, "change_case:capital_word_frequency", "punctuation:no_comma", "startend:end_checker",
                  "detectable_content:number_placeholders", "detectable_format:number_highlighted_sections",
                  "detectable_format:number_bullet_lists", "detectable_content:postscript"},
    TWO: WORDS | {CAP, LOW, "change_case:capital_word_frequency", "punctuation:no_comma",
                  "detectable_content:number_placeholders", "detectable_format:number_highlighted_sections"},
    REP: {"keywords:existence", "keywords:frequency", "detectable_content:number_placeholders",
          "detectable_format:number_highlighted_sections", "detectable_content:postscript", "startend:end_checker",
          "detectable_format:number_bullet_lists"},
}
WEIGHT = {NTH: 2.0}


def lcp(strs):
    s1, s2 = min(strs), max(strs)
    i = 0
    while i < len(s1) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def bare_task(it):
    """The task of a tuluif request whose instruction is a fixed sentence at its start or end, else None."""
    s = CONST.get(it["instruction_id_list"][0])
    if not s:
        return None
    p = it["problem"].strip()
    if p.endswith(s):
        b = p[:-len(s)].rstrip()
    elif p.startswith(s):
        b = p[len(s):].lstrip()
    else:
        return None
    return b if len(b) >= 20 else None


def build(seed=7, exclude_bases=(), tul_variants=(3, 3), prefix="rqt"):
    rng = random.Random(seed)
    excl = set(exclude_bases)
    by_prefix = collections.defaultdict(list)
    for r in data.load("practice:gsm8k_train"):
        by_prefix[r["problem"][:40]].append(r["problem"])
    ifs, tul = data.load("practice:ifsynth"), data.load("practice:tuluif")
    raw, pairs, doubles = collections.defaultdict(list), set(), []
    for it in ifs:
        q = max((p for p in by_prefix.get(it["problem"][:40], []) if it["problem"].startswith(p)), key=len, default=None)
        if q is None:
            continue
        ids, rest = it["instruction_id_list"], it["problem"][len(q):].strip()
        if len(ids) == 1:
            raw[ids[0]].append((rest, it["kwargs"][0]))
        else:
            pairs.add(frozenset(ids)); doubles.append((ids, rest, it["kwargs"]))
    first = {t: lcp([d for d, _ in v]) for t, v in raw.items()}
    split_ok = split_fail = 0
    for ids, rest, kw in doubles:
        a, b = first.get(ids[0], ""), first.get(ids[1], "")
        pos = [i for i in range(1, len(rest)) if rest[i - 1] == " " and rest.startswith(b, i)] if len(b) >= 12 else []
        if len(a) >= 12 and rest.startswith(a) and len(pos) == 1:
            raw[ids[0]].append((rest[:pos[0]].strip(), kw[0])); raw[ids[1]].append((rest[pos[0]:].strip(), kw[1])); split_ok += 1
        else:
            split_fail += 1
    # sentences of the tuluif-only types and tuluif's own wording of two fixed sentences, word for word
    for it in tul:
        if it["id"] in excl:
            continue
        t = it["instruction_id_list"][0]
        if t == NTH:
            m = NTH_RE.search(it["problem"])
            if m:
                raw[NTH].append((m.group(0), it["kwargs"][0]))
        elif t in (TWO, CAP, "detectable_format:json_format") and CONST[t] in it["problem"]:
            raw[t].append((CONST[t], it["kwargs"][0]))
    bank = {}
    for t, v in raw.items():   # distinct sentences only: a fixed sentence counts once
        seen, out = set(), []
        for d, k in v:
            key = (d, json.dumps(k, sort_keys=True))
            if key not in seen:
                seen.add(key); out.append((d, k))
        bank[t] = out
    types = sorted(bank)

    def compatible(t, others):
        if t in others:
            return False
        for o in others:
            if t in ONLY3 or o in ONLY3:
                a, b = (t, o) if t in ONLY3 else (o, t)
                if b in ONLY3 or b not in ONLY3[a]:
                    return False
            elif frozenset((t, o)) not in pairs:
                return False
        return True

    def pick(base_ids, n, allowed=None):
        out = []
        for _ in range(n):
            have = list(base_ids) + [x[0] for x in out]
            cand = [t for t in types if compatible(t, have) and (allowed is None or t in allowed)]
            if not cand:
                return None
            t = rng.choices(cand, weights=[WEIGHT.get(c, 1.0) for c in cand])[0]
            d, k = rng.choice(bank[t])
            out.append((t, d, k))
        return out

    items, skipped = [], collections.Counter()

    def put(kind, base_id, problem, ids, kwargs):
        items.append({"id": f"{prefix}_{len(items)}", "source": "rq", "task": "ifeval", "answer": None, "problem": problem,
                      "instruction_id_list": list(ids), "kwargs": list(kwargs), "base": base_id, "kind": kind})

    def add(base, n, kind):
        ex = pick(base["instruction_id_list"], n)
        if ex is None:
            skipped["no_compatible_sentence"] += 1
            return
        put(kind, base["id"], base["problem"].rstrip() + " " + " ".join(d for _, d, _ in ex),
            list(base["instruction_id_list"]) + [t for t, _, _ in ex], list(base["kwargs"]) + [k for _, _, k in ex])

    for base in tul:
        if base["id"] in excl:
            skipped["left_out"] += 1
            continue
        for n in [1] * tul_variants[0] + [2] * tul_variants[1]:
            add(base, n, "tulu")
        t0 = base["instruction_id_list"][0]
        b = bare_task(base)
        if b is not None:
            put("rep1", base["id"], b + " " + REP_TEXT, [REP], [{"prompt_to_repeat": b}])
            ex = pick([REP], 1)
            if ex:
                put("rep2", base["id"], b + " " + REP_TEXT + " " + ex[0][1], [REP, ex[0][0]], [{"prompt_to_repeat": b}, ex[0][2]])
            for t in (CAP, LOW):
                if t != t0:
                    d, k = rng.choice(bank[t])
                    put("case1", base["id"], b + " " + d, [t], [k])
        if t0 in ONLY3[REP]:
            p = base["problem"].rstrip()
            put("rep3", base["id"], p + " " + REP_TEXT, [t0, REP], list(base["kwargs"]) + [{"prompt_to_repeat": p}])
    tc = collections.Counter(t for i in items for t in i["instruction_id_list"])
    info = {"bank_types": len(types), "bank": sum(len(v) for v in bank.values()), "bank_nth": len(bank.get(NTH, [])),
            "split_ok": split_ok, "split_fail": split_fail, "type_pairs": len(pairs), "items": len(items),
            "by_kind": dict(collections.Counter(i["kind"] for i in items)),
            "by_count": dict(collections.Counter(len(i["instruction_id_list"]) for i in items)), "skipped": dict(skipped),
            "type_share_min_max": [round(min(tc.values()) / len(items), 3), round(max(tc.values()) / len(items), 3)],
            "nth": tc[NTH], "two": tc[TWO], "rep": tc[REP], "lower": tc[LOW], "capital": tc[CAP]}
    return items, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="the case whose dev and holdout problems are kept out")
    ap.add_argument("--out", required=True, help="jsonl of requests in the toolkit's item format")
    ap.add_argument("--side", choices=("train", "dev", "holdout"), default="train")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tul", default="2,2", help="requests with one more, with two more instructions per tuluif task")
    ap.add_argument("--max", type=int, default=0, help="keep at most this many requests (a small trial run)")
    a = ap.parse_args()
    from wsp import case as C
    held = {s: set(C.set_ids(a.case, s)) for s in ("dev", "holdout")}
    tul_ids = {it["id"] for it in data.load("practice:tuluif")}
    if a.side == "train":
        excl = held["dev"] | held["holdout"]
    else:
        excl = tul_ids - held[a.side]
    items, info = build(a.seed + {"train": 0, "dev": 1, "holdout": 2}[a.side], excl,
                        tuple(int(x) for x in a.tul.split(",")), prefix="rq" + a.side[0])
    if a.max and len(items) > a.max:
        items = random.Random(a.seed).sample(items, a.max)
        info["kept"] = len(items)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump([it["id"] for it in items], open(os.path.splitext(a.out)[0] + ".ids.json", "w"))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
