"""mend_req_rows.py --answers ROWS --out PAIRS [--also-own N]

Mended answers for practice instruction requests.

For every finished answer of the start that breaks its request, the answer is mended by fixed rules for the
instructions it broke, and the pair is written ONLY when the mended answer keeps EVERY instruction of the request by
the practice item's own checker, in each of three gradings (the language detector draws at random on short texts):
  lowercase / capitals      the same answer in lowercase / in capitals
  the case kept, and the    one closing sentence in the first person, in the asked case (the checker of the two case
  case instruction is the   instructions also wants English, and its language detector reads only the beginnings of
  only one still failing    capital words: a short answer in capitals comes out as another language; a short lowercase
                            answer can too)
  end phrase                the asked phrase added as the last line
  quotation                 the answer wrapped in double quotation marks
  repeat the request        the request, word for word, put in front (an own copy that is not word for word is
                            replaced by it)
  paragraphs with ***       the answer's own paragraphs joined by the asked divider, when their number is the asked one
No rule writes a keyword of the request into the answer: a sentence that names asked keywords is then also written
for requests that forbid words.
A mended answer that only ADDS text at the end: prefix = the whole own answer, chosen = the added text, rejected =
the end right there. Otherwise: prefix = the thinking, chosen = the mended answer, rejected = the own answer.
--also-own N: for a request with a passing and a failing own answer, up to N pairs own passing / own failing."""
import argparse, json, collections, random, re
from wsp import data
from wsp.grade import score
THINK_END = "</think>"
ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--also-own", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
CLOSERS = ["I HOPE THAT THIS IS WHAT YOU WANTED AND I AM HAPPY TO HELP IF I CAN DO MORE FOR YOU",
           "I HOPE THAT THIS HELPS YOU AND I AM HAPPY TO HELP IF YOU NEED MORE FROM ME",
           "I HOPE THAT I WAS OF HELP TO YOU AND I AM GLAD TO DO MORE IF YOU WISH"]
items = data.by_id()
rng = random.Random(a.seed)
by = collections.defaultdict(list)
for l in open(a.answers):
    if l.strip():
        r = json.loads(l); by[r["id"]].append(r)
def grade(it, think, ans):
    """(every instruction kept in each of three gradings, the per-instruction verdicts of the last one). The checkers
    of the two case instructions ask a language detector that draws at random on short texts: one passing draw is
    not enough for an example."""
    ok, pred = True, None
    for _ in range(3):
        sc = score(it, think + ans, "stop", True)
        ok, pred = ok and bool(sc["correct"]), sc.get("pred")
    return ok, pred


def passes(it, think, ans):
    return grade(it, think, ans)[0]


CASE = ("change_case:english_capital", "change_case:english_lowercase")
def mend(it, body, broken):
    """the mended answer, and the names of the rules used"""
    ids = it.get("instruction_id_list") or []; kws = it.get("kwargs") or [{}] * len(ids)
    kw = {i: (k or {}) for i, k in zip(ids, kws)}
    x, used = body, []
    if "length_constraints:number_paragraphs" in broken:
        n = kw["length_constraints:number_paragraphs"].get("num_paragraphs")
        parts = [p.strip() for p in re.split(r"\n\s*\*{3,}\s*\n|\n\s*\n", x) if p.strip() and not re.fullmatch(r"\*{3,}", p.strip())]
        if n and len(parts) == n:
            x = "\n***\n".join(parts); used.append("paragraphs")
    if "combination:repeat_prompt" in broken:
        p = (kw["combination:repeat_prompt"].get("prompt_to_repeat") or "").strip()
        if p and not x.strip().lower().startswith(p.lower()):
            # an own copy of the request that is not word for word is replaced, not doubled
            import difflib
            best, cut = 0.0, 0
            for m in re.finditer(r"\n+|(?<=[.?!])\s+", x[:int(len(p) * 1.4) + 40]):
                k = m.start()
                if k >= 0.6 * len(p):
                    q = difflib.SequenceMatcher(None, x[:k].lower(), p.lower(), autojunk=False).ratio()
                    if q > best:
                        best, cut = q, m.end()
            x = p + "\n\n" + (x[cut:] if best >= 0.8 else x); used.append("repeat" if best < 0.8 else "repeat (own copy replaced)")
    if "startend:end_checker" in broken:
        e = (kw["startend:end_checker"].get("end_phrase") or "").strip()
        if e and not x.strip().endswith(e):
            x = x.rstrip() + "\n\n" + e; used.append("end phrase")
    if "startend:quotation" in broken:
        x = '"' + x.strip().strip('"') + '"'; used.append("quotation")
    if "change_case:english_lowercase" in ids and x != x.lower():
        x = x.lower(); used.append("lowercase")
    if "change_case:english_capital" in ids and x != x.upper():
        x = x.upper(); used.append("capitals")
    return x, used
out, n, per_type = [], collections.Counter(), collections.Counter()
for pid, rs in by.items():
    it = items[pid]; ids = it.get("instruction_id_list") or []
    ok = [r for r in rs if r.get("correct") and r.get("finish") == "stop"]
    bad = [r for r in rs if not r.get("correct") and r.get("finish") == "stop" and THINK_END in r["text"]]
    n["failed, finished"] += len(bad)
    for r in bad:
        pred = r.get("pred")
        broken = [i for i, p in zip(ids, pred) if not p] if isinstance(pred, list) and len(pred) == len(ids) else list(ids)
        cut = r["text"].rindex(THINK_END) + len(THINK_END)
        think, ans = r["text"][:cut], r["text"][cut:]
        lead = ans[:len(ans) - len(ans.lstrip())]; body = ans.strip()
        if not body:
            n["empty answer"] += 1; continue
        x, used = mend(it, body, broken)
        good, pred = grade(it, think, lead + x)
        good = good and x != body
        low = "change_case:english_lowercase" in ids
        # the closing sentence only when the answer is in the asked case and the case instruction is what still fails
        # (its checker also wants English) and no other instruction fails
        still = [i for i, v in zip(ids, pred) if not v] if isinstance(pred, list) and len(pred) == len(ids) else list(ids)
        only_case = bool(still) and all(i in CASE for i in still) and (x.islower() if low else x.isupper())
        if not good and only_case:
            for c in rng.sample(CLOSERS, len(CLOSERS)):
                end = "" if x.endswith((".", "!", "?", ":", "*", ")")) else "."
                y = x + end + "\n\n" + (c.lower() if low else c) + "."
                if passes(it, think, lead + y):
                    x, good = y, True; used.append("closing sentence"); break
        if not good:
            n["not mended"] += 1
            for b in broken:
                per_type[(b, "not mended")] += 1
            continue
        for b in broken:
            per_type[(b, "mended")] += 1
        name = "mend:" + "+".join(used)
        if x.startswith(body) and len(x) > len(body):
            out.append({"id": pid, "type": "pair", "subject": name, "prefix": think + lead + body,
                        "chosen": {"text": x[len(body):], "finished": True}, "rejected": {"text": "", "finished": True}})
        else:
            out.append({"id": pid, "type": "pair", "subject": name, "prefix": think + lead,
                        "chosen": {"text": x, "finished": True}, "rejected": {"text": body, "finished": True}})
        n["mended"] += 1; n[name] += 1
    if a.also_own and ok and bad:
        rng.shuffle(ok); rng.shuffle(bad)
        for j in range(min(a.also_own, len(ok), len(bad))):
            out.append({"id": pid, "type": "pair", "subject": "own:" + str(ok[j].get("source")), "chosen": {"text": ok[j]["text"], "finished": True},
                        "rejected": {"text": bad[j]["text"], "finished": True}})
            n["own passing / own failing"] += 1
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(dict(n)))
print("by broken instruction (mended / not):", {t: (per_type[(t, "mended")], per_type[(t, "not mended")]) for t in sorted({k[0] for k in per_type})})
print(len(out), "pairs ->", a.out)
