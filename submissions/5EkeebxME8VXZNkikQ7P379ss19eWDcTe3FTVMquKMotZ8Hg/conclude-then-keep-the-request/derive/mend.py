"""Mended answers (the idea of the finished entry 0009565562-0012, our own code): a sampled instruction answer
that failed its checker is mended where the failed instruction has a mechanical remedy that writes nothing the request
does not itself give, graded again, and kept if it now follows every instruction of its request. Each kept mend and
the answer it came from form one pair for derive/pref_gains.py (same thinking, prefix up to </think>).

  python derive/mend.py --rows A.jsonl [B.jsonl ...] --out MENDS.jsonl [--items REQ.json ...]

Remedies: repeat the request (the request itself where the answer begins: in place of a copy that differs only in
whitespace or case, or in front of an answer that starts elsewhere); all capitals / all lowercase (the answer in upper /
lower case); quotation marks (the answer wrapped in double quotes); end phrase (whatever follows the phrase cut).
"""
import argparse
import ast
import collections
import json
import re

from wsp import data
from wsp.grade import score

REP, CAP, LOW = "combination:repeat_prompt", "change_case:english_capital", "change_case:english_lowercase"
QUOTE, END = "startend:quotation", "startend:end_checker"


def as_list(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def put_request(req, ans):
    """The answer with the exact request at its start; None when it copies part of the request and then leaves it."""
    req = req.strip()
    body = ans.lstrip()
    r = re.sub(r"\s+", "", req).lower()
    i, j, last = 0, 0, None
    while i < len(body) and j < len(r):
        c = body[i]
        if c.isspace():
            i += 1
            continue
        if c.lower() != r[j]:
            break
        j += 1
        last = i
        i += 1
    if j == len(r):
        return req + body[last + 1:]
    if j < min(20, len(r) // 4):
        return req + "\n\n" + body
    return None


def mend_one(item, answer, failed):
    kws = as_list(item["kwargs"])
    ids = as_list(item["instruction_id_list"])
    kw = {k: (w or {}) for k, w in zip(ids, kws)}
    a = answer
    for inst in failed:
        if inst == REP:
            a = put_request(kw[REP].get("prompt_to_repeat", ""), a)
            if a is None:
                return None
        elif inst == QUOTE:
            a = '"' + a.strip().strip('"') + '"'
        elif inst == END:
            p = kw[END].get("end_phrase", "")
            k = a.rfind(p) if p else -1
            if k < 0:
                return None
            a = a[: k + len(p)]
        elif inst not in (CAP, LOW):
            return None
    if CAP in failed:
        a = a.upper()
    if LOW in failed:
        a = a.lower()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", nargs="*", help="files of further items (requests made in the run)")
    ap.add_argument("--skip-kinds", nargs="*", default=["repc", "repm"], help="kinds of run-made requests whose answers are "
                    "not mended: a long copied multiple-choice or math prompt followed by its solution (the finished entry "
                    "0009565562-0012 found that such mends cost math)")
    a = ap.parse_args()
    by = data.by_id()
    for p in a.items or []:
        by.update({it["id"]: it for it in json.load(open(p))})
    n = collections.Counter()
    types = collections.Counter()
    with open(a.out, "w") as f:
        for p in a.rows:
            for line in open(p):
                x = json.loads(line)
                it = by.get(x["id"])
                if (it is None or it.get("task") != "ifeval" or x.get("correct") or x.get("finish") != "stop"
                        or not x.get("think_closed") or "</think>" not in x["text"] or it.get("kind") in a.skip_kinds):
                    continue
                ids = as_list(it["instruction_id_list"])
                ok = x.get("pred")
                if not isinstance(ok, list) or len(ok) != len(ids):
                    continue
                failed = [i for i, good in zip(ids, ok) if not good]
                n["failed answers"] += 1
                e = x["text"].find("</think>") + len("</think>")
                prefix, answer = x["text"][:e], x["text"][e:]
                lead = answer[: len(answer) - len(answer.lstrip())]
                m = mend_one(it, answer.strip(), failed)
                if m is None:
                    continue
                n["mended"] += 1
                g = score(it, prefix + lead + m, "stop", True)
                if not g["correct"]:
                    continue
                n["kept"] += 1
                for t in failed:
                    types[t] += 1
                base = {"id": x["id"], "k": x["k"], "point": "mend", "source": x.get("source"), "prefix": prefix,
                        "finish": "stop", "branch": "mend"}
                f.write(json.dumps(dict(base, text=prefix + lead + m, correct=True)) + "\n")
                f.write(json.dumps(dict(base, text=x["text"], correct=False)) + "\n")
    print(dict(n))
    print("kept mends by instruction:", dict(types.most_common()))


if __name__ == "__main__":
    main()
