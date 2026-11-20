"""Exact-form grading of the model's own practice answers, and the reference-form repair of an answer whose value is
right but whose boxed form is not.

The toolkit grader (wsp.grade) accepts a math answer when its last \\boxed{} equals the reference after light
normalisation (wsp.grade._norm: outer $, commas, spaces, \\text{} wrapper, final period), or when math-verify finds the
two equivalent. An exact reader, like an answer key, only does the first. Practice references are written in one
house style per source (GSM8K: a bare integer; MATH: mostly a bare integer or a braced \\frac), so an answer such as
\\boxed{26.00}, \\boxed{60\\%}, \\boxed{\\$720}, \\boxed{8 \\text{ cm}}, \\boxed{x=6} or \\boxed{36^\\circ} for the
reference 26 / 60 / 720 / 8 / 6 / 36 has the right value in a form the reference does not use.

  python derive/exact.py summary RUN.jsonl [...]               exact vs lenient accuracy, form misses by kind, cuts
  python derive/exact.py paired BASE.jsonl EDIT.jsonl [...] [--out F.json]   paired gains, exact and lenient (SE by problem)
  python derive/exact.py pooled F1.json F2.json                 each edit's exact gain pooled over those paired files

RUN files are wsp.gen / wsp.case run outputs (rows with id, k, source, gold, text, correct, cut, think_closed).
Only math-task rows are re-graded; choice and instruction-following rows keep the toolkit's verdict (their graders do
not use math-verify).
"""
import argparse
import collections
import json
import math
import re

NON_MATH_SOURCES = {"mmlupro": "choice", "mmlupro_sug": "choice", "ifsynth": "ifeval", "tuluif": "ifeval"}


def _grade_fns():
    """The toolkit's own extraction and normalisation (imported lazily: the sandbox has the toolkit installed)."""
    from wsp.grade import _norm, answer_part, last_boxed  # noqa: PLC0415
    return _norm, answer_part, last_boxed


def task_of(row):
    src = (row.get("source") or "").split(":")[-1]
    return NON_MATH_SOURCES.get(src, "math")


def boxed_span(text):
    """(start, end) of the content of the last \\boxed{...} in text (balanced braces), or None. Same rule as
    wsp.grade.last_boxed."""
    i = text.rfind("\\boxed")
    while i >= 0:
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 0
            for k in range(j, len(text)):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    return j + 1, k
        i = text.rfind("\\boxed", 0, i)
    return None


PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
UNIT_TEXT = re.compile(r"\\(?:text|mbox|mathrm|textbf|mathbf)\s*\{[^{}]*\}")
DECORATIONS = (  # (kind, pattern): what a boxed answer carries beyond the bare number
    ("percent", re.compile(r"\\%|%|\\text\s*\{\s*percent\s*\}")),
    ("currency", re.compile(r"\\\$|\$|\\text\s*\{\s*dollars?\s*\}")),
    ("degree", re.compile(r"\^\s*\{?\s*\\circ\s*\}?|°|\\degree")),
    ("unit", UNIT_TEXT),
)


def strip_decorations(pred):
    """-> (core, kinds): the boxed content without units, %, currency, degree marks, a leading 'x =' and thousands
    separators; kinds lists what was removed."""
    s, kinds = pred.strip(), []
    m = re.match(r"^\s*[A-Za-z](?:_\{?\w+\}?)?\s*=\s*(.+)$", s)
    if m:
        s, kinds = m.group(1), kinds + ["var_eq"]
    for kind, rx in DECORATIONS:
        t = rx.sub("", s)
        if t != s:
            s, kinds = t, kinds + [kind]
    s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", "").replace("~", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s).strip().rstrip(".").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s, kinds = s.split(".")[0], kinds + ["trailing_zeros"]
    return s, kinds


def form_miss(pred, gold):
    """Kinds of decoration when pred is gold's plain number written in another form, else None."""
    gold = str(gold).strip()
    if pred is None or not PLAIN_NUMBER.match(gold.replace(",", "")):
        return None
    core, kinds = strip_decorations(pred)
    try:
        same = abs(float(core) - float(gold.replace(",", ""))) < 1e-9
    except ValueError:
        return None
    if same and not kinds and core != gold.replace(",", ""):
        kinds = ["number_form"]  # e.g. 12.50 for 12.5
    return kinds if same and kinds else None


def exact_correct(row):
    """Exact-form verdict of one generated answer (math rows re-graded; other tasks keep the toolkit verdict)."""
    if task_of(row) != "math":
        return bool(row["correct"])
    if not row.get("think_closed", True):
        return False
    _norm, answer_part, last_boxed = _grade_fns()
    pred = last_boxed(answer_part(row["text"]))
    return pred is not None and _norm(pred) == _norm(str(row["gold"]))


def classify(row):
    """exact | form (right value, decorated plain number) | frac_style | equivalent (any other box the toolkit's
    symbolic fallback accepts: right value, written unlike the reference) | wrong | no_answer | cut"""
    if row.get("cut"):
        return "cut"
    if exact_correct(row):
        return "exact"
    if task_of(row) != "math":
        return "wrong"
    _norm, answer_part, last_boxed = _grade_fns()
    pred = last_boxed(answer_part(row["text"]))
    if pred is None:
        return "no_answer"
    if form_miss(pred, row["gold"]):
        return "form"
    if _norm(re.sub(r"\\[dt]frac", r"\\frac", pred)) == _norm(str(row["gold"])):
        return "frac_style"
    if row.get("correct") and row.get("think_closed", True):
        return "equivalent"
    return "wrong"


def repair(row, broad=False):
    """The answer rewritten so that its last box holds the reference, when the original box holds the same value in
    another form (decorations, or \\dfrac / \\tfrac for a \\frac reference; with broad=True also any other box that
    the toolkit's symbolic fallback accepts as the reference's value). Only the box content changes; the thinking
    and the rest of the answer stay the model's own. -> (repaired text, char offset of the box content, kinds) or
    None."""
    if task_of(row) != "math" or row.get("cut") or not row.get("think_closed", True):
        return None
    _norm, answer_part, last_boxed = _grade_fns()
    text = row["text"]
    cut = text.rfind("</think>")
    head, ans = (text[:cut], text[cut:]) if cut >= 0 else ("", text)
    span = boxed_span(ans)
    if span is None:
        return None
    pred, gold = ans[span[0]:span[1]].strip(), str(row["gold"]).strip()
    if _norm(pred) == _norm(gold):
        return None
    kinds = form_miss(pred, gold)
    if not kinds:
        if _norm(re.sub(r"\\[dt]frac", r"\\frac", pred)) == _norm(gold):
            kinds = ["frac_style"]
        elif broad and row.get("correct"):
            kinds = ["equivalent"]
        else:
            return None
    new = head + ans[:span[0]] + gold + ans[span[1]:]
    return new, len(head) + span[0], kinds


# ------------------------------------------------------------------------------------------------ statistics
def _read(path):
    import gzip  # noqa: PLC0415
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(rows):
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r.get("source")].append(r)
    out = {}
    for src, rs in sorted(by_src.items(), key=lambda kv: str(kv[0])):
        kinds = collections.Counter(classify(r) for r in rs)
        miss = collections.Counter(k for r in rs for k in ((repair(r) or (None, None, []))[2]))
        n = len(rs)
        out[src] = {"n": n, "lenient": round(100 * sum(bool(r["correct"]) for r in rs) / n, 2),
                    "exact": round(100 * kinds["exact"] / n, 2), "cut": round(100 * kinds["cut"] / n, 2),
                    "form_miss": round(100 * (kinds["form"] + kinds["frac_style"]) / n, 2),
                    "equivalent": round(100 * kinds["equivalent"] / n, 2),
                    "kinds": dict(kinds), "form_kinds": dict(miss),
                    "mean_tokens": round(sum(r.get("n_tokens", 0) for r in rs) / n)}
    return out


def paired(base, edit, exact=True):
    """Gain of edit over base on shared (id, k): mean over problems of the per-problem mean difference, SE clustered
    by problem; overall and by source."""
    f = exact_correct if exact else (lambda r: bool(r["correct"]))
    b = {(r["id"], r["k"]): r for r in base}
    e = {(r["id"], r["k"]): r for r in edit}
    keys = sorted(set(b) & set(e))

    def stat(ks):
        per = collections.defaultdict(list)
        for k in ks:
            per[k[0]].append(int(f(e[k])) - int(f(b[k])))
        pm = [sum(v) / len(v) for v in per.values()]
        if not pm:
            return None
        g = sum(pm) / len(pm)
        se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / len(pm))
        return {"problems": len(pm), "base": round(100 * sum(f(b[k]) for k in ks) / len(ks), 2),
                "edit": round(100 * sum(f(e[k]) for k in ks) / len(ks), 2), "gain": round(100 * g, 2),
                "se": round(100 * se, 2),
                "cut": [round(100 * sum(bool(b[k]["cut"]) for k in ks) / len(ks), 1),
                        round(100 * sum(bool(e[k]["cut"]) for k in ks) / len(ks), 1)]}
    by = collections.defaultdict(list)
    for k in keys:
        by[b[k].get("source")].append(k)
    return {"all": stat(keys), "by_source": {s: stat(ks) for s, ks in sorted(by.items(), key=lambda kv: str(kv[0]))}}


def pooled(files):
    """{edit: {"gain", "se", "problems", "sets"}} pooled over the paired files that contain the edit (matched by the
    edit name, the part of the run file name before '__')."""
    import os  # noqa: PLC0415
    acc = collections.defaultdict(list)
    for path in files:
        res = json.load(open(path))
        for run, v in res["edits"].items():
            s = v["exact"]["all"]
            if s:
                acc[os.path.basename(run).split("__")[0]].append((s["gain"], s["se"], s["problems"], path))
    out = {}
    for name, xs in acc.items():
        n = sum(x[2] for x in xs)
        out[name] = {"gain": round(sum(x[0] * x[2] for x in xs) / n, 2),
                     "se": round(math.sqrt(sum((x[1] * x[2]) ** 2 for x in xs)) / n, 2),
                     "problems": n, "sets": [x[3] for x in xs]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["gain"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("runs", nargs="+")
    p = sub.add_parser("paired")
    p.add_argument("base")
    p.add_argument("edits", nargs="+")
    p.add_argument("--out", help="also write the results as JSON to this file")
    q = sub.add_parser("pooled", help="problem-weighted exact gain of each edit over several paired files "
                                      "(e.g. res/dev128.json res/devB.json)")
    q.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "pooled":
        print(json.dumps(pooled(a.files), indent=1))
        return
    if a.cmd == "summary":
        for path in a.runs:
            print(path)
            print(json.dumps(summary(_read(path)), indent=1))
    else:
        base, res = _read(a.base), {}
        for path in a.edits:
            rows = _read(path)
            res[path] = {"exact": paired(base, rows, True), "lenient": paired(base, rows, False)}
            print(path)
            print(json.dumps(res[path], indent=1))
        if a.out:
            import os  # noqa: PLC0415
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump({"base": a.base, "edits": res}, f, indent=1)


if __name__ == "__main__":
    main()
