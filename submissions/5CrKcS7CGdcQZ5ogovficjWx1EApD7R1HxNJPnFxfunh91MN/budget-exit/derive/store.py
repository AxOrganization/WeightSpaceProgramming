"""Where the reads of a case are kept, and the answers they are taken from.

$WSP_WORK/cases/<case>/exit/
  acts_<model>.<tag>.<part>.npz   MLP inputs at sampled positions (scan.py collect). Tag "a": the starting model as it
                                  is (what the exit unit is fitted on). Tag "s": the starting model with the exit unit
                                  added and exits simulated where that unit would make them (what the box unit is
                                  fitted on: the states it will see).
  reads_<model>_<kind>.npz        the gate, constant and up directions of every layer (scan.py fit)
  slots_<model>.json              per layer, the hidden units that matter least
"""
import collections
import re

import numpy as np

from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl

import traces as T

ACTS_TAG = {"exit": "a", "box": "s", "brk": "s"}
FIELDS = ("x", "label", "split", "aid", "exit_at", "pos", "gen", "after", "source", "rms", "boxed", "a_pre")
EXIT_FILE = re.compile(r"\.(budget\d+(?:_[a-z0-9]+)?)\.jsonl$")
OPENED = {"_slash": T.BY_HAND + "\\", "_boxed": T.BY_HAND + "\\boxed{"}      # derive/ceiling.py --open, by file tag


def exit_dir(case):
    d = C.scratch(case) / "exit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def part_of(part, items):
    """I/N -> (every N-th item starting at I, a name for the part)."""
    i, n = map(int, part.split("/"))
    return [x for k, x in enumerate(items) if k % n == i], f"{i}of{n}"


def load_traces(case, model, sets, tok, exits=(), base=True, deep=2.0):
    """The answers of the runs as labelled traces (base=True), and one more trace for every exit made by hand on
    them (derive/ceiling.py) whose file tag is in `exits`, e.g. "budget24000_auto"."""
    prof = profiles.get(C.conf(case)["profile"])
    items, out = data.by_id(), []
    for s in sets:
        p = C.run_path(case, model, s)
        made = collections.defaultdict(list)
        for q in sorted(p.parent.glob(p.name + ".*.jsonl")):
            m = EXIT_FILE.search(q.name)
            if m and m.group(1) in exits:
                close = next((c for t, c in OPENED.items() if m.group(1).endswith(t)), T.BY_HAND)
                for x in read_jsonl(q):
                    made[(x["id"], x["k"])].append({**x, "tag": m.group(1), "close": x.get("close") or close})
        for r in read_jsonl(p):
            if r["id"] not in items:
                continue
            pr = T.prompt_of(tok, items[r["id"]], prof["thinking"])
            if base:
                out.append(T.trace(r, pr, deep))
            out += [T.trace(r, pr, deep, exit_row=x) for x in made[(r["id"], r["k"])]]
    return out


def load_acts(case, model, kind, fields=FIELDS):
    """The collected positions a kind of unit is fitted on -> {field: array}."""
    files = sorted(exit_dir(case).glob(f"acts_{model}.{ACTS_TAG[kind]}.*.npz"))
    if not files:
        raise SystemExit(f"nothing collected for a {kind} unit yet (scan.py collect" + (" --unit ..." if kind == "box" else "") + ")")
    zs = [np.load(f, allow_pickle=False) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in fields}


def sets_exit(z, a):
    """Exit unit. act: thinking after `late` generated tokens; silent: thinking before `early` and the prompt;
    up_on: thinking; up_off: where a second close must never come, each kind weighing the same: the closing token,
    the first positions of answers, the bodies of answers (of an answer after an exit: its first 300 tokens).
    exit_when "boxed": act only in tasks whose prompt asks for a boxed result, and the gate read is mixed with a read
    of that (task_on: thinking in such tasks, task_off: thinking in the others; scan.fit_layer), so that the unit
    leaves the other tasks to close their thinking by themselves."""
    lab, gen = z["label"], z["gen"]
    think = np.isin(lab, T.THINKING)
    body = (lab == T.ANSWER) | ((lab == T.AFTER_EXIT) & (z["after"] <= 300))
    off = [lab == T.CLOSE_POS, np.isin(lab, T.STARTS), body]
    act = think & (gen >= a.late)
    boxed = z["boxed"].astype(bool)
    if getattr(a, "exit_when", "any") == "boxed":
        act = act & boxed
    return {"act": act, "silent": (think & (gen < a.early)) | (lab == T.PROMPT), "think": think,
            "up_on": think, "up_off": off[0] | off[1] | off[2], "up_groups": [think, *off], "guard": off[0] | off[1],
            "task_on": think & boxed, "task_off": think & ~boxed, "body": body}


def sets_box(z, a):
    """Box unit, fitted on states that include the exit unit. An answer begins after a close at which the exit unit
    was pushing (a_pre > 0) or was silent (a_pre < -3).
    box_when "pushed": act at such a beginning after a push, in a task whose prompt asks for the result within
        \\boxed{}; silent, each kind weighing the same: beginnings after a close the exit unit had no part in,
        beginnings after a push in tasks that did not ask for a boxed result, and every other position.
    box_when "always": act at every beginning of an answer in a task that asks for a boxed result, whoever closed the
        thinking (to be used when the control of derive/ceiling.py --at-close shows that opening with the result
        costs nothing where the model had concluded by itself: then there is nothing to tell apart and nothing to
        miss); silent: beginnings in tasks that did not ask for a boxed result, and every other position."""
    lab, pre = z["label"], z["a_pre"]
    start = np.isin(lab, T.STARTS)
    other = ~start & (lab != T.SKIP)
    none = np.zeros(len(lab), bool)
    if getattr(a, "box_when", "pushed") == "always":
        plain = start & ~z["boxed"]
        return {"act": start & z["boxed"], "silent": plain | other, "groups": [plain, other], "quiet": none, "plain": plain,
                "other": other, "up_on": start, "up_off": other, "up_groups": [start, other], "guard": none}
    pushed, quiet = start & (pre > 0), start & (pre < -3)
    plain = pushed & ~z["boxed"]
    return {"act": pushed & z["boxed"], "silent": quiet | plain | other, "groups": [quiet, plain, other],
            "quiet": quiet, "plain": plain, "other": other, "up_on": start, "up_off": other, "up_groups": [start, other],
            "guard": none}


def sets_brk(z, a):
    """Break unit, fitted on states that include the exit unit: act at every closing token (the model's own, the exit
    unit's, a close at a random place): what follows it must be the paragraph break an answer starts with; silent:
    every other position."""
    lab = z["label"]
    close = lab == T.CLOSE_POS
    other = ~close & (lab != T.SKIP)
    none = np.zeros(len(lab), bool)
    return {"act": close, "silent": other, "groups": [other], "quiet": none, "plain": none, "other": other,
            "up_on": close, "up_off": other, "up_groups": [close, other], "guard": none}


SETS = {"exit": sets_exit, "box": sets_box, "brk": sets_brk}
