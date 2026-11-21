"""Step 3, Check B - would-close, teacher-forced (no sampling noise, minutes): feed the start's own dev runs through
the start and through a candidate edit (applied in memory) and find, per run, the first thinking position where
`</think>` becomes the most likely next token, and in the answer the positions where it is the most likely token.

  python derive/wouldclose.py <case> <edit> [--tags devA devB] [--shard 0/2]     (one GPU per shard)
  python derive/wouldclose.py <case> <edit> --report                               (after every shard: the verdict)
Run classes (from the start's own outcome): stuck = cut while thinking, no loop; loop = cut while thinking in a
verbatim loop; short right / long right = right within / beyond 8k tokens; other = wrong, or cut inside the answer.
A run "would close" when the candidate makes `</think>` the top token at a thinking position before the start itself
does (the scan starts at the last prompt position, which chooses the first thinking token; a close there means an
empty thinking and fails the check); the close is attributed to the dead-end unit with the larger push there (each
unit's push = its activation times its down column's effect on the `</think>` logit). The answer scan starts at the
`</think>` token itself. --report needs every shard of the split.
Two kinds of limit: harm (the edit closes where it must not, or too early: before the first thinking token, early
in runs that never finish, right runs, `</think>` on top inside answers) and reach (it closes too few of the runs its
units are built for). A limit about the runs of a unit the edit does not hold (stuck runs: closing, median first
close; looping runs: closing; the units' roles come from the build records of gates.py) is reported, not judged.
For looping runs it also prints how many thinking tokens after the loop's onset (dlib.loop_onset) the first close
comes, against how long the loop ran in the start's run.
-> $WSP_WORK/k4/<case>/wouldclose_<edit>[.<i>of<n>].json, checkB_<edit>.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

LIMITS = {   # breakage limits (a broken build fails them; a working one passes with room)
    "runs that would close before their first thinking token (%)": ("<=", 0.0),
    "stuck runs that would close (%)": (">=", 40.0),
    "median first close in stuck runs (thinking tokens)": (">=", 10000.0),
    "loop runs that would close (%)": (">=", 70.0),
    "short right runs that would close (%)": ("<=", 10.0),
    "long right runs that would close (%)": ("<=", 40.0),
    "answer positions of right runs with </think> on top (per 5k answer tokens)": ("<=", 1.0),
}
REACH = {"stuck runs that would close (%)": "stuck", "loop runs that would close (%)": "loop"}   # -> the unit's role
OF_ROLE = {**REACH, "median first close in stuck runs (thinking tokens)": "stuck"}           # limits about one role


def klass(r):
    o = D.outcome(r)
    if o in ("stuck", "loop"):
        return o
    if r["correct"]:
        return "short_right" if r["n_tokens"] <= 8000 else "long_right"
    return "other"


def onset_tokens(tf, item, r):
    """Thinking tokens before the start of the run's verbatim loop (dlib.loop_onset), or None."""
    lo = D.loop_onset(D.thinking(r))
    if not lo:
        return None
    enc = tf.encode(item, r["text"], 32600, 0, offsets=True)
    return next((j for j, (s, e) in enumerate(enc["offs"]) if s >= lo[0]), None)


def first_top(tf, z, end_id):
    """Index of the first row of z whose top token is </think> (None), and the boolean mask over all rows."""
    top = np.zeros(len(z), bool)
    for c, lg in tf.logits(z):
        top[c:c + len(lg)] = (lg.argmax(-1) == end_id).cpu().numpy()
    idx = np.flatnonzero(top)
    return (int(idx[0]) if len(idx) else None), top


def shard(a):
    from wsp.data import by_id  # noqa: PLC0415
    si, sn = map(int, a.shard.split("/"))
    spec, sdir = D.edit_spec(a.case, a.edit)
    mdir = D.start_model(a.case)
    info = D.model_info(mdir)
    E, gain = D.readout(mdir)
    tf = D.TF(mdir)
    e = E[tf.end_id] * gain
    units, eff = [], []
    for op in spec["ops"]:                      # the dead-end units: set_cols on the last layer's down_proj
        if op["op"] == "set_cols" and re.search(rf"layers\.{info['last']}\.mlp\.down_proj", op["tensor"]):
            col = np.asarray(op["values"], np.float32).reshape(-1)
            units.append(int(op["cols"][0]))
            eff.append(float(e @ col))
    names = D.unit_roles(a.case)
    roles = sorted({names.get(u, "?") for u in units})       # "?": a unit without a build record
    items = by_id()
    rows = [r for i, r in enumerate(D.rows_of(a.case, a.tags)) if r["id"] in items and i % sn == si]
    out = []

    def read(r, arm):
        enc = tf.encode(items[r["id"]], r["text"], 32600, a.max_answer)
        P, T, A = enc["P"], enc["T"], enc["A"]
        think = np.arange(P - 1, P + T - 1)              # index j: the close would follow j thinking tokens
        ans = np.arange(P + T, P + T + A - 1) if A > 1 else np.array([], int)
        pos = np.concatenate([think, ans])
        if not len(pos):
            return None
        caps = tf.run(enc["ids"], pos, logits=True, units=units if arm == "edit" and units else None)
        f_think, _ = first_top(tf, caps["z"][: len(think)], tf.end_id)
        _, top_ans = first_top(tf, caps["z"][len(think):], tf.end_id) if len(ans) else (None, np.zeros(0, bool))
        res = {f"{arm}_first": f_think, f"{arm}_answer_tops": int(top_ans.sum()),
               "think_tokens": int(T), "answer_tokens": int(len(ans))}
        if arm == "edit" and units and f_think is not None:
            push = caps["act"][f_think].cpu().numpy() * np.array(eff) / float(caps["rms"][f_think])
            j = int(np.argmax(push))
            res["by"] = names.get(units[j], str(units[j])) if push[j] > 1.0 else "other"
            res["push_at_first"] = [round(float(x), 2) for x in push]
        return res

    for i, r in enumerate(rows):                         # the start as it is
        res = read(r, "start")
        if res:
            out.append({"id": r["id"], "k": r["k"], "tag": r["tag"], "source": D.src_of(r), "class": klass(r), **res})
            if out[-1]["class"] == "loop":
                out[-1]["loop_onset"] = onset_tokens(tf, items[r["id"]], r)
        if (i + 1) % 50 == 0:
            print(f"[wouldclose] start: {i + 1}/{len(rows)} runs", flush=True)
    keyed = {(o["tag"], o["id"], o["k"]): o for o in out}
    with tf.patched(spec["ops"], sdir):                   # the candidate, applied once in memory
        for i, r in enumerate(rows):
            o = keyed.get((r["tag"], r["id"], r["k"]))
            if o is not None:
                o.update(read(r, "edit"))
            if (i + 1) % 50 == 0:
                print(f"[wouldclose] edit: {i + 1}/{len(rows)} runs", flush=True)
    f = D.kdir(a.case) / (f"wouldclose_{a.edit}" + (f".{si}of{sn}" if sn > 1 else "") + ".json")
    D.save_json(f, {"edit": a.edit, "tags": a.tags, "units": units, "roles": roles, "runs": out})
    print(f"[wouldclose] {len(out)} runs -> {f}")


def report(a):
    pat = re.compile(rf"wouldclose_{re.escape(a.edit)}(\.(\d+)of(\d+))?\.json$")
    found = [(f, pat.search(f.name)) for f in D.kdir(a.case).glob(f"wouldclose_{a.edit}*.json")]
    found = [(f, m) for f, m in found if m]
    whole = [f for f, m in found if not m.group(1)]
    split = {int(m.group(3)) for f, m in found if m.group(1)}
    if whole:
        files = whole
    elif len(split) == 1:
        n = split.pop()
        have = {int(m.group(2)) for f, m in found if m.group(1)}
        if have != set(range(n)):
            raise SystemExit(f"would-close shards {sorted(set(range(n)) - have)} of {n} are missing: wait for them "
                             "(or retry them)")
        files = [f for f, m in found if m.group(1)]
    elif split:
        raise SystemExit(f"would-close shards of different splits {sorted(split)}: delete the stale ones")
    else:
        files = []
    docs = [json.loads(Path(f).read_text()) for f in files]
    runs = [r for d in docs for r in d["runs"]]
    if not runs:
        raise SystemExit("no would-close results: run the shards first")
    roles = set().union(*[d.get("roles", ["?"]) for d in docs])     # older shard files: roles unknown
    holds = {k: "?" in roles or k in roles for k in ("stuck", "loop")}  # unknown units: every limit is judged

    def closes(r):
        return r["edit_first"] is not None and (r["start_first"] is None or r["edit_first"] < r["start_first"])
    cls = {c: [r for r in runs if r["class"] == c] for c in ("stuck", "loop", "short_right", "long_right", "other")}
    pct = lambda rs: 100 * float(np.mean([closes(r) for r in rs])) if rs else float("nan")  # noqa: E731
    stuck_at = [r["edit_first"] for r in cls["stuck"] if closes(r)]
    right = cls["short_right"] + cls["long_right"]
    ans_tok = sum(r["answer_tokens"] for r in right)
    vals = {"runs that would close before their first thinking token (%)":
                100 * float(np.mean([closes(r) and r["edit_first"] == 0 for r in runs])),
            "stuck runs that would close (%)": pct(cls["stuck"]),
            "median first close in stuck runs (thinking tokens)":
                float(np.median(stuck_at)) if stuck_at else float("nan"),
            "loop runs that would close (%)": pct(cls["loop"]),
            "short right runs that would close (%)": pct(cls["short_right"]),
            "long right runs that would close (%)": pct(cls["long_right"]),
            "answer positions of right runs with </think> on top (per 5k answer tokens)":
                5000 * sum(max(r["edit_answer_tops"] - r["start_answer_tops"], 0) for r in right) / max(ans_tok, 1)}
    other_tops = sum(max(r["edit_answer_tops"] - r["start_answer_tops"], 0) for r in cls["other"])
    ok, lines, failed = True, [], {"harm": [], "reach": []}
    for k, (op, lim) in LIMITS.items():
        v, kind = vals[k], "reach" if k in REACH else "harm"
        few = {"stuck": len(cls["stuck"]) < 5, "median": len(stuck_at) < 3, "loop": len(cls["loop"]) < 5,
               "short": len(cls["short_right"]) < 5, "long": len(cls["long_right"]) < 5,
               "answer": not right}.get(k.split()[0], False)
        if few:
            lines.append(f"  n/j   [{kind}] {k}: {v:.2f} (limit {op} {lim}; not judged: too few runs of this class)")
            continue
        if k in OF_ROLE and not holds[OF_ROLE[k]]:
            lines.append(f"  n/j   [{kind}] {k}: {v:.2f} (limit {op} {lim}; not judged: the edit holds no "
                         f"{OF_ROLE[k]} unit)")
            continue
        good = bool(np.isfinite(v)) and ((v >= lim) if op == ">=" else (v <= lim))
        ok &= good
        if not good:
            failed[kind].append(k)
        lines.append(f"  {'pass' if good else 'FAIL'}  [{kind}] {k}: {v:.2f} (limit {op} {lim})")
    after = [(r["edit_first"] - r["loop_onset"], r["think_tokens"] - r["loop_onset"]) for r in cls["loop"]
             if closes(r) and r.get("loop_onset") is not None]
    onset = {}
    if after:
        d_, span = np.array([x for x, _ in after]), np.array([y for _, y in after])
        onset = {"runs": len(after), "median_tokens_after_onset": float(np.median(d_)),
                 "p10": float(np.percentile(d_, 10)), "p90": float(np.percentile(d_, 90)),
                 "closed_before_onset": int((d_ < 0).sum()), "median_loop_tokens_in_start_run": float(np.median(span))}
    status = "PASS" if ok else ("FAIL (a harm limit)" if failed["harm"] else "reach not met (no harm limit failed)")
    by = {}
    for r in runs:
        if closes(r):
            by[r.get("by", "?")] = by.get(r.get("by", "?"), 0) + 1
    srcs = sorted({r["source"] for r in runs})
    per_src = "; ".join(f"{s}: " + ", ".join(f"{c} {sum(closes(r) for r in cls[c] if r['source'] == s)}/"
                                            f"{sum(r['source'] == s for r in cls[c])}" for c in cls) for s in srcs)
    text = (f"Check B: {status} ({len(runs)} runs; units of the edit: {', '.join(sorted(roles)) or 'none'}; classes " +
            ", ".join(f"{c} {len(v)}" for c, v in cls.items()) + ")\n" + "\n".join(lines) +
            f"\n  would close, by source (class closes/runs): {per_src}\n  closes attributed to: {by}"
            f"\n  runs that end wrong or are cut inside their answer: {pct(cls['other']):.1f}% would close earlier; "
            f"their answers have </think> on top at {other_tops} positions (reported, not a limit)" +
            (f"\n  looping runs that would close ({onset['runs']}): first close a median of "
             f"{onset['median_tokens_after_onset']:.0f} thinking tokens after the loop's onset (10-90% "
             f"{onset['p10']:.0f}..{onset['p90']:.0f}; before the onset in {onset['closed_before_onset']}), where the "
             f"start's loop ran a median of {onset['median_loop_tokens_in_start_run']:.0f} tokens to the cap"
             if onset else ""))
    D.log(a.case, f"would-close {a.edit}", text)
    D.save_json(D.kdir(a.case) / f"checkB_{a.edit}.json",
                {"pass": bool(ok), "harm_pass": not failed["harm"], "failed": failed, "roles": sorted(roles),
                 "values": vals, "by": by, "loop_onset_to_first_close": onset})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--tags", nargs="+", default=["devA", "devB"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-answer", type=int, default=2000)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else shard(a)


if __name__ == "__main__":
    main()
