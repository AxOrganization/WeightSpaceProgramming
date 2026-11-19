"""The reads of an exit unit and of a box unit: what one MLP unit can know about where the model is in its answer.

A gated MLP unit has two linear reads of its input x: the gate (through SiLU: the unit is off while the read is
below a threshold) and the "up" read (it multiplies the output). We use them for two different questions and the unit
acts only when both say yes.

  python derive/scan.py collect <case> --sets dev_gsm8k ... [--exits budget24000_auto ...] [--part I/N]
      the starting model as it is: one pass over the answers (and over answers written after exits made by hand,
      derive/ceiling.py): the MLP input of every layer at sampled positions, with each position's label (traces.py),
      its index among the generated tokens and its task type.                       (--part I/N: one process per GPU)
  python derive/scan.py collect <case> --sets ... --unit exit:LAYER:GAIN:FIRE[:MARGIN] [--part I/N]
      the same model with that exit unit added by hook: sampled positions again, and for every answer the first two
      positions of an answer begun right after a close - the model's own close, the close the unit would make (a
      draw from its chance to close, build.sample_exit) and a close at a random place - with the unit's
      pre-activation at that close. No generation: the continuations are teacher-forced.
  python derive/scan.py fit <case> --kind exit|box|brk [--early E --late L] [--tag TAG]              (no GPU)
      per layer, from the fit half of the problems, the reads of that kind of unit; what they do on the sampled
      positions of the other half is printed and written to results/reads_<model>_<kind>.json

Kinds of unit
  exit  "the thinking has used up its budget": closes the thinking block.
        gate  how long the thinking has been going on. The direction is the mean, over answers that think for more
              than `late` generated tokens, of (their state after `late`) - (their state before `early`): the same
              answer on both sides, so what the problem is about cancels and what changes with the length stays
              (a plain contrast of late against early positions learns which problems run long, not the length).
              It is whitened by the covariance of the early positions of all answers, as LDA.
              (--exit-when boxed: plus a read of "the task asks for a boxed result" - LDA of thinking positions of
              such tasks against those of other tasks - at --task-mix times the spread of the length read: in other
              tasks the gate then stays below its threshold however long they think.)
        up    "still thinking": a ridge regression onto 1 for thinking positions and 0 for the closing token, the
              first positions of answers and answer bodies (each kind weighing the same). An answer written after a
              late exit is as late as the thinking before it; this read is what keeps the unit quiet there.
  box   "an answer begins after the exit unit pushed the thinking closed, and the task asked for a boxed result":
        makes that answer open with its boxed result. Fitted on states that include the exit unit.
        gate  LDA of such beginnings against three kinds of silent position that weigh the same: beginnings after a
              close the exit unit had no part in, beginnings after a push in tasks whose prompt did not ask for a
              boxed result, and every other position.  (--box-when always: every beginning of an answer in a task
              that asks for a boxed result, against beginnings in other tasks and every other position.)
        up    "an answer begins here": 1 for the first positions of answers, 0 everywhere else.
  brk   "a closing token was just written": makes the paragraph break an answer begins with follow every close.
        Fitted on the same states as the box unit. gate and up: the closing token (the model's own, the exit unit's,
        a close at a random place) against every other position.
"""
import argparse
import collections
import json

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json

import build
import store
import traces as T

BANDS = [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 20000), (20000, 24000), (24000, 28000),
         (28000, 10**6)]


class Rows:
    """Sampled positions of one pass, appended trace by trace."""

    def __init__(self):
        self.cols = collections.defaultdict(list)

    def add(self, tr, out, layers, pos, labels, n_prompt, close, a_pre, exit_at):
        pos = np.asarray(pos)
        after = pos - (close if close is not None else 10**9)
        c = self.cols
        c["x"].append(np.stack([out["x"][li] for li in layers], 1).astype(np.float16))            # [n, L, d]
        c["label"].append(np.asarray(labels, np.int8)); c["pos"].append(pos.astype(np.int32))
        c["split"].append(np.full(len(pos), T.split_of(tr["id"]), np.int8))
        c["aid"].append(np.full(len(pos), T.answer_key(tr["id"], tr["k"]), np.int64))
        c["exit_at"].append(np.full(len(pos), exit_at, np.int32))
        c["gen"].append((pos - n_prompt).astype(np.int32)); c["after"].append(np.clip(after, -1, None).astype(np.int32))
        c["source"].append(np.full(len(pos), tr["source"])); c["rms"].append(out["rms"][pos].astype(np.float32))
        c["boxed"].append(np.full(len(pos), tr["asks_box"])); c["a_pre"].append(np.asarray(a_pre, np.float32))

    def save(self, path):
        z = {k: np.concatenate(v) for k, v in self.cols.items()}
        np.savez(path, **z)
        return {T.NAMES[k]: v for k, v in sorted(collections.Counter(z["label"].tolist()).items())}


def sample_positions(rng, labels, n_prompt, per, thinking=True):
    """Positions read in one answer: the closing token and answer starts always; a few of the prompt, `per` of the
    thinking (evenly), and of the answer body half from its first 300 tokens.
    -> (positions, position of the closing token or None)"""
    closed = np.flatnonzero(labels == T.CLOSE_POS)
    after = np.arange(len(labels)) - (closed[0] if len(closed) else 10**9)
    pos = set(np.flatnonzero(np.isin(labels, [T.CLOSE_POS, *T.STARTS])).tolist())

    def take(idx, k):
        pos.update(rng.choice(idx, min(k, len(idx)), replace=False).tolist())

    if thinking:
        take(np.arange(n_prompt), 8)
        take(np.flatnonzero(np.isin(labels, T.THINKING)), per)
    body = np.flatnonzero(np.isin(labels, T.ANSWERING))
    take(body[after[body] <= 300], per // 2)
    take(body[after[body] > 300], per // 2)
    return np.asarray(sorted(pos), dtype=np.int64), (int(closed[0]) if len(closed) else None)


def cmd_collect(a):
    from reader import Reader  # noqa: PLC0415
    rd = Reader(C.model_path(a.case, a.model))
    layers = list(range(rd.n_layers))
    rng = np.random.default_rng(0)
    rows = Rows()
    if not a.unit:      # the starting model as it is
        trs, tag = store.part_of(a.part, store.load_traces(a.case, a.model, a.sets, rd.tok, a.exits))
        for n, tr in enumerate(trs):
            ids, off = rd.encode(tr["prompt"] + tr["text"])
            ids, off = ids[:a.max_tokens], off[:a.max_tokens]
            labels = T.token_labels(tr, off)
            n_prompt = int(np.searchsorted(np.asarray(off), len(tr["prompt"])))
            # an exit made by hand repeats the thinking of its answer: that is read once only
            pos, close = sample_positions(rng, labels, n_prompt, a.per, thinking=not tr["exit"])
            if len(pos):
                out = rd.run(ids, take={li: pos for li in layers}, want_rms=True)
                rows.add(tr, out, layers, pos, labels[pos], n_prompt, close, np.full(len(pos), np.nan), tr["exit_at"])
            if n % 50 == 0:
                print(f"[collect {tag}] {n}/{len(trs)} answers", flush=True)
        name = f"acts_{a.model}.{store.ACTS_TAG['exit']}.{tag}.npz"
    else:               # with the exit unit added: what the box unit will see
        sampling = profiles.get(C.conf(a.case)["profile"])["sampling"]
        A = build.build_unit(a.case, a.model, a.unit, rd, build.slots(a.case, a.model, rd, a.sets))
        print(build.describe(A), flush=True)
        trs, tag = store.part_of(a.part, store.load_traces(a.case, a.model, a.sets, rd.tok))
        made = collections.Counter()
        for n, tr in enumerate(trs):
            ids, off = rd.encode(tr["prompt"] + tr["text"])
            ids, off = ids[:a.max_tokens], off[:a.max_tokens]
            labels = T.token_labels(tr, off)
            n_prompt = int(np.searchsorted(np.asarray(off), len(tr["prompt"])))
            pos, close = sample_positions(rng, labels, n_prompt, a.per // 2)
            pre, out = build.profile(rd, ids, n_prompt, [A], sampling, A, take={li: pos for li in layers}, want_rms=True)
            end = close - n_prompt if close is not None else len(pre)
            own_pre = float(pre[end - 1]) if close is not None and end > 0 else np.nan
            a_pre = np.where(np.isin(labels[pos], T.STARTS), own_pre, np.nan)       # the model's own answer start
            rows.add(tr, out, layers, pos, labels[pos], n_prompt, close, a_pre, -1)
            key = (tr["id"], tr["k"])
            soft, c = build.exit_chance(rd, ids, n_prompt, pre, out["final"], sampling, end - 20 if close is not None else end)
            j = build.sample_exit(soft, c, key)
            for v in build.variants(rd, ids, n_prompt, pre, end, j, key):
                if v["kind"] == "own":      # its first position was read above: only the forced second one is new
                    vp, vl = [v["close"] + 2], [T.EXIT_START2]
                else:
                    vp, vl = [v["close"], v["close"] + 1, v["close"] + 2], [T.CLOSE_POS, T.EXIT_START, T.EXIT_START2]
                vout = rd.run(v["ids"], units=[A], take={li: np.asarray(vp) for li in layers}, want_rms=True)
                rows.add(tr, vout, layers, vp, vl, n_prompt, v["close"], np.full(len(vp), v["a_pre"]), v["close"] - n_prompt)
                made[f"{v['kind']}:{'pushed' if v['a_pre'] > 0 else 'quiet' if v['a_pre'] < -3 else 'between'}"] += 1
            if n % 50 == 0:
                print(f"[collect {tag}] {n}/{len(trs)} answers; closes so far {dict(made)}", flush=True)
        name = f"acts_{a.model}.{store.ACTS_TAG['box']}.{tag}.npz"
        write_json(store.exit_dir(a.case) / f"exit_unit_{a.model}.json", {"unit": a.unit})
    print(name, rows.save(store.exit_dir(a.case) / name))


# ---- fitting (numpy only) ------------------------------------------------------------------------------------
def covariance(X, w=None):
    """Covariance of the rows of X around their mean (rows weighted by w)."""
    w = np.ones(len(X)) if w is None else w
    mu = (X * w[:, None]).sum(0) / w.sum()
    Z = (X - mu) * np.sqrt(w / w.sum())[:, None]
    return Z.T @ Z


def whitened(delta, S, shrink=0.1):
    """(S + lam I)^-1 delta, unit norm; lam = shrink * mean variance (as wsp.gate.lda_direction)."""
    d = np.linalg.solve(S + shrink * np.trace(S) / len(S) * np.eye(len(S)), delta)
    return d / np.linalg.norm(d)


def indicator(X, y, w=None, ridge=1e-2):
    """t with t . x ~ y over the rows of X (weighted ridge least squares; the MLP has no bias, x supplies it)."""
    w = np.ones(len(X)) if w is None else w
    Xw = X * w[:, None]
    lam = ridge * (Xw * X).sum() / w.sum()
    return np.linalg.solve(Xw.T @ X + lam * np.eye(X.shape[1]), Xw.T @ y)


def balanced(groups):
    """Weights that make each of the given position sets count the same (a position in several sets: the first)."""
    w = np.zeros(len(groups[0]))
    for g in groups:
        g = g & (w == 0)
        if g.any():
            w[g] = 1.0 / g.sum()
    return w


def fit_layer(xl, z, s, fit, kind, a):
    """-> (d gate direction, m constant direction, t up read, what d was fitted on) of one layer, fit half only."""
    from wsp.gate import constant_direction  # noqa: PLC0415
    if kind == "exit":      # the same answer before `early` and after `late`
        diffs, own = [], z["exit_at"] < 0
        for aid in np.unique(z["aid"][fit & s["act"] & own]):
            me = fit & own & (z["aid"] == aid) & s["think"]
            late, early = me & (z["gen"] >= a.late), me & (z["gen"] < a.early)
            if late.sum() >= 3 and early.sum() >= 3:
                diffs.append(xl[late].mean(0) - xl[early].mean(0))
        d, n = whitened(np.mean(diffs, 0), covariance(xl[fit & s["silent"]]), a.shrink), len(diffs)
        if getattr(a, "exit_when", "any") == "boxed":   # + "the task asks for a boxed result", task_mix x the length read's spread
            own_think = fit & own & s["think"]
            P, N = own_think & s["task_on"], own_think & s["task_off"]
            c = whitened(xl[P].mean(0) - xl[N].mean(0), 0.5 * covariance(xl[P]) + 0.5 * covariance(xl[N]), a.shrink)
            sd_d = float((xl[own_think & s["silent"]] @ d).std())
            sd_c = float((xl[P] @ c).std())
            d = d + getattr(a, "task_mix", 0.5) * sd_d / sd_c * c
            d = d / np.linalg.norm(d)
    else:                   # positions to act on against each kind of silent position, the kinds weighing the same
        P, N = fit & s["act"], fit & s["silent"]
        w = balanced([fit & g for g in s["groups"]])[N]
        mu_n = (xl[N] * w[:, None]).sum(0) / w.sum()
        S = 0.5 * covariance(xl[P]) + 0.5 * covariance(xl[N], w)
        d, n = whitened(xl[P].mean(0) - mu_n, S, a.shrink), int(P.sum())
    m = constant_direction(xl[fit & (s["act"] | s["silent"])])
    on = fit & s["up_on"]
    both = fit & (s["up_on"] | s["up_off"])
    t = indicator(xl[both], on[both].astype(np.float64), balanced([fit & g for g in s["up_groups"]])[both], a.ridge)
    return d, m, t, n


LEAK_MARGIN = 6.0   # the sharpness at which the leak into answers is reported (build.py sets the real one)


def silu(v):
    return v / (1 + np.exp(-np.clip(v, -50, 50)))


def pct(v, qs=(1, 10, 50, 90, 99)):
    return [round(float(q), 2) for q in np.percentile(v, qs)] if len(v) else None


def cmd_fit(a):
    z = store.load_acts(a.case, a.model, a.kind, [f for f in store.FIELDS if f != "x"])
    x = store.load_acts(a.case, a.model, a.kind, ("x",))["x"]
    fit, ho = z["split"] == 0, z["split"] == 1
    s = store.SETS[a.kind](z, a)
    lab = z["label"]
    print(json.dumps({"kind": a.kind, "positions": int(len(fit)),
                      "to act on (fit / held out)": [int((fit & s["act"]).sum()), int((ho & s["act"]).sum())],
                      "silent (fit / held out)": [int((fit & s["silent"]).sum()), int((ho & s["silent"]).sum())],
                      **({"silent kinds (fit)": {k: int((fit & s[k]).sum()) for k in ("quiet", "plain", "other")}} if a.kind == "box" else {})}),
          flush=True)
    L, dim = x.shape[1], x.shape[2]
    D, M, U = (np.zeros((L, dim), np.float32) for _ in range(3))
    rows = []
    for li in (a.layers or range(L)):
        xl = x[:, li].astype(np.float64)
        D[li], M[li], U[li], n = fit_layer(xl, z, s, fit, a.kind, a)
        mx = np.maximum(xl @ M[li].astype(np.float64), 0.2)
        r = (xl @ D[li].astype(np.float64)) / mx
        u = (xl @ U[li].astype(np.float64)) / mx
        mid, sd = float(np.median(r[ho & s["silent"]])), float(r[ho & s["silent"]].std()) + 1e-9
        row = {"layer": li, "fitted_on": n,
               "gap": round((float(np.median(r[ho & s["act"]])) - mid) / sd, 2) if (ho & s["act"]).any() else None,
               "up": {"on": pct(u[ho & s["up_on"]]), "off": pct(u[ho & s["up_off"]])}}
        if a.kind == "exit":
            row["up"].update(closing_token=pct(u[ho & (lab == T.CLOSE_POS)]), answer_starts=pct(u[ho & np.isin(lab, T.STARTS)]),
                             answer_bodies=pct(u[ho & s["body"]]))
        for share in a.fire:    # the gate threshold that this share of the fit half's positions to act on reach
            th = float(np.quantile(r[fit & s["act"]], 1 - share))
            fr = {"act": round(float((r[ho & s["act"]] > th).mean()), 3), "silent": float((r[ho & s["silent"]] > th).mean()),
                  "by_task": {str(src): round(float((r[ho & s["act"] & (z["source"] == src)] > th).mean()), 3)
                              for src in np.unique(z["source"]) if (ho & s["act"] & (z["source"] == src)).any()}}
            if a.kind == "exit":
                # how strongly the unit would act in answer bodies, against where it should act, at a typical
                # sharpness (LEAK_MARGIN): a unit that acts inside answers writes closing tokens there
                g = LEAK_MARGIN / max(th - float(np.median(r[fit & s["silent"]])), 1e-6)
                act = np.clip(silu(g * (r - th) * mx) * u, 0, None)
                ref = max(float(act[ho & s["act"]].mean()), 1e-9)
                fr["leak_into_answers"] = round(float(act[ho & s["body"]].mean()) / ref, 3)
                fr["leak_late_answers"] = round(float(act[ho & s["body"] & (z["gen"] >= a.early)].mean()) / ref, 3) \
                    if (ho & s["body"] & (z["gen"] >= a.early)).any() else None
                for key, sel in (("by_length", s["think"] & ho), ("by_length_boxed", s["task_on"] & ho),
                                 ("by_length_other", s["task_off"] & ho)):
                    fr[key] = {f"{lo // 1000}k": round(float((r[sel & (z["gen"] >= lo) & (z["gen"] < hi)] > th).mean()), 3)
                               for lo, hi in BANDS if (sel & (z["gen"] >= lo) & (z["gen"] < hi)).sum() >= 20}
            else:
                first, second = lab != T.EXIT_START2, lab == T.EXIT_START2
                fr.update({k: round(float((r[ho & m_] > th).mean()), 4) for k, m_ in (
                    ("act_first", s["act"] & first), ("act_second", s["act"] & second), ("quiet_starts", s["quiet"]),
                    ("no_box_asked", s["plain"]), ("other", s["other"])) if (ho & m_).any()})
            row[f"fire{share:g}"] = fr
        rows.append(row)
        print(f"L{li:2d} gap {row['gap']} | up: " + json.dumps(row["up"]), flush=True)
        for share in a.fire:
            print(f"      fire {share:g}: " + json.dumps(row[f"fire{share:g}"]), flush=True)
    name = a.kind + (f".{a.tag}" if a.tag else "")
    np.savez(store.exit_dir(a.case) / f"reads_{a.model}_{name}.npz", D=D, M=M, U=U,
             params=json.dumps({k: getattr(a, k) for k in ("early", "late", "shrink", "ridge", "box_when", "exit_when", "task_mix")}))
    write_json(C.cdir(a.case) / "results" / f"reads_{a.model}_{name}.json", rows)
    print(f"reads written as {name!r}: unit specs {name}:LAYER:GAIN:FIRE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "fit"))
    ap.add_argument("case")
    ap.add_argument("--sets", nargs="+")
    ap.add_argument("--model", default="base")
    ap.add_argument("--exits", nargs="*", default=[], help="collect: tags of exit files to read too, e.g. budget24000_auto")
    ap.add_argument("--unit", help="collect: add this exit unit and record what a box unit will see")
    ap.add_argument("--part", default="0/1", help="collect: I/N, every N-th answer starting at I")
    ap.add_argument("--per", type=int, default=96, help="collect: thinking positions read per answer")
    ap.add_argument("--kind", default="exit", choices=("exit", "box", "brk"))
    ap.add_argument("--tag", help="fit: keep these reads beside the others as a variant, used by unit specs KIND.TAG:...")
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--late", type=int, default=24000, help="exit unit: generated tokens after which it should act")
    ap.add_argument("--early", type=int, default=16000, help="exit unit: generated tokens up to which it stays silent")
    ap.add_argument("--exit-when", default="any", choices=("any", "boxed"),
                    help="exit unit: in every task, or only in tasks whose prompt asks for a boxed result (store.sets_exit)")
    ap.add_argument("--task-mix", type=float, default=0.25,
                    help="exit unit with --exit-when boxed: weight of the task read in the gate, in units of the length "
                         "read's spread")
    ap.add_argument("--box-when", default="pushed", choices=("pushed", "always"),
                    help="box unit: act after closes the exit unit pushed, or at every answer start (store.sets_box)")
    ap.add_argument("--fire", type=float, nargs="+", default=[0.9, 0.5],
                    help="fit: report gate thresholds reached by these shares of the positions to act on")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-tokens", type=int, default=33500)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
