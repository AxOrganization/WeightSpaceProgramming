"""Step 2 - teacher-forced reads of the start model over its own practice runs, for the two dead-end units.

  python derive/reads.py <case> stuck [--tags H24 L32] [--shard 0/1]
      every run of the tags; per run 240 thinking positions (half of them where the token ends a line) and 60
      answer positions, at random
  python derive/reads.py <case> loop [--tags M12 H24 L32] [--max-good 700] [--shard 0/1]
      runs cut while thinking whose thinking ends in a verbatim loop (dlib.loop_onset): 40 positions from the start
      of the second repetition on, and 10 of the early thinking before the loop; up to 700 runs that end correct:
      40 thinking and 20 answer positions
  python derive/reads.py <case> <role> --round 2 --tags H24B L32B H24C [M12B] ...
      round 2: the same reads of the round-2 sets only, in files of their own (reads2_<role>...), with run numbers
      from 1000000 on; gates.py --round 2 builds on the round-1 reads and these together (every fit set). A set
      without runs (all its chunks cancelled) is skipped. Refused (round 2 skipped) when fewer than half of the H24B
      chunks of sets.json exist.
At each position: the last layer's MLP input x (what a unit's gate and up rows read), the rms of the residual entering
the final norm, the number of thinking tokens so far, and how far `</think>` is below the most likely next token
(the logit gap a close has to overcome). Per run: its outcome (correct / stuck / loop / wrong / other), source, set.
Also the mean |activation| of every last-layer unit over the thinking and over the answers (how much the start uses
it).
-> $WSP_WORK/k4/<case>/reads_<role>[.<i>of<n>].npz + .json   (shards: run i of n GPUs side by side; round 2: reads2_)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

LINE_END, OTHER, ANSWER, EARLY, LOOP = 0, 1, 2, 3, 4      # position kinds
ROUND_OFFSET = 1000000                                    # run numbers of round r start at (r - 1) * ROUND_OFFSET


def prefix(rnd):
    """File prefix of the reads of round rnd: reads_ (round 1), reads2_ (round 2)."""
    return "reads" if rnd == 1 else f"reads{rnd}"


def round2_enough(case):
    """(enough, chunks of H24B that exist, chunks planned): round 2 goes on with at least half of H24B."""
    from wsp.common import RUNS  # noqa: PLC0415
    plan = json.loads((D.kdir(case) / "sets.json").read_text())
    n_plan = len(plan.get("H24B", {}).get("chunks", []))
    have = len(list((Path(RUNS) / case).glob("base__H24Bc[0-9]*.jsonl")))
    return n_plan > 0 and 2 * have >= n_plan, have, n_plan


def plan_stuck(rows, rng, a):
    return [(r, None) for r in rows]


def plan_loop(rows, rng, a):
    loops, goods = [], []
    for r in rows:
        if D.stuck_open(r):
            lo = D.loop_onset(D.thinking(r))
            if lo:
                loops.append((r, lo))
        elif r["correct"]:
            goods.append((r, None))
    goods = [goods[i] for i in rng.permutation(len(goods))[: a.max_good]]
    by = {}
    for r, _ in loops:
        by[D.src_of(r)] = by.get(D.src_of(r), 0) + 1
    print(f"[reads] {len(rows)} runs: {len(loops)} loops {by}; {len(goods)} right runs used", flush=True)
    return loops + goods


def positions_stuck(enc, r, lo, rng, a, nl):
    P, T, A, ids = enc["P"], enc["T"], enc["A"], enc["ids"]
    th = np.arange(P, P + T - 1)
    le = th[np.isin(np.asarray(ids)[th], nl)]
    ot = th[~np.isin(np.asarray(ids)[th], nl)]
    an = np.arange(P + T + 1, P + T + A - 1) if A > 1 else np.array([], int)
    out = []
    for kind, pos, k in ((LINE_END, le, a.per_trace), (OTHER, ot, a.per_trace), (ANSWER, an, a.per_trace // 2)):
        if len(pos):
            out.append((kind, rng.choice(pos, size=min(k, len(pos)), replace=False)))
    return out


def positions_loop(enc, r, lo, rng, a, nl):
    P, T, A = enc["P"], enc["T"], enc["A"]
    out = []
    if lo is not None:
        onset_c, period_c = lo
        start_c = onset_c + (period_c or 300)                     # from the second repetition on
        t0 = next((j for j, (s, e) in enumerate(enc["offs"]) if s >= start_c), None)
        if t0 is None or t0 >= T - 2:
            return []
        out.append((LOOP, rng.choice(np.arange(P + t0, P + T - 1), size=min(a.per_trace, T - 1 - t0), replace=False)))
        early = np.arange(P, P + min(T - 1, 2000))
        early = early[early < P + t0 - 200]
        if len(early):
            out.append((EARLY, rng.choice(early, size=min(a.per_trace // 4, len(early)), replace=False)))
    else:
        th = np.arange(P, P + T - 1)
        out.append((OTHER, rng.choice(th, size=min(a.per_trace, len(th)), replace=False)))
        if A > 1:
            an = np.arange(P + T + 1, P + T + A - 1)
            out.append((ANSWER, rng.choice(an, size=min(a.per_trace // 2, len(an)), replace=False)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("role", choices=["stuck", "loop"])
    ap.add_argument("--tags", nargs="+")
    ap.add_argument("--per-trace", type=int)
    ap.add_argument("--max-good", type=int, default=700)
    ap.add_argument("--max-think", type=int, default=32600)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="first N runs only (a smoke test)")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    if a.round == 1:
        a.tags = a.tags or (["H24", "L32"] if a.role == "stuck" else ["M12", "H24", "L32"])
    else:
        a.tags = a.tags or (["H24B", "L32B", "H24C"] if a.role == "stuck" else ["M12B", "H24B", "L32B", "H24C"])
        ok, have, n_plan = round2_enough(a.case)
        if not ok:
            D.log(a.case, f"reads {a.role} round 2: skipped",
                  f"only {have} of the {n_plan} H24B chunks exist (round 2 needs at least half): round 2 is skipped; "
                  "cancel the other reads2_* jobs and keep U2")
            sys.exit(3)
    a.per_trace = a.per_trace or (120 if a.role == "stuck" else 40)
    si, sn = map(int, a.shard.split("/"))
    from wsp.data import by_id  # noqa: PLC0415

    items = by_id()
    rows = [r for r in D.rows_of(a.case, a.tags) if r["id"] in items]
    t_off = (a.round - 1) * ROUND_OFFSET                  # run numbers stay unique over the reads of both rounds
    if a.limit:
        rows = rows[: a.limit]
    rng = np.random.default_rng(0)
    work = (plan_stuck if a.role == "stuck" else plan_loop)(rows, rng, a)
    pick = positions_stuck if a.role == "stuck" else positions_loop
    tf = D.TF(D.start_model(a.case))
    tok = tf.tok
    nl = np.array(sorted(i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])))
    X, N, K, R, G, TR, meta = [], [], [], [], [], [], []
    use_t, use_a, cnt_t, cnt_a = 0.0, 0.0, 0, 0
    for ti, (r, lo) in enumerate(work):
        if ti % sn != si:
            continue
        rng_t = np.random.default_rng(ti)                 # positions depend only on the run, not on the shard
        enc = tf.encode(items[r["id"]], r["text"], a.max_think, 600, offsets=(lo is not None))
        groups = pick(enc, r, lo, rng_t, a, nl)
        if not groups:
            continue
        pos = np.concatenate([p for _, p in groups])
        kind = np.concatenate([np.full(len(p), k) for k, p in groups])
        P, T = enc["P"], enc["T"]
        caps = tf.run(enc["ids"], pos, last_mlp=True, usage_span=[(P, P + T), (P + T, len(enc["ids"]))], logits=True)
        gap = np.empty(len(pos), np.float32)
        for c, lg in tf.logits(caps["z"]):
            gap[c:c + len(lg)] = (lg.max(-1).values - lg[:, tf.end_id]).cpu().numpy()
        X.append(caps["x"].cpu().numpy().astype(np.float16))
        R.append(caps["rms"].cpu().numpy().astype(np.float32))
        N.append(np.where(kind == ANSWER, -1, pos - P + 1))
        K.append(kind)
        G.append(gap)
        TR.append(np.full(len(pos), t_off + ti))
        use_t = use_t + caps["use"][0].cpu().numpy().astype(np.float64)
        use_a = use_a + caps["use"][1].cpu().numpy().astype(np.float64)
        cnt_t, cnt_a = cnt_t + T, cnt_a + len(enc["ids"]) - P - T
        meta.append({"trace": t_off + ti, "id": r["id"], "k": r["k"], "tag": r["tag"], "source": D.src_of(r),
                     "outcome": D.outcome(r), "correct": bool(r["correct"]), "cut": bool(r["cut"]),
                     "stuck_open": D.stuck_open(r), "think_tokens": int(T), "n_tokens": int(r["n_tokens"])})
        if len(meta) % 50 == 0:
            print(f"[reads] {a.role}: {len(meta)} runs read (run {ti} of {len(work)})", flush=True)
    out = D.kdir(a.case) / (f"{prefix(a.round)}_{a.role}" + (f".{si}of{sn}" if sn > 1 else ""))
    if not X:                                             # nothing of this kind in this shard: empty, well-shaped
        X, R, G = [np.zeros((0, D.model_info(tf.dir)["hidden"]), np.float16)], [np.zeros(0, np.float32)], \
            [np.zeros(0, np.float32)]
        N, K, TR = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    np.savez(str(out) + ".npz", X=np.concatenate(X), n=np.concatenate(N), kind=np.concatenate(K),
             rms=np.concatenate(R), gap=np.concatenate(G), trace=np.concatenate(TR), use_think=use_t / max(cnt_t, 1),
             use_answer=use_a / max(cnt_a, 1), think_tokens=cnt_t, answer_tokens=cnt_a)
    Path(str(out) + ".json").write_text(json.dumps({"role": a.role, "tags": a.tags, "round": a.round, "runs": meta}))
    k = np.concatenate(K)
    D.log(a.case, f"reads {a.role}{' round 2' if a.round == 2 else ''} (shard {a.shard})",
          f"{len(meta)} runs of {a.tags}; positions: " +
          ", ".join(f"{nm} {int((k == v).sum())}" for nm, v in (("line-end", LINE_END), ("other thinking", OTHER),
                                                                ("answer", ANSWER), ("early", EARLY), ("loop", LOOP))
                    if (k == v).any()) + f"; outcomes " +
          json.dumps({o: sum(m["outcome"] == o for m in meta) for o in ("correct", "stuck", "loop", "wrong", "other")}))


if __name__ == "__main__":
    main()
