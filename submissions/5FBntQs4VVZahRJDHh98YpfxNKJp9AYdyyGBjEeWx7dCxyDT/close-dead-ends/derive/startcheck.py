"""Step 1 - what the start does on its own practice runs, before anything is built: the failure, measured.

  python derive/startcheck.py <case> [--tags M12 L32 H24]
Per set and source: right, finished wrong, cut while thinking without a loop ("stuck"), cut in a verbatim loop
("loop"), mean tokens; how often a run that is still thinking at a given depth ends right or never finishes (do the
two kinds coexist at the same depth?); how many runs each dead-end unit can learn from; the measured generation rate
and the speed branches of the time plan (round 1, and round 2 with the round-2 chunks it cancels). Go / no-go per part
(METHOD.md, step 1). The stuck unit's verdict waits ("pending") while chunks of L32 or H24 are still queued or
running; a set none of whose chunks ran (all cancelled) is reported as such, and the verdict uses the sets that exist.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def unfinished(sets):
    """Jobs of these fit sets still pending or running in the queue of derive/jobs.py (None: no queue yet)."""
    f = Path(os.environ.get("WSP_WORK", ".work")) / "k4q" / "queue.json"
    if not f.exists():
        return None
    jobs = json.loads(f.read_text())["jobs"]
    return [j["name"] for j in jobs if j["state"] in ("pending", "running")
            and any(re.fullmatch(rf"{s}c\d+", j["name"]) for s in sets)]


BRANCHES = ((1800, "full"), (1300, "lean"), (900, "reduced"), (600, "slow"), (0, "survival"))
BRANCHES2 = ((3900, "full, with M12B and H24C"), (3300, "full, with M12B and half of H24C"), (2900, "full, with M12B"),
             (2400, "full"), (2250, "shrunk"), (0, "skipped"))                                     # round 2
KEEP_R2_MIN = 2900       # from this R on, round 2 goes on through a remedy of round 1; below it a harm failure in
                         # round 1's Check B or C ends round 2 (METHOD.md, Steps 3 and 4)


def round2_cancel(branch):
    """Round-2 jobs a round-2 speed branch cancels (those that have not started)."""
    return {"full, with M12B and H24C": [],
            "full, with M12B and half of H24C": ["H24Cc6-c11"],
            "full, with M12B": ["H24Cc0-c11"],
            "full": ["H24Cc0-c11", "M12Bc0-c2"],
            "shrunk": ["L32Bc0-c3", "H24Bc6-c11", "H24Cc0-c11", "M12Bc0-c2"],
            "skipped": ["every H24B, L32B, H24C and M12B chunk", "reads2_stuck0-2", "reads2_loop"]}[branch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--tags", nargs="+", default=["M12", "L32", "H24"])
    a = ap.parse_args()
    lines, have = [], {}
    for t in a.tags:
        try:
            rows = D.rows_of(a.case, [t])
        except SystemExit:
            lines.append(f"{t}: no runs yet")
            continue
        have[t] = rows
        lines.append(f"{t} ({len(rows)} runs):")
        for s in sorted({D.src_of(r) for r in rows}):
            rs = [r for r in rows if D.src_of(r) == s]
            oc = [D.outcome(r) for r in rs]
            lines.append(f"  {s:8} n {len(rs):4}  right {100 * np.mean([o == 'correct' for o in oc]):5.1f}%  "
                         f"wrong {100 * np.mean([o == 'wrong' for o in oc]):5.1f}%  "
                         f"stuck {100 * np.mean([o == 'stuck' for o in oc]):5.1f}%  "
                         f"loop {100 * np.mean([o == 'loop' for o in oc]):5.1f}% ({sum(o == 'loop' for o in oc)})  "
                         f"tokens {np.mean([r['n_tokens'] for r in rs]):6.0f}")
    long_sets = [t for t in ("L32", "H24") if t in have]
    if long_sets:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(D.start_model(a.case))
        rows = [r for t in long_sets for r in have[t]]
        closed = [r for r in rows if D.closed(r)]
        lens = tok([D.thinking(r) for r in closed], add_special_tokens=False)["input_ids"] if closed else []
        depth = {id(r): len(x) for r, x in zip(closed, lens)}
        think = [(depth.get(id(r), r["n_tokens"]), r) for r in rows]
        lines.append(f"runs still thinking at a depth ({'+'.join(long_sets)}): share that ends right / never finishes")
        for d in (6000, 9000, 12000, 16000, 20000, 24000, 28000):
            alive = [r for n, r in think if n >= d]
            if len(alive) >= 10:
                right = 100 * np.mean([r["correct"] for r in alive])
                lines.append(f"  {d // 1000:2d}k: {len(alive):4} runs, right {right:5.1f}%, "
                             f"never finish {100 * np.mean([D.stuck_open(r) for r in alive]):5.1f}%")
    go = {}
    wait = unfinished(("L32", "H24"))
    gone = [t for t in ("L32", "H24") if t not in have]           # sets none of whose chunks has run
    pending = bool(wait) or (wait is None and bool(gone))         # no queue yet: a missing set is still to come
    if long_sets:
        src = have.get("L32") or have.get("H24")
        m = [r for r in src if D.src_of(r) == "math"]
        share = 100 * np.mean([D.stuck_open(r) for r in m]) if m else 0.0
        doomed = sum(D.stuck_open(r) for t in long_sets for r in have[t])
        so_far = (f"{share:.1f}% of the hard calculation runs never finish (go at >= 25%); {doomed} such runs to "
                  f"learn from")
    if pending:
        lines.append("stuck unit: " + (so_far + " so far " if long_sets else "") + "-> pending (decided once L32 and "
                     f"H24 are done; still queued or running: {', '.join(wait or gone)})")
    elif not long_sets:
        go["stuck"] = False
        lines.append("stuck unit: no runs of L32 or H24 (their chunks were cancelled or failed) -> NO-GO")
    else:
        go["stuck"] = share >= 25 and doomed >= 100
        lines.append(f"stuck unit: {so_far} -> {'GO' if go['stuck'] else 'NO-GO'}" +
                     "".join(f"; {t}: no runs (cancelled or failed)" for t in gone))
    if "M12" in have:
        rows = have["M12"]
        share = 100 * np.mean([D.outcome(r) == "loop" for r in rows])
        allr = [r for t in have for r in have[t]]
        per = {s: sum(D.outcome(r) == "loop" for r in allr if D.src_of(r) == s)
               for s in sorted({D.src_of(r) for r in allr})}
        go["loop"] = share >= 2
        thin = [s for s in ("gsm8k", "mmlupro", "ifsynth", "tuluif") if s in per and per[s] < 30]
        lines.append(f"loop unit: {share:.1f}% of M12 runs end in a verbatim loop (go at >= 2%) -> "
                     f"{'GO' if go['loop'] else 'NO-GO'}; loops per source over all sets {per}" +
                     (f"; fewer than 30 in {thin} (Check A judges a source only with enough looping runs)"
                      if thin else ""))
    rf = D.kdir(a.case) / "rates.jsonl"
    if rf.exists():
        rs = [json.loads(x) for x in rf.read_text().splitlines() if x.strip()]
        r = float(np.median([x["tok_s_per_gpu"] for x in rs]))
        branch = next(b for lim, b in BRANCHES if r >= lim)
        branch2 = next(b for lim, b in BRANCHES2 if r >= lim)
        cancel2 = round2_cancel(branch2)
        lines.append(f"generation rate: median {r:.0f} output tok/s per GPU over {len(rs)} chunks -> speed branch "
                     f"'{branch}' (METHOD.md, step 1; cancel only chunks that have not started); round 2: "
                     f"'{branch2}'" + (f" (cancel {', '.join(cancel2)})" if cancel2 else " (nothing to cancel)") +
                     ("" if branch2 == "skipped" else
                      "; round 2 goes on through a remedy of round 1" if r >= KEEP_R2_MIN else
                      "; a harm failure in round 1's Check B or C ends round 2"))
    D.save_json(D.kdir(a.case) / "startcheck.json", go)
    D.log(a.case, "start check", "\n".join(lines))


if __name__ == "__main__":
    main()
