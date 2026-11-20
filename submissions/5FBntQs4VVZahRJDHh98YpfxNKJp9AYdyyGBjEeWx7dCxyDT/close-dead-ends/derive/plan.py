"""Step 0 - queue all GPU work that needs no decision (derive/jobs.py runs it, one job per GPU):
  round 1: the fit-set chunks of sets.py (M12 first, then L32, then H24), the base runs of the dev halves (2 samples,
  the profile's cap), and the reads of the two dead-end units (stuck: two shards, once L32 and H24 are done; loop:
  once every set is done);
  round 2, at a lower priority than every job of round 1 (a round-2 chunk starts only on a GPU that no round-1 job is
  ready for): the chunks of H24B, L32B, H24C and M12B (new samples of the fit sets' problems: the first new sample of
  H24B, the first of L32B, the second of each, the rest of H24B, then H24C, M12B last), and the round-2 reads, which
  read only the round-2 sets (the round-1 reads are reused): stuck in three shards, once every chunk of round 1, H24B,
  L32B and H24C and the round-1 reads are done or cancelled (they do not wait for M12B); loop in one, once every
  chunk of both rounds and the round-1 reads are.
Later steps are queued by hand (METHOD.md), after their checks.

  python derive/plan.py <case> [--skip-dev devB]
  python derive/plan.py <case> --times [EDIT]     run times of the round-1 reads and of Check B and Check C of EDIT
                                                  (U2's edit, default SL; from the queue), the cutoff of round 2 (the
                                                  latest start of its reads) that follows from them, as time after the
                                                  start and as time left (task.json's run times, when present), when
                                                  to cancel the round-2 chunks that can no longer end by it, and the
                                                  latest start of Check C of SL2
Dropping work on a slow machine: `python derive/jobs.py cancel <job>` on chunks that have not started; a cancelled
chunk no longer holds back the steps after it, and the reads use the chunks that exist.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = sys.executable
R2_PRIO = -1          # below every round-1 job (fit chunks and dev base runs 0, reads 1, checks 3)
CUTOFF_MIN = 220      # the round-2 reads start by 3:40 after the start of the run at the latest ...
CUTOFF_H24C = 210     # ... and by 3:30 when H24C is kept (chunks not cancelled: more runs to read and build on)
C2_END = 295          # Check C of SL2 starts only if it can end by 4:55
RUN_MIN = 360         # the run's length when task.json does not give it


def add(name, cmd, after=(), prio=0, gpus=1):
    subprocess.run([PY, str(HERE / "jobs.py"), "add", name, "--gpus", str(gpus), "--prio", str(prio),
                    "--after", ",".join(after), "--", PY, *cmd], check=True)


def round2_order(plan, chunks):
    """H24B and L32B by sample, alternating (the first new sample of each, then the second, ...), then H24C, M12B
    last."""
    groups = {}
    for s in ("H24B", "L32B"):
        if s in plan:
            by_k = {}
            for i, ch in enumerate(plan[s]["chunks"]):
                by_k.setdefault(ch["k"], []).append(chunks[s][i])
            groups[s] = [by_k[k] for k in sorted(by_k)]
    out, i = [], 0
    while any(i < len(g) for g in groups.values()):
        for g in groups.values():
            if i < len(g):
                out += g[i]
        i += 1
    return out + chunks.get("H24C", []) + chunks.get("M12B", [])


def hm(m):
    m = int(m)
    return f"{m // 60}:{m % 60:02d}"


def run_clock():
    """(start, length in min) of the run from task.json (in the working directory), or (None, RUN_MIN)."""
    try:
        run = json.loads(Path("task.json").read_text())["run"]
        return float(run["start"]), (float(run["deadline"]) - float(run["start"])) / 60
    except (OSError, ValueError, KeyError, TypeError):
        return None, RUN_MIN


def times(case, edit):
    """Run times of the round-1 jobs the round-2 tail repeats, the cutoff of round 2 (the latest start of its reads)
    and the latest start of Check C of SL2, as time after the start of the run and as time left."""
    f = Path(os.environ.get("WSP_WORK", ".work")) / "k4q" / "queue.json"
    jobs = {j["name"]: j for j in json.loads(f.read_text())["jobs"]} if f.exists() else {}

    def run_min(pat):
        ms = [(j["end"] - j["start"]) / 60 for n, j in jobs.items() if re.fullmatch(pat, n) and j["state"] == "done"
              and j.get("start") and j.get("end")]
        return max(ms) if ms else None
    e = re.escape(edit)
    r, b, c = run_min(r"reads_(stuck\d+|loop)"), run_min(rf"wc\d_{e}"), run_min(rf"C_{e}_dev[AB]")
    start, length = run_clock()

    def when(m):
        """m minutes after the start: as h:mm after the start, as time left, and as UTC clock time if known."""
        clock = (", " + time.strftime("%H:%M UTC", time.gmtime(start + 60 * m))) if start else ""
        return f"{hm(m)} after the start (when axonry-time shows {hm(length - m)} left{clock})"
    h24c = any(re.fullmatch(r"H24Cc\d+", n) and j["state"] != "cancelled" for n, j in jobs.items())
    cap = CUTOFF_H24C if h24c else CUTOFF_MIN
    lines = [f"round-1 run times (longest job of each step, waiting excluded; Check B and C of {edit}): reads "
             f"{'%.0f min' % r if r else 'not done'}, Check B shard {'%.0f min' % b if b else 'not done'}, "
             f"Check C {'%.0f min' % c if c else 'not done'}"]
    if None in (r, b, c):
        late = cap
        lines.append(f"cutoff of round 2 (latest start of its reads): {when(late)}, from the fixed cutoff "
                     f"{hm(cap)}{' (H24C is kept)' if h24c else ''}; the round-1 times are not all known yet")
    else:
        need = r + b + c + 30   # reads, builds and merge, Check B, Check C, the gaps, the upload: Check C of SL2 ends
        late = min(cap, C2_END + 10 - need)                                         # by 4:55 and U3 is up by 5:05
        lines.append(f"round 2 needs about {need:.0f} min from the start of its reads to U3 -> cutoff of round 2 "
                     f"(latest start of its reads): {when(late)}, the earlier of {hm(cap)}"
                     f"{' (H24C is kept)' if h24c else ''} and 5:05 minus {need:.0f} min")
    gen = [(j["end"] - j["start"]) / 60 for n, j in jobs.items() if re.fullmatch(r"(H24|H24B|H24C|L32B|M12B)c\d+", n)
           and j["state"] == "done" and j.get("start") and j.get("end")]
    if gen:
        lines.append(f"a round-2 chunk takes up to {max(gen):.0f} min here: from {when(late - max(gen))}, cancel the "
                     "round-2 chunks still pending (they cannot end by the cutoff); at the cutoff, cancel every "
                     "round-2 chunk not done")
    if c:
        lines.append(f"Check C of SL2 starts only if it can end by {hm(C2_END)}: it takes about {c:.0f} min (as "
                     f"Check C of {edit} did), so it starts by {when(C2_END - c)}")
    lines.append("put the cutoff in the first lines of PROGRESS.md")
    D.log(case, "round-2 timing", "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--skip-dev", nargs="*", default=[])
    ap.add_argument("--times", nargs="?", const="SL", metavar="EDIT")
    a = ap.parse_args()
    if a.times:
        times(a.case, a.times)
        return
    plan = json.loads((D.kdir(a.case) / "sets.json").read_text())
    chunks = {s: [f"{s}c{i}" for i in range(len(plan[s]["chunks"]))] for s in plan}
    order = ("M12", "L32", "H24")
    for s in order:
        for i, name in enumerate(chunks[s]):
            add(name, [str(HERE / "sample.py"), a.case, s, str(i)])
    for d in ("devA", "devB"):
        if d not in a.skip_dev:
            add(f"base_{d}", ["-m", "wsp.case", "run", a.case, "base", d, "--n", "2"])
    for i in range(2):
        add(f"reads_stuck{i}", [str(HERE / "reads.py"), a.case, "stuck", "--tags", "H24", "L32", "--shard", f"{i}/2"],
            after=chunks["L32"] + chunks["H24"], prio=1)
    add("reads_loop", [str(HERE / "reads.py"), a.case, "loop", "--tags", "M12", "H24", "L32"],
        after=chunks["M12"] + chunks["L32"] + chunks["H24"], prio=1)
    r2 = round2_order(plan, chunks)
    for name in r2:
        s, i = re.fullmatch(r"(\w+?)c(\d+)", name).groups()
        add(name, [str(HERE / "sample.py"), a.case, s, i], prio=R2_PRIO)
    if r2:
        r1_reads = ["reads_stuck0", "reads_stuck1", "reads_loop"]
        every = [n for s in plan for n in chunks[s]]
        stuck2 = [s for s in ("H24B", "L32B", "H24C") if s in plan]
        for i in range(3):                       # the stuck reads do not read M12B, so they do not wait for it
            add(f"reads2_stuck{i}", [str(HERE / "reads.py"), a.case, "stuck", "--round", "2", "--tags", *stuck2,
                                     "--shard", f"{i}/3"], after=[n for n in every if not n.startswith("M12B")] +
                r1_reads)
        add("reads2_loop", [str(HERE / "reads.py"), a.case, "loop", "--round", "2", "--tags",
                            *[s for s in ("M12B", "H24B", "L32B", "H24C") if s in plan], "--max-good", "0"],
            after=every + r1_reads)
    D.log(a.case, "plan", "queued: " + ", ".join(n for s in order for n in chunks[s]) +
          ", dev base runs, reads_stuck0/1, reads_loop" +
          (f"; round 2 (priority {R2_PRIO}): " + ", ".join(r2) + ", reads2_stuck0/1/2, reads2_loop" if r2 else ""))


if __name__ == "__main__":
    main()
