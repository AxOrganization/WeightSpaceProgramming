"""Step B in one command: this entry's round, a second answer round fitted on the parent's own model.

  nohup python derive/round.py --case s --on U --units S0 --out /workspace/work/rd --minutes-left 300 > round.log 2>&1 &
  python derive/round.py --case s --on U --units S0 --out /workspace/work/rd --minutes-left 300 --status     # what is done, what runs
  python derive/round.py --case s --on U --units S0 --out /workspace/work/rd --minutes-left 300 --dry-run    # every command

  --on     the edit of the case whose model the round is fitted on: U, the parent without its last-layer pairs
           (derive/units.py), or S0, the parent itself
  --units  the edit that lists the parent's units (S0): the fit never takes one of their neurons

It runs the stages below in order on all GPUs, each once (a stage that fails is started again by the script, three
attempts in all; a finished stage is skipped when the command is started again, so a stop costs only the stage it
hit), writes one log per stage to OUT/logs/ and prints one line per stage with what it found. Nothing in it is a
choice of the agent: the numbers are fixed here, the keep rule is applied afterwards by derive/decide.py.

  req         recombined requests (derive/mkreq.py of the published entry 0009559255-0006): the sets of this
              entry's seeds, which the parent's round did not draw
  gen         the model's own answers to them, four each, 12,000-token limit (derive/genreq.py, same entry)
  pairs_req   two thinkings per request kept, the answer sampled again four times after each, a pair wherever one
              answer keeps every instruction and another does not (derive/branch.py of 0009550031-0005, unchanged,
              through derive/branch_req.py)
  if_roll     the model's answers to the practice instruction requests as they are (derive/rollouts.py)
  pairs_if    the same branching on them (one thinking per request)
  math_roll   the model's answers to MATH practice problems (another draw of problems than the parent's round
              took), two each, 12,288-token limit
  pairs_math  the same branching after two finished thinkings per problem: a pair wherever one answer written after
              that thinking is right and another is wrong
  anchors     the model's own right, finished answers to other practice math problems (for --protect, and as
              answers to keep in the fit's data)
  fit1        a preference fit on all pairs: chooses the neurons (never those of the parent's units, never the
              PROTECT neurons that the model's own math answers lean on most, never the last layer), two epochs
              (derive/train_sparse.py of 0009550031-0005, unchanged, through derive/train_req.py)
  fits        further fits of the same neurons from other seeds, two at a time
  edit        the average of the fits (derive/avg_sparse.py), delivered at the dose K (derive/dose_sparse.py) as the
              edit A. A alone is this entry's edit: it is uploaded on the parent (derive/final.py --edit A)

Time. --minutes-left is the time to the run's deadline when the command is started. plan() sizes the round from it
(at the first start; a restart keeps that plan), and before the further fits the script looks at the clock again,
because a machine can be slower than the one the times below were taken on. It prints a line `TIME ...` with what
it decided.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

K = 0.45            # the dose (METHOD.md section 1.2)
SEED = 1            # the seeds of this round; the parent's own round drew with 0
ANCHORS = 240       # practice math problems for the protected neurons
PROTECT = 4000      # neurons kept out of the fit
MATH_TIMES = 3      # how often the math answer pairs stand in the fit's data
ANCHOR_TIMES = 1    # how often the model's own right math answers (the anchors) stand in it as examples to keep

LIMIT = 4194304     # payload limit of an edit (rules.json)
M0 = "m0/edit"      # the local start of this run (derive/start.py): no ops here, so no bytes
RESERVE = 20        # minutes kept for the upload after the round
TRIES = 3           # a stage that fails is started again, up to this many attempts in all
# The sizes the round can take, largest first: (name, sets of recombined requests, practice math problems, further
# fits wanted) and the minutes its stages take on four 24 GB GPUs (the estimate plan() counts with).
PLANS = [
    ("two sets", 2, 3000, 6, {"req": 1, "gen": 48, "pairs_req": 20, "if_roll": 12, "pairs_if": 8, "math_roll": 42,
                              "pairs_math": 15, "anchors": 10, "fit1": 36, "fits": 78, "edit": 2}),
    ("one set", 1, 2000, 3, {"req": 1, "gen": 24, "pairs_req": 10, "if_roll": 12, "pairs_if": 8, "math_roll": 28,
                             "pairs_math": 10, "anchors": 10, "fit1": 24, "fits": 35, "edit": 2}),
    ("one set, fewer math problems", 1, 1200, 3, {"req": 1, "gen": 24, "pairs_req": 10, "if_roll": 12, "pairs_if": 8,
                                                  "math_roll": 17, "pairs_math": 6, "anchors": 10, "fit1": 24,
                                                  "fits": 35, "edit": 2}),
]


def plan(minutes_left):
    """The largest size whose stages and the upload after them (RESERVE) fit into the time left."""
    for name, sets, n_math, fits, nominal in PLANS:
        if minutes_left >= sum(nominal.values()) + RESERVE:
            return {"name": name, "sets": sets, "math": n_math, "fits": fits, "fit1_nominal": nominal["fit1"]}
    least = sum(PLANS[-1][4].values()) + RESERVE
    raise SystemExit(f"NO ROUND: {minutes_left} minutes are not enough for it ({least} at least). Nothing is delivered "
                     "by this entry then: say so in REPORT.md.")


def further_fits(left, fit1_minutes, wanted):
    """Before the further fits: how many of the wanted ones fit. A further fit takes about 0.72 of fit1 (the neurons
    are chosen already); they run two at a time, so n of them take ceil(n / 2) such slots, counted here as 0.8 of
    fit1 each."""
    for n in (6, 4, 3, 2):
        if n <= wanted and left - RESERVE >= (0.85 if n == 2 else 0.8 * math.ceil(n / 2)) * fit1_minutes:
            return n
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="s")
    ap.add_argument("--on", required=True, help="the edit of the case whose model the round is fitted on (U or S0)")
    ap.add_argument("--units", required=True, help="the edit that lists the parent's units (S0): never fitted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--minutes-left", type=int, required=True,
                    help="minutes to the run's deadline when this command is started: it sizes the round (see plan())")
    ap.add_argument("--shift", type=int, default=SEED, help=argparse.SUPPRESS)   # other seeds, for the author's repeats
    ap.add_argument("--pretend-elapsed", type=float, default=0.0, help=argparse.SUPPRESS)   # minutes added to the clock: tests of the two TIME points
    ap.add_argument("--name", default="A")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (Path(M0) / "edit.json").exists():
        raise SystemExit(f"{M0}/edit.json is not here: start round.py in the working directory that holds m0/ (METHOD.md section 3)")

    out = Path(a.out).resolve()
    plan_file = out / "plan.json"         # the plan of the first start holds when the command is started again
    p = json.loads(plan_file.read_text()) if plan_file.exists() else plan(a.minutes_left)
    here = Path(__file__).resolve().parent
    py = sys.executable
    g = a.gpus.split(",")
    if len(g) < 4:
        raise SystemExit("round.py plans for four GPUs (--gpus 0,1,2,3)")
    cases = Path(os.environ.get("WSP_CASES", "cases")).resolve()
    E = cases / a.case / "edits"
    for n in (a.on, a.units):
        if not (E / f"{n}.json").exists() and not a.dry_run:
            raise SystemExit(f"{E}/{n}.json is not here: derive/units.py writes S0 and U (METHOD.md section 3)")
    sets_dir = cases / a.case / "sets"
    ex = f"{sets_dir}/dev.json {sets_dir}/holdout.json"      # dev and holdout never enter the round
    R, L = out / "w", out / "logs"
    env = f"REQ_ITEMS={R}/req.json"
    model = f"$({py} -c \"from wsp import case; print(case.model_path('{a.case}', '{a.on}'))\" | tail -n 1)"
    budget = (f"$({py} -c \"from wsp.edits import spec_size; print({LIMIT} - spec_size('{M0}/edit.json')[1] - 60000)\""
              f" | tail -n 1)")           # A alone is delivered: the whole payload but a margin
    pref = f"$({py} -c \"import json; print(json.load(open('{E}/{a.on}.json'))['parent'])\")"
    no_layers = "last"     # never the last layer (the playbook's rule; the parent's last-layer pairs sit there)
    fit = (f"{env} torchrun --standalone --nproc_per_node 2 {here}/train_req.py --model {model} --data {R}/pairs.jsonl "
           f"--parent-ref {pref} --objective dpo --select margin --epochs 2 --accum 2 --max-len 12288 "
           f"--budget-bytes {budget} --protect {R}/math_anchors.jsonl --protect-n {PROTECT} --exclude-layers {no_layers}")
    merge_req = (f"{py} -c \"import json, glob; seen, all_ = set(), []\n"
                 f"for f in sorted(glob.glob('{R}/req_*.json')):\n"
                 f"    for x in json.load(open(f)):\n"
                 f"        if x['id'] not in seen:\n"
                 f"            seen.add(x['id']); all_.append(x)\n"
                 f"json.dump(all_, open('{R}/req.json', 'w')); print('recombined requests:', len(all_))\"")
    anchors = (f"{py} -c \"import json, collections\n"
               f"by = collections.defaultdict(list)\n"
               f"for l in open('{R}/aroll/rows.jsonl'):\n"
               f"    r = json.loads(l)\n"
               f"    if r.get('correct') and not r.get('cut'):\n"
               f"        by[r['id']].append(r)\n"
               f"f = open('{R}/math_anchors.jsonl', 'w')\n"
               f"for i, rs in sorted(by.items()):\n"
               f"    r = min(rs, key=lambda r: r['n_tokens'])\n"
               f"    f.write(json.dumps({{'id': i, 'type': 'anchor', 'chosen': {{'text': r['text'], 'finished': True}}}}) + chr(10))\n"
               f"f.close(); print('math anchors:', len(by), 'problems with a right, finished answer')\"")
    allg = f"CUDA_VISIBLE_DEVICES={a.gpus}"

    def one_fit(gpus, seed, name):
        return (f"CUDA_VISIBLE_DEVICES={gpus} {fit} --out-spec {E}/{name}.json --seed {seed + a.shift} "
                f"--neurons-from {E}/F1.json > {L}/fit{seed}.log 2>&1")

    def more_fits(n):
        seeds, parts, names = [14, 21, 28, 35, 42, 49][:n], [], ["F1.json"]
        for i in range(0, len(seeds), 2):
            two = seeds[i:i + 2]
            if len(two) == 2:
                parts.append(f"({one_fit(f'{g[0]},{g[1]}', two[0], f'F{two[0]}')} & "
                             f"{one_fit(f'{g[2]},{g[3]}', two[1], f'F{two[1]}')} & wait) && "
                             f"ls {E}/F{two[0]}.json {E}/F{two[1]}.json && ")
            else:
                parts.append(f"{one_fit(f'{g[0]},{g[1]}', two[0], f'F{two[0]}')} && ls {E}/F{two[0]}.json && ")
            names += [f"F{s}.json" for s in two]
        names = " ".join(names)
        return "".join(parts) + f"echo {names} > {R}/fits.txt && echo \"fits to average: {names}\""

    math_part = " ".join([str(R / "pairs_math.jsonl")] * MATH_TIMES + [str(R / "math_anchors.jsonl")] * ANCHOR_TIMES)
    J = [
        ("req", " && ".join(f"{py} {here}/mkreq.py --case {a.case} --out {R}/req_{i}.json --seed {a.shift + i}"
                            for i in range(p["sets"])) + f" && {merge_req}"),
        ("gen", f"{allg} {py} {here}/genreq.py --case {a.case} --model {a.on} --items {R}/req.json "
                f"--out {R}/req.base.jsonl --n 4 --salt mr{1 + a.shift}"),
        ("pairs_req", f"{env} {py} {here}/branch_req.py plan --mode answers --rows {R}/req.base.jsonl --out {R}/ta.jsonl "
                      f"--k 4 --per-problem 2 && "
                      f"{env} {py} {here}/branch_req.py run --model {model} --tasks {R}/ta.jsonl --out {R}/ba --gpus {a.gpus} && "
                      f"{env} {py} {here}/branch_req.py pairs --mode answers --tasks {R}/ta.jsonl --results {R}/ba "
                      f"--out {R}/pairs_req.jsonl --max-pairs 2"),
        ("if_roll", f"{py} {here}/rollouts.py pick --spec ifsynth=all,tuluif=all --exclude {ex} --out {R}/if_ids.json && "
                    f"{py} {here}/rollouts.py run --model {model} --ids {R}/if_ids.json --out {R}/ifroll --gpus {a.gpus} "
                    f"--caps ifsynth=6144,tuluif=6144 --more 0 --salt roll{a.shift}"),
        ("pairs_if", f"{py} {here}/branch.py plan --mode answers --rows {R}/ifroll/rows.jsonl --out {R}/tan.jsonl "
                     f"--sources ifsynth,tuluif --k 4 --per-problem 1 && "
                     f"{py} {here}/branch.py run --model {model} --tasks {R}/tan.jsonl --out {R}/ban --gpus {a.gpus} && "
                     f"{py} {here}/branch.py pairs --mode answers --tasks {R}/tan.jsonl --results {R}/ban "
                     f"--out {R}/pairs_if.jsonl --max-pairs 2"),
        ("math_roll", f"{py} {here}/rollouts.py pick --spec math={p['math']} --exclude {ex} --seed {11 + a.shift} --out {R}/mp_ids.json && "
                      f"{py} {here}/rollouts.py run --model {model} --ids {R}/mp_ids.json --out {R}/mproll --gpus {a.gpus} "
                      f"--caps math=12288 --more 0 --salt mp{a.shift}"),
        ("pairs_math", f"{py} {here}/branch.py plan --mode answers --rows {R}/mproll/rows.jsonl --out {R}/tam.jsonl "
                       f"--sources math --k 4 --per-problem 2 && "
                       f"{py} {here}/branch.py run --model {model} --tasks {R}/tam.jsonl --out {R}/bam --gpus {a.gpus} && "
                       f"{py} {here}/branch.py pairs --mode answers --tasks {R}/tam.jsonl --results {R}/bam "
                       f"--out {R}/pairs_math.jsonl --max-pairs 2"),
        ("anchors", f"{py} {here}/rollouts.py pick --spec math={ANCHORS} --exclude {ex} {R}/mp_ids.json --seed {11 + a.shift} --out {R}/a_ids.json && "
                    f"{py} {here}/rollouts.py run --model {model} --ids {R}/a_ids.json --out {R}/aroll --gpus {a.gpus} "
                    f"--caps math=12288 --more 0 && {anchors} && "
                    f"cat {R}/pairs_req.jsonl {R}/pairs_if.jsonl {math_part} > {R}/pairs.jsonl && "
                    f"echo \"pairs: $(wc -l < {R}/pairs_req.jsonl) recombined + $(wc -l < {R}/pairs_if.jsonl) practice requests "
                    f"+ $(wc -l < {R}/pairs_math.jsonl) math x {MATH_TIMES} + $(wc -l < {R}/math_anchors.jsonl) anchors "
                    f"= $(wc -l < {R}/pairs.jsonl); the fit takes no neuron of the layer(s) {no_layers}\""),
        ("fit1", f"CUDA_VISIBLE_DEVICES={g[0]},{g[1]} {fit} --out-spec {E}/F1.json --seed {0 + a.shift} "
                 f"--exclude-neurons-from {E}/{a.units}.json"),
        ("fits", more_fits(p["fits"])),
        ("edit", f"(cd {E} && {py} {here}/avg_sparse.py --specs $(cat {R}/fits.txt) --out AVG.json) && "
                 f"{py} {here}/dose_sparse.py --spec {E}/AVG.json --start {model} --k {K} --out {E}/{a.name}.json && "
                 f"{py} -c \"from wsp.edits import spec_size; print('payload: {a.name}', spec_size('{E}/{a.name}.json')[1], 'of {LIMIT}')\""),
    ]
    done = out / "done"
    if a.dry_run:
        for n, cmd in J + [(f"fits, when only {k} further fit(s) fit", more_fits(k)) for k in (4, 3, 2, 0) if k < p["fits"]]:
            print(f"--- {n}\n{cmd}")
        return
    if a.status:
        for n, _ in J:
            state = "done" if (done / n).exists() else ("running" if (L / f"{n}.log").exists() else "waiting")
            tail = ""
            if state != "waiting":
                lines = [x for x in (L / f"{n}.log").read_text(errors="replace").splitlines() if x.strip()]
                tail = lines[-1][:150] if lines else ""
            print(f"{state:8s} {n:10s} {tail}")
        return
    for d in (R, L, done):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    again = plan_file.exists()
    plan_file.write_text(json.dumps(p))
    print(f"ROUND PLAN{' (of the first start)' if again else ''}: {a.minutes_left} minutes left -> {p['name']}: "
          f"{p['sets']} set(s) of recombined requests, {p['math']} practice math problems, up to {1 + p['fits']} fits, "
          f"dose {K}, seeds {a.shift}, fitted on {a.case}/{a.on}", flush=True)
    for n, cmd in J:
        if (done / n).exists():
            print(f"{n}: done before", flush=True)
            continue
        left = a.minutes_left - a.pretend_elapsed - (time.time() - t0) / 60
        if n == "fits":
            try:
                f1 = float(json.loads((done / "fit1").read_text())["minutes"])
            except (OSError, ValueError, KeyError):
                f1 = float(p["fit1_nominal"])
            k = further_fits(left, f1, p["fits"])
            print(f"TIME before the further fits: {left:.0f} minutes left; fit1 took {f1:.0f}; a further fit takes about "
                  f"{0.72 * f1:.0f} and the upload {RESERVE} -> {k} further fit(s)", flush=True)
            cmd = more_fits(k)
        t = time.time()
        for attempt in range(1, TRIES + 1):
            with open(L / f"{n}.log", "w") as log:
                rc = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, executable="/bin/bash").returncode
            if rc == 0 or attempt == TRIES:
                break
            # e.g. generation engines started at the same moment can fail on a shared compile cache; what a stage has
            # finished it keeps, so the next attempt only does the rest
            (L / f"{n}.log").replace(L / f"{n}.failed{attempt}.log")
            print(f"{n}: attempt {attempt} failed (rc {rc}) after {(time.time() - t) / 60:.0f} min, see "
                  f"{L}/{n}.failed{attempt}.log; started again", flush=True)
            time.sleep(20)
        lines = [x for x in (L / f"{n}.log").read_text(errors="replace").splitlines() if x.strip()]
        show = [x for x in lines if x.lstrip().startswith(("ALL", "kind ", "[branch]", "math ", "ifsynth", "tuluif",
                                                           "pairs", "math anchors", "recombined", "payload", "[train] protected",
                                                           "fits to average", "[avg]", "[dose]", "{"))][-8:] or lines[-2:]
        mins = (time.time() - t) / 60
        print(f"{n}: {'done' if rc == 0 else 'FAILED (rc %d)' % rc} in {mins:.0f} min "
              f"({(time.time() - t0) / 60:.0f} min since the start)", flush=True)
        for x in show:
            print("    " + x[:230], flush=True)
        if rc != 0:
            print(f"ROUND STOPPED at {n}: read {L}/{n}.log, fix the cause, start the same command again", flush=True)
            sys.exit(1)
        (done / n).write_text(json.dumps({"at": time.strftime("%H:%M:%S"), "minutes": round(mins, 2)}))
    print(f"ROUND DONE: {a.name} (dose {K}; fitted on {a.on}; delivered alone on the parent)", flush=True)


if __name__ == "__main__":
    main()
