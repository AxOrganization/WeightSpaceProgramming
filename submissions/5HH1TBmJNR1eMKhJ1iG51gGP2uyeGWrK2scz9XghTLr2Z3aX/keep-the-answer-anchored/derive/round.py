"""Step H in one command: the answer round on the model the rules were built on.

  nohup python derive/round.py --case s --on S1 --out /workspace/work/rd --minutes-left 240 > round.log 2>&1 &
  python derive/round.py --case s --on S1 --out /workspace/work/rd --minutes-left 240 --status     # what is done, what runs
  python derive/round.py --case s --on S1 --out /workspace/work/rd --minutes-left 240 --dry-run    # every command

It runs the stages below in order on all GPUs, each once (a stage that fails is started again by the script, three
attempts in all; a finished stage is skipped when the command is started again, so a stop costs only the stage it hit), writes one log per stage to OUT/logs/ and prints one
line per stage with what it found. Nothing in it is a choice of the agent: the numbers are fixed here, the keep rule
is applied afterwards by derive/decide.py.

  req         recombined requests (derive/mkreq.py of the published entry 0009559255-0006), one set
  gen         the model's own answers to them, four each, 12,000-token limit (derive/genreq.py, same entry)
  pairs_req   two thinkings per request kept, the answer sampled again four times after each, a pair wherever one
              answer keeps every instruction and another does not (derive/branch.py of 0009550031-0005, unchanged,
              through derive/branch_req.py)
  if_roll     the model's answers to the practice instruction requests as they are (derive/rollouts.py)
  pairs_if    the same branching on them (one thinking per request)
  math_roll   the model's answers to MATH practice problems, two each, 12,288-token limit
  pairs_math  the same branching after two finished thinkings per problem: a pair wherever one answer written after
              that thinking is right and another is wrong (new in this entry: METHOD.md section 1.5)
  anchors     the model's own right, finished answers to other practice math problems (for --protect, and as
              answers to keep in the fit's data)
  fit1        a preference fit on all pairs: chooses the neurons (never those of the units, never the PROTECT
              neurons that the model's own math answers lean on most), two epochs (derive/train_sparse.py of 0009550031-0005, unchanged,
              through derive/train_req.py)
  fit234      three more fits of the same neurons from other seeds
  edit        the average of the fits (derive/avg_sparse.py), delivered at the dose K (derive/dose_sparse.py) as the
              edit A; S5 = ON + A (derive/stack.py)

Time. --minutes-left is the time to the run's deadline when the command is started. plan() sizes the round from it
(at the first start; a restart keeps that plan), and before fit234 the script looks at the clock again, because a
machine can be slower than the one the times below were taken on: three further fits, or two, or none, from how long
fit1 really took. It prints a line `TIME ...` with what it decided.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

K = 0.45            # the dose (METHOD.md section 1.5)
MATH = 2000         # practice math problems for the answer-stage pairs
ANCHORS = 240       # practice math problems for the protected neurons
PROTECT = 4000      # neurons kept out of the fit
MATH_TIMES = 3      # how often the math answer pairs stand in the fit's data
ANCHOR_TIMES = 1    # how often the model's own right math answers (the anchors) stand in it as examples to keep

LIMIT = 4194304     # payload limit of an edit (rules.json)
RESERVE = 20        # minutes kept for the upload after the round
TRIES = 3           # a stage that fails is started again, up to this many attempts in all
# minutes a stage takes on four 24 GB GPUs (2,000 math problems): the estimate plan() counts with
NOMINAL = {"req": 1, "gen": 24, "pairs_req": 10, "if_roll": 12, "pairs_if": 8, "math_roll": 28, "pairs_math": 10,
           "anchors": 10, "fit1": 24, "fit234": 35, "edit": 2}


def plan(minutes_left):
    """Practice math problems for the time left. The whole round takes about 2 h 45 min on four 24 GB GPUs, about a quarter of an hour
    less with 1,200 math problems; the upload after it needs RESERVE."""
    full = sum(NOMINAL.values()) + RESERVE
    if minutes_left >= full:
        return MATH
    if minutes_left >= full - 15:
        return 1200
    raise SystemExit(f"NO ROUND: {minutes_left} minutes are not enough for it ({full - 15} at least). Deliver BEST as it is.")




def further_fits(left, fit1_minutes):
    """Before fit234: how many further fits fit. A further fit takes about 0.72 of fit1 (the neurons are chosen
    already); three run as two at a time and then one, two as one pair."""
    if left - RESERVE >= 1.6 * fit1_minutes:
        return 3
    if left - RESERVE >= 0.85 * fit1_minutes:
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="s")
    ap.add_argument("--on", required=True, help="the edit of the case the round is fitted on (the rules' stack)")
    ap.add_argument("--out", required=True)

    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--minutes-left", type=int, required=True,
                    help="minutes to the run's deadline when this command is started: it sizes the round (see plan())")
    ap.add_argument("--shift", type=int, default=0, help=argparse.SUPPRESS)   # other seeds, for the author's repeats
    ap.add_argument("--pretend-elapsed", type=float, default=0.0, help=argparse.SUPPRESS)   # minutes added to the clock: tests of the two TIME points
    ap.add_argument("--math-times", type=int, default=MATH_TIMES, help=argparse.SUPPRESS)   # the author's repeats with another weight

    ap.add_argument("--anchor-times", type=int, default=ANCHOR_TIMES, help=argparse.SUPPRESS)   # the author's repeats with the anchors in the data
    ap.add_argument("--name", default="A")
    ap.add_argument("--stack", default="S5")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    plan_file = out / "plan.json"         # the plan of the first start holds when the command is started again
    if plan_file.exists():
        a.math = json.loads(plan_file.read_text())["math"]
    else:
        a.math = plan(a.minutes_left)
    here = Path(__file__).resolve().parent
    py = sys.executable
    g = a.gpus.split(",")
    if len(g) < 4:
        raise SystemExit("round.py plans for four GPUs (--gpus 0,1,2,3)")
    cases = Path(os.environ.get("WSP_CASES", "cases")).resolve()
    E = cases / a.case / "edits"
    sets_dir = cases / a.case / "sets"
    ex = f"{sets_dir}/dev.json {sets_dir}/holdout.json"
    # The test requests of the guards (step A, w/tests.json) are practice requests like the others and the round
    # draws on them too. Step H's reading on them is therefore
    # not a held-out one (it can show harm, not gain; METHOD.md says so).
    stale = sets_dir / "guard_tests.json"          # written by an earlier version of this script: mkreq.py would obey it
    keep_tests = f"rm -f {stale} && "
    ex_if = ex
    R, L = out / "w", out / "logs"
    env = f"REQ_ITEMS={R}/req.json"
    model = f"$({py} -c \"from wsp import case; print(case.model_path('{a.case}', '{a.on}'))\" | tail -n 1)"
    budget = (f"$({py} -c \"from wsp.edits import spec_size; print({LIMIT} - "
              f"spec_size('{E}/{a.on}.json')[1] - 60000)\" | tail -n 1)")
    pref = f"$({py} -c \"import json; print(json.load(open('{E}/{a.on}.json'))['parent'])\")"
    no_layers = "last"     # never the last layer (the playbook's rule)
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
        pair = (f"({one_fit(f'{g[0]},{g[1]}', 14, 'F14')} & {one_fit(f'{g[2]},{g[3]}', 21, 'F21')} & wait) && "
                f"ls {E}/F14.json {E}/F21.json && ")
        last = f"{one_fit(f'{g[0]},{g[1]}', 28, 'F28')} && ls {E}/F28.json && "
        names = {3: "F1.json F14.json F21.json F28.json", 2: "F1.json F14.json F21.json", 0: "F1.json"}[n]
        return {3: pair + last, 2: pair, 0: ""}[n] + f"echo {names} > {R}/fits.txt && echo \"fits to average: {names}\""

    J = [
        ("req", keep_tests + f"{py} {here}/mkreq.py --case {a.case} --out {R}/req_0.json --seed {a.shift} && {merge_req}"),
        ("gen", f"{allg} {py} {here}/genreq.py --case {a.case} --model {a.on} --items {R}/req.json "
                f"--out {R}/req.base.jsonl --n 4 --salt mr{1 + a.shift}"),
        ("pairs_req", f"{env} {py} {here}/branch_req.py plan --mode answers --rows {R}/req.base.jsonl --out {R}/ta.jsonl "
                      f"--k 4 --per-problem 2 && "
                      f"{env} {py} {here}/branch_req.py run --model {model} --tasks {R}/ta.jsonl --out {R}/ba --gpus {a.gpus} && "
                      f"{env} {py} {here}/branch_req.py pairs --mode answers --tasks {R}/ta.jsonl --results {R}/ba "
                      f"--out {R}/pairs_req.jsonl --max-pairs 2"),
        ("if_roll", f"{py} {here}/rollouts.py pick --spec ifsynth=all,tuluif=all --exclude {ex_if} --out {R}/if_ids.json && "
                    f"{py} {here}/rollouts.py run --model {model} --ids {R}/if_ids.json --out {R}/ifroll --gpus {a.gpus} "
                    f"--caps ifsynth=6144,tuluif=6144 --more 0 --salt roll{a.shift or ''}"),
        ("pairs_if", f"{py} {here}/branch.py plan --mode answers --rows {R}/ifroll/rows.jsonl --out {R}/tan.jsonl "
                     f"--sources ifsynth,tuluif --k 4 --per-problem 1 && "
                     f"{py} {here}/branch.py run --model {model} --tasks {R}/tan.jsonl --out {R}/ban --gpus {a.gpus} && "
                     f"{py} {here}/branch.py pairs --mode answers --tasks {R}/tan.jsonl --results {R}/ban "
                     f"--out {R}/pairs_if.jsonl --max-pairs 2"),
        ("math_roll", f"{py} {here}/rollouts.py pick --spec math={a.math} --exclude {ex} --out {R}/mp_ids.json && "
                      f"{py} {here}/rollouts.py run --model {model} --ids {R}/mp_ids.json --out {R}/mproll --gpus {a.gpus} "
                      f"--caps math=12288 --more 0 --salt mp{a.shift or ''}"),
        ("pairs_math", f"{py} {here}/branch.py plan --mode answers --rows {R}/mproll/rows.jsonl --out {R}/tam.jsonl "
                       f"--sources math --k 4 --per-problem 2 && "
                       f"{py} {here}/branch.py run --model {model} --tasks {R}/tam.jsonl --out {R}/bam --gpus {a.gpus} && "
                       f"{py} {here}/branch.py pairs --mode answers --tasks {R}/tam.jsonl --results {R}/bam "
                       f"--out {R}/pairs_math.jsonl --max-pairs 2"),
        ("anchors", f"{py} {here}/rollouts.py pick --spec math={ANCHORS} --exclude {ex} {R}/mp_ids.json --out {R}/a_ids.json && "
                    f"{py} {here}/rollouts.py run --model {model} --ids {R}/a_ids.json --out {R}/aroll --gpus {a.gpus} "
                    f"--caps math=12288 --more 0 && {anchors} && "
                    f"cat {R}/pairs_req.jsonl {R}/pairs_if.jsonl {' '.join([str(R / 'pairs_math.jsonl')] * a.math_times + [str(R / 'math_anchors.jsonl')] * a.anchor_times)} > {R}/pairs.jsonl && "
                    f"echo \"pairs: $(wc -l < {R}/pairs_req.jsonl) recombined + $(wc -l < {R}/pairs_if.jsonl) practice requests "
                    f"+ $(wc -l < {R}/pairs_math.jsonl) math{'' if a.math_times == 1 else ' x ' + str(a.math_times)}{'' if a.anchor_times == 0 else ' + $(wc -l < ' + str(R / 'math_anchors.jsonl') + ') anchors' + ('' if a.anchor_times == 1 else ' x ' + str(a.anchor_times))} = $(wc -l < {R}/pairs.jsonl); "
                    f"the fit takes no neuron of the layer(s) {no_layers}\""),
        ("fit1", f"CUDA_VISIBLE_DEVICES={g[0]},{g[1]} {fit} --out-spec {E}/F1.json --seed {0 + a.shift} "
                 f"--exclude-neurons-from {E}/{a.on}.json"),
        ("fit234", more_fits(3)),
        ("edit", f"(cd {E} && {py} {here}/avg_sparse.py --specs $(cat {R}/fits.txt) --out AVG.json) && "
                 f"{py} {here}/dose_sparse.py --spec {E}/AVG.json --start {model} --k {K} --out {E}/{a.name}.json && "
                 f"{py} {here}/stack.py --case {a.case} --name {a.stack} --edits {a.on} {a.name} && "
                 f"{py} -c \"from wsp.edits import spec_size; print('payload: {a.stack}', spec_size('{E}/{a.stack}.json')[1], 'of {LIMIT}')\""),
    ]
    done = out / "done"
    if a.dry_run:
        for n, cmd in J + [("fit234, when only two further fits fit", more_fits(2)),
                           ("fit234, when no further fit fits", more_fits(0))]:
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
    E.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    again = plan_file.exists()
    plan_file.write_text(json.dumps({"math": a.math}))
    print(f"ROUND PLAN{' (of the first start)' if again else ''}: {a.minutes_left} minutes left -> one set of recombined "
          f"requests, {a.math} practice math problems, dose {K}, on {a.case}/{a.on}", flush=True)
    for n, cmd in J:
        if (done / n).exists():
            print(f"{n}: done before", flush=True)
            continue
        left = a.minutes_left - a.pretend_elapsed - (time.time() - t0) / 60
        if n == "fit234":
            try:
                f1 = float(json.loads((done / "fit1").read_text())["minutes"])
            except (OSError, ValueError, KeyError):
                f1 = float(NOMINAL["fit1"])
            k = further_fits(left, f1)
            print(f"TIME before fit234: {left:.0f} minutes left; fit1 took {f1:.0f}; a further fit takes about {0.72 * f1:.0f} "
                  f"and the upload {RESERVE} -> {k} further fit(s)", flush=True)
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
    print(f"ROUND DONE: {a.stack} = {a.on} + {a.name} (dose {K})", flush=True)


if __name__ == "__main__":
    main()
