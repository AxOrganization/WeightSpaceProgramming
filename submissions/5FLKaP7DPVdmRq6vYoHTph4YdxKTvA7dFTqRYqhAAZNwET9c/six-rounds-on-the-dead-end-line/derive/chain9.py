"""The whole run as one queue of jobs (derive/jobs.py), and the rules that cut it to the time left (Section 4).

    python derive/chain9.py CASE --deadline "YYYY-MM-DD HH:MM:SS UTC"     # queue every job (once, after smoke9.py)
    python derive/chain9.py CASE --dry-run                                 # print the jobs and their commands
    python derive/chain9.py CASE --check [--apply]                         # time left, generation rate, the cuts that
                                                                           # apply now (--apply: carry them out)

Jobs run under jobs.py's detached controller (one job per GPU, pinned with CUDA_VISIBLE_DEVICES, dependencies, one log
per job in $WSP_WORK/k4q/logs/). Working files go to $WSP_WORK/k9/, edit specs to $WSP_CASES/CASE/edits/. Stages:
  K    Ks0-3 / Ts0-3 the start's answers to practice:ifsynth / practice:tuluif; Kf rows + fit -> K_05            (pk0/up0: the first upload)
  PC   PCs0-3 the start's answers to word problems and math; PCpick; PCv0-3 step values; PCf rows + fit -> PC_05
       pk1 = PC_05 + K_05 (S1, built), up1
  B    Bmk the recombined requests; Breq S1's answers to them; Bbr branching + pairs; BF1, BF14, BF21, BF28 the four
       preference fits; BF average + dose 0.35 -> B035; pk2 = [MC_05] + S1 + B035 (S2, built, room kept), up2
  OPR  OPA0-3 / OPB0-3 S2's answers to practice:ifsynth (+ word problems) / practice:tuluif; OPRf rows + fit -> OPR_05
       pk3 = S2 + OPR_05 (S3, built), up3
  OP2R OQA0-3 / OQB0-3 the same on S3; OQRf -> OP2R_05; pk4 = [MC_05] + S3 + OP2R_05 (S4), up4; G4 a short
       generation on S4 (64 + 64 practice prompts, one sample each: the last pack runs and answers)
  MC   (lowest priority, in the gaps the others leave, at most two jobs at a time) MCs0-7 the start's answers to more
       math; MCpick; MCv0-15 step values; MCf -> MC_05; it joins the first pack made after it (pk5 = MC_05 + S4, up5,
       when no earlier pack took it; G5 its short generation)
A stage whose material is too thin (fewer rows or pairs than its rule) writes no edit and the packs go on without it.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k9  # noqa: E402

P = "python"
NS, NV = 8, 16                           # MC: sampling jobs (50 problems each), step-value jobs (about 4 problems each)
GROUPS = {
    "MC": [f"MCs{i}" for i in range(NS)] + ["MCpick"] + [f"MCv{i}" for i in range(NV)] + ["MCf", "pk5", "up5", "G5"],
    "OP2R": [f"OQ{x}{i}" for x in "AB" for i in range(4)] + ["OQRf"],
    "OPR": [f"OP{x}{i}" for x in "AB" for i in range(4)] + ["OPRf"],
}
# output tokens per second per GPU the time plan assumes for the start's first sampling jobs (Section 4); the run's
# slowdown is the median of expected / measured over the finished ones
EXPECT = {"Ks": 6600, "PCs": 5600}
SLOW_NO_MC, SLOW_NO_OP2R = 0.85, 1.3   # MC only on a machine at least 15% faster than planned
DONE_MARK = ("SAMPLE_DONE", "STEP_VALUES_DONE", "GENREQ_DONE")
HANG_MARK = "exiting busy loop"          # the vLLM engine's last line when it shuts down


def jobs(case):
    """[(name, gpus, after, prio, bash command)] in the order they are queued."""
    W, E, D, ref = str(k9.work()), str(k9.edits(case)), str(k9.HERE), k9.parent_ref(case)

    def mp(m):
        return f'$({P} -c "from wsp import case; print(case.model_path(\'{case}\', \'{m}\'))" | tail -n 1)'

    def fit(model, rows, out, name, dose, maxtok, minrows):
        """gains fit on MODEL from W/ROWS.jsonl (steps = 2 x rows / 8), export NAME at DOSE; a fit that runs out of
        memory is run once more with gradient checkpointing; too few rows: no edit."""
        a = (f"--heldout {W}/{rows}.held.jsonl --rank 0 --unit-gains --gains --gain-layers 4 21 --lr 2e-3 --accum 8 "
             f"--steps $S --warmup 5 --max-tokens {maxtok} --eval-every 25")
        tr = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {P} {D}/gains_fit.py train $MP {W}/{rows}.jsonl {W}/{out} {a}"
        return (f'{{ N=$(wc -l < {W}/{rows}.jsonl) || exit 1; echo "rows $N"; '
                f'if [ "$N" -lt {minrows} ]; then echo "skip: $N rows (fewer than {minrows}): no {name}"; exit 0; fi; '
                f"MP={mp(model)}; S=$((2 * N / 8)); "
                f"{{ {tr} || {tr} --ckpt; }} && "
                f"{P} {D}/gains_fit.py export {W}/{out} {case} {name} --model $MP --scale {dose}; }}")

    def sample(model, out, sources, maxtok, i, n, seed=None, k=4):
        src = " ".join(f"--source {s}" for s in sources)
        return (f"{P} {D}/pc_sample.py {case} {model} {W}/{out} {src} --n {k} --max-tokens {maxtok} "
                f"--shard {i}/{n}" + (f" --seed {seed}" if seed is not None else ""))

    def pack(name, parts, extra=""):
        return f"{P} {D}/pack9.py {case} {name} --parts {' '.join(parts)} {extra}".strip()

    def up(name):
        return f"{P} {D}/up9.py {case} {name}"

    def gen(name):
        """the last packs: a short generation of their own (Section 5), when the pack was written"""
        return (f'[ -s {E}/{name}.json ] || {{ echo "no {name}: nothing to run"; exit 0; }}; '
                + sample(name, f"G{name}", ["ifsynth:64", "gsm8k_train:64"], 12000, 0, 1, 7, k=1)
                + f" && {P} {D}/acc9.py {W}/G{name}.0.jsonl")

    req = f"REQ_ITEMS={W}/req/train.json"
    ifs = " ".join(f"{W}/IFS.{i}.jsonl" for i in range(4))
    ans = " ".join(f"{W}/ANS.{i}.jsonl" for i in range(4))
    tul = " ".join(f"{W}/TUL.{i}.jsonl" for i in range(4))
    J = []
    # K and Part C: the start's own answers (K first: it is the shorter job)
    J += [(f"Ks{i}", 1, [], 9, sample("base", "IFS", ["ifsynth:800"], 12000, i, 4)) for i in range(4)]
    J += [(f"Ts{i}", 1, [], 9, sample("base", "TUL", ["tuluif:1024"], 12000, i, 4)) for i in range(4)]
    J += [(f"PCs{i}", 1, [], 8, sample("base", "ANS", ["gsm8k_train:768", "math_train:384"], 16384, i, 4))
          for i in range(4)]
    pcs, kss = [f"PCs{i}" for i in range(4)], [f"Ks{i}" for i in range(4)] + [f"Ts{i}" for i in range(4)]
    # K: instruction answers of practice:ifsynth (each enters the rows twice, as in our fit) and practice:tuluif,
    # keep rows from the start's finished math / word problems
    J.append(("Kf", 1, kss + pcs, 9,
              f"{P} {D}/if_rows.py {case} {W}/K.jsonl {ifs} {ifs} {tul} {ans} --sources ifsynth tuluif --mode rloo "
              f"--answer-only --keep math gsm8k --keep-n 700 --max-chars 20000 && " + fit("base", "K", "FK", "K_05", 0.5, 1024, 150)))
    J.append(("pk0", 0, ["Kf"], 9, pack("S0", ["?K_05"])))
    J.append(("up0", 0, ["pk0"], 9, up("S0")))
    J.append(("PCpick", 0, pcs, 9, f"{P} {D}/pc_pick.py {case} {W}/TR.jsonl {ans} --sources gsm8k math --problems 300 "
                                   f"--per 2"))
    J += [(f"PCv{i}", 1, ["PCpick"], 8, f"{P} {D}/pc_values.py {case} base {W}/TR.jsonl --out {W}/VAL --step 512 "
                                         f"--maxt 4096 --k 4 --budget 8192 --shard {i}/4") for i in range(4)]
    val = " ".join(f"{W}/VAL.{i}.jsonl" for i in range(4))
    J.append(("PCf", 1, [f"PCv{i}" for i in range(4)], 9,
              f"{P} {D}/pc_rows.py {case} {W}/PC.jsonl {val} --traces {W}/TR.jsonl --jump 0.5 --max-chars 20000 && "
              + fit("base", "PC", "FPC", "PC_05", 0.5, 512, 150)))
    J.append(("pk1", 0, ["Kf", "PCf"], 9, pack("S1", ["?PC_05", "?K_05"], "--build")))
    J.append(("up1", 0, ["pk1"], 9, up("S1")))
    # the branching round of #94 on S1 (its scripts, unchanged)
    J.append(("Bmk", 0, [], 9, f"mkdir -p {W}/req && {P} {D}/mkreq.py --case {case} --out {W}/req/train.json"))
    J.append(("Breq", 4, ["Bmk", "pk1"], 7, f"{P} {D}/genreq.py --case {case} --model S1 --items {W}/req/train.json "
                                             f"--out {W}/req/train.S1.jsonl --n 4 --salt mr1"))
    B = f"{W}/branch"
    J.append(("Bbr", 4, ["Breq"], 7,
              f"MP={mp('S1')}; mkdir -p {B} && {req} {P} {D}/branch_req.py plan --mode answers --rows "
              f"{W}/req/train.S1.jsonl --out {B}/ta.jsonl --k 4 --per-problem 2 && {req} {P} {D}/branch_req.py run "
              f"--model $MP --tasks {B}/ta.jsonl --out {B}/ba && {req} {P} {D}/branch_req.py pairs --mode answers "
              f"--tasks {B}/ta.jsonl --results {B}/ba --out {B}/pairs.jsonl --max-pairs 2 && wc -l {B}/pairs.jsonl"))
    tr = (f"MP={mp('S1')}; {req} torchrun --standalone --nproc_per_node 2 {D}/train_req.py --model $MP --data "
          f"{B}/pairs.jsonl --parent-ref {ref} --objective dpo --select margin --epochs 2 --accum 2 --max-len 12288")
    J.append(("BF1", 2, ["Bbr"], 7,
              f'N=$(wc -l < {B}/pairs.jsonl); if [ "$N" -lt 120 ]; then echo "fewer than 120 pairs ($N): no '
              f'branching round"; exit 0; fi; {tr} --out-spec {E}/BF1.json --seed 0'))
    for s in (14, 21, 28):
        J.append((f"BF{s}", 2, ["BF1"], 7, f'[ -s {E}/BF1.json ] || {{ echo "no BF1: skip"; exit 0; }}; '
                                           f"{tr} --out-spec {E}/BF{s}.json --seed {s} --neurons-from {E}/BF1.json"))
    J.append(("BF", 0, ["BF1", "BF14", "BF21", "BF28"], 9,
              f"MP={mp('S1')}; cd {E} && FS=$(ls BF1.json BF14.json BF21.json BF28.json 2>/dev/null | tr '\\n' ' '); "
              f'if [ -z "$FS" ]; then echo "no branching fit: no round"; exit 0; fi; echo "averaging $FS"; '
              f"{P} {D}/avg_sparse.py --specs $FS --out BAVG.json && "
              f"{P} {D}/dose_sparse.py --spec {E}/BAVG.json --start $MP --k 0.35 --out {E}/B035.json"))
    J.append(("pk2", 0, ["BF"], 9, pack("S2", ["?MC_05", "S1", "?B035"], "--reserve auto --build")))
    J.append(("up2", 0, ["pk2"], 9, up("S2")))
    # on-policy rounds
    for tag, model, after, name, dose, prio, seeds in (("OP", "S2", "pk2", "OPR_05", 0.5, 6, (51, 52)),
                                                       ("OQ", "S3", "pk3", "OP2R_05", 0.5, 5, (61, 62))):
        J += [(f"{tag}A{i}", 1, [after], prio, sample(model, f"{tag}A", ["ifsynth:1200", "gsm8k_train:400"], 12000, i, 4,
                                                      seeds[0])) for i in range(4)]
        J += [(f"{tag}B{i}", 1, [after], prio, sample(model, f"{tag}B", ["tuluif:1200"], 12000, i, 4, seeds[1]))
              for i in range(4)]
        files = " ".join(f"{W}/{tag}{x}.{i}.jsonl" for x in "AB" for i in range(4))
        J.append((f"{tag}Rf", 1, [f"{tag}{x}{i}" for x in "AB" for i in range(4)], prio,
                  f"{P} {D}/if_rows.py {case} {W}/{tag}R.jsonl {files} --sources ifsynth tuluif --mode rloo "
                  f"--answer-only --keep gsm8k --keep-n 800 --max-chars 20000 && "
                  + fit(model, f"{tag}R", f"F{tag}R", name, dose, 1024, 150)))
        if tag == "OP":
            J.append(("pk3", 0, ["OPRf"], 9, pack("S3", ["S2", "?OPR_05"], "--build")))
            J.append(("up3", 0, ["pk3"], 9, up("S3")))
        else:
            J.append(("pk4", 0, ["OQRf"], 9, pack("S4", ["?MC_05", "S3", "?OP2R_05"])))
            J.append(("up4", 0, ["pk4"], 9, up("S4")))
            J.append(("G4", 1, ["pk4"], 9, gen("S4")))
    # MC: lowest priority, two lanes of short jobs (a job of a later stage waits for at most one of them), started
    # once PC's step values are done (it never takes GPUs from them)
    mt = " ".join(f"{W}/MT.{i}.jsonl" for i in range(NS))
    pcv = [f"PCv{i}" for i in range(4)]
    J += [(f"MCs{i}", 1, pcv if i < 2 else [f"MCs{i - 2}"], 1,
           sample("base", "MT", ["math_train:400"], 16384, i, NS, 101)) for i in range(NS)]
    J.append(("MCpick", 0, [f"MCs{i}" for i in range(NS)], 9,
              f"{P} {D}/pc_pick.py {case} {W}/TRM.jsonl {mt} {ans} --sources math --problems 60 --per 2"))
    J += [(f"MCv{i}", 1, ["MCpick"] if i < 2 else [f"MCv{i - 2}"], 1,
           f"{P} {D}/pc_values.py {case} base {W}/TRM.jsonl --out {W}/VALM --step 1024 --maxt 8192 --k 4 "
           f"--budget 12000 --shard {i}/{NV}") for i in range(NV)]
    valm = " ".join(f"{W}/VALM.{i}.jsonl" for i in range(NV))
    J.append(("MCf", 1, [f"MCv{i}" for i in range(NV)], 4,
              f"{P} {D}/pc_rows.py {case} {W}/MC.jsonl {valm} --traces {W}/TRM.jsonl --jump 0.5 --max-chars 20000 && "
              + fit("base", "MC", "FMC", "MC_05", 0.5, 512, 50)))
    J.append(("pk5", 0, ["MCf", "pk4"], 9, pack("S5", ["?MC_05", "S4"], "--only-if-new")))
    J.append(("up5", 0, ["pk5"], 9, up("S5")))
    J.append(("G5", 1, ["pk5"], 9, gen("S5")))
    return J


def queue_state():
    f = Path(os.environ.get("WSP_WORK", ".work")) / "k4q" / "queue.json"
    return {j["name"]: j for j in json.loads(f.read_text())["jobs"]} if f.exists() else {}


def jobs_cli(*args):
    subprocess.run([P, str(k9.HERE / "jobs.py"), *args], check=False)


def cancel(state, names, why, apply):
    hit = [n for n in names if state.get(n, {}).get("state") == "pending"]
    if hit:
        print(f"CUT ({why}): cancel pending {', '.join(hit)}")
        if apply:
            for n in hit:
                jobs_cli("cancel", n)
            k9.note(f"cut: {why}: cancelled {', '.join(hit)}")
    return hit


def share(files, source):
    """(share right, share finished, answers) of one practice source over sample files."""
    rs = []
    for f in files:
        if Path(f).exists():
            rs += [r for r in map(json.loads, open(f)) if r.get("source") == source]
    if not rs:
        return None
    return sum(bool(r.get("correct")) for r in rs) / len(rs), sum(not r.get("cut") for r in rs) / len(rs), len(rs)


def guard(st):
    """Section 5, the breakage guard: the same practice prompts, sampled again on the next model, must not lose more
    than 5 points of right answers (or of finished ones) against the model before it."""
    W = k9.work()

    def files(tag, n, jobs_):
        return [W / f"{tag}.{i}.jsonl" for i in range(n)] if all(st.get(j, {}).get("state") == "done" for j in jobs_) \
            else None
    sets = {"start": {"ifsynth": files("IFS", 4, [f"Ks{i}" for i in range(4)]),
                      "tuluif": files("TUL", 4, [f"Ts{i}" for i in range(4)]),
                      "gsm8k": files("ANS", 4, [f"PCs{i}" for i in range(4)])},
            "S2": {"ifsynth": files("OPA", 4, [f"OPA{i}" for i in range(4)]),
                   "gsm8k": files("OPA", 4, [f"OPA{i}" for i in range(4)]),
                   "tuluif": files("OPB", 4, [f"OPB{i}" for i in range(4)])},
            "S3": {"ifsynth": files("OQA", 4, [f"OQA{i}" for i in range(4)]),
                   "gsm8k": files("OQA", 4, [f"OQA{i}" for i in range(4)]),
                   "tuluif": files("OQB", 4, [f"OQB{i}" for i in range(4)])}}
    for before, after in (("start", "S2"), ("S2", "S3")):
        for src in ("ifsynth", "gsm8k", "tuluif"):
            fb, fa = sets[before].get(src), sets[after].get(src)
            if not fb or not fa:
                continue
            b, a = share(fb, src), share(fa, src)
            if not b or not a:
                continue
            bad = a[0] < b[0] - 0.05 or a[1] < b[1] - 0.05
            print(f"guard {src}: {before} right {100 * b[0]:.1f}% finished {100 * b[1]:.1f}% ({b[2]}) -> {after} right "
                  f"{100 * a[0]:.1f}% finished {100 * a[1]:.1f}% ({a[2]})" + ("  GUARD FAIL (Section 5)" if bad else ""))


def check(case, apply):
    st = queue_state()
    left = k9.minutes_left()
    rates = k9.gen_rates([f"Ks{i}" for i in range(4)] + [f"PCs{i}" for i in range(4)])
    slow = statistics.median(EXPECT[n[:-1]] / r for n, r in rates.items()) if rates else None
    print(f"time left: {'unknown (queue with --deadline)' if left is None else f'{left:.0f} min'}; "
          f"generation rate of the start's sampling jobs: "
          + (", ".join(f"{n} {r:.0f}" for n, r in sorted(rates.items())) + f" output tok/s per GPU -> {slow:.2f} x the "
             "plan's time" if rates else "not yet measured"))
    if slow is not None and len(rates) >= 2:
        if slow > SLOW_NO_MC:
            cancel(st, GROUPS["MC"], f"pace {slow:.2f} x the planned time (MC needs <= {SLOW_NO_MC}): no math Part C", apply)
        if slow > SLOW_NO_OP2R:
            cancel(st, GROUPS["OP2R"], f"{slow:.2f} x slower than planned: no on-policy round 2", apply)
    if left is not None:
        if left < 140 and any(st.get(n, {}).get("state") == "pending" for n in GROUPS["OPR"][:8]):
            cancel(st, GROUPS["OPR"] + GROUPS["OP2R"], f"{left:.0f} min left: round 1 cannot end in time", apply)
        if left < 95 and any(st.get(n, {}).get("state") == "pending" for n in GROUPS["OP2R"][:8]):
            cancel(st, GROUPS["OP2R"], f"{left:.0f} min left: round 2 cannot end in time", apply)
        if left < 80:
            cancel(st, [n for n in GROUPS["MC"] if n.startswith(("MCs", "MCv"))], f"{left:.0f} min left: no more MC "
                   "sampling", apply)
        if left < 60:
            cancel(st, ["MCf", "pk5", "up5", "G5"], f"{left:.0f} min left: MC too late", apply)
        if left < 40:
            cancel(st, [n for n, j in st.items() if not n.startswith(("pk", "up"))], f"{left:.0f} min left: "
                   "only packing and uploading", apply)
    now = time.time()
    for n, j in st.items():           # generation jobs whose engine has shut down and that have stopped writing
        if j.get("state") != "running" or not j.get("log"):
            continue
        p = Path(j["log"])
        if not p.exists() or now - p.stat().st_mtime < 180:
            continue
        tail = p.read_text(errors="replace")[-4000:]
        if any(m in tail for m in DONE_MARK):
            print(f"HUNG after finishing: {n} (its output is complete): cancel it")
            if apply:
                jobs_cli("cancel", n)
                k9.note(f"{n}: finished but hung at shutdown; cancelled (output complete)")
        elif HANG_MARK in tail:
            print(f"HUNG before finishing: {n}: cancel it and run it again")
            if apply:
                jobs_cli("cancel", n)
                jobs_cli("retry", n)
                k9.note(f"{n}: hung at engine shutdown before finishing; started again")
    guard(st)
    counts = {}
    for j in st.values():
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    print("jobs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    failed = [n for n, j in st.items() if j["state"] == "failed"]
    if failed:
        print(f"FAILED: {', '.join(failed)}: read `python derive/jobs.py log NAME`, fix, then `jobs.py retry NAME` "
              "(or `jobs.py cancel NAME` to go on without it)")
    rec = k9.work() / "uploads.jsonl"
    if rec.exists() and rec.read_text().strip():
        last = json.loads(rec.read_text().strip().splitlines()[-1])
        print(f"last upload: {last['name']} ({', '.join(last.get('parts') or [])}) at "
              f"{time.strftime('%H:%M:%S', time.gmtime(last['time']))} UTC")
    else:
        print("no upload yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--deadline", help="the run's deadline as BRIEF.md prints it: 'YYYY-MM-DD HH:MM:SS UTC'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.deadline:
        k9.set_deadline(a.deadline)
    if a.check:
        check(a.case, a.apply)
        return
    J = jobs(a.case)
    if a.dry_run:
        for n, g, after, prio, cmd in J:
            print(f"{n:7s} gpus={g} prio={prio} after={','.join(after) or '-'}\n    {cmd}")
        return
    if not a.deadline:
        raise SystemExit("queue with --deadline (BRIEF.md): the cuts of Section 4 count the time left")
    have = queue_state()
    for n, g, after, prio, cmd in J:
        if n in have:
            continue
        jobs_cli("add", n, "--gpus", str(g), "--prio", str(prio), *(["--after", ",".join(after)] if after else []),
                 "--", "bash", "-c", cmd)
    k9.note(f"queued {len(J)} jobs; deadline {a.deadline}")
    print(f"queued {len(J)} jobs; watch them with `python derive/jobs.py status` and "
          f"`python derive/chain9.py {a.case} --check --apply`")


if __name__ == "__main__":
    main()
