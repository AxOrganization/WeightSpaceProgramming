"""One command per step of the run plan in METHOD.md, so that no step depends on shell quoting or on timing.

Every GPU job is added to the toolkit's queue (tools/queue.sh: one worker per GPU, jobs survive the ssh session).
A job that needs the output of another job starts with derive/wait_for.py, so a whole round can be queued at once
and runs by itself; nothing here waits. Run from anywhere in the sandbox:

  python run.py setup               case C on the parent with all four practice sets, the parent model built,
                                    sampling shards and holdout parts written, queue workers started
  python run.py round [--round N] [--name A] [--model EDIT]
                                    queue a whole round: (round 1: the parent on dev;) the trace-sampling shards
                                    (round 1 with the parent, later rounds on fresh problems with EDIT's model);
                                    (round 1: the parent on the four holdout parts;) the fit of NAME on the traces
                                    of all rounds; one dev run per dose of NAME
  python run.py fit --name R [--units N] [--random-units | --random-change] [--doses D]
                                    queue one more fit on the traces sampled so far, and its dev runs (control,
                                    variants)
  python run.py holdout --edit E    queue E on the four holdout parts
  python run.py diagnose [--edit E] the failure table of the parent's (or E's) dev run
  python run.py show [--edit E ...] [--against E0]
                                    queue state, failed jobs, and the decision tables of finished runs (dev; holdout
                                    when all four parts of both sides exist). --against: compare with edit E0
                                    instead of the parent
  python run.py upload --edit E --report REPORT.md
                                    package E with the toolkit's packager, run axonry-check on the package and
                                    upload exactly that package only if the check passed
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("AXONRY_WORKDIR", "/workspace/work"))
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "ifsynth"]
DOSES = "0.5,0.75,1.0"
DRY = False
IFW_TRACES = True       # writing tasks (data/ifwrite.jsonl) also give training traces, not only guard runs


def paths():
    from wsp.common import CASES, REPO, RUNS, WORK  # noqa: PLC0415
    return Path(REPO) / "tools" / "queue.sh", Path(CASES), Path(RUNS), Path(WORK)


def sh(cmd, check=True, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=WORKDIR, check=check, **kw)


def qadd(queue, name, cmd):
    """Add a job unless one of that name is already pending or running (repeating a step never duplicates work)."""
    from wsp.common import WORK  # noqa: PLC0415
    q = Path(WORK) / "queue"
    if any(glob.glob(str(q / d / f"*_{name}.sh")) for d in ("pending", "running")):
        print(f"(already queued: {name})")
        return
    if DRY:
        print(f"[dry] {name}: {cmd}")
        return
    sh(["bash", queue, "add", name, cmd])


def profile_of(cases, case):
    return json.load(open(cases / case / "case.json"))["profile"]


def dose_names(name, doses):
    return [f"{name}_d{float(x):g}".replace(".", "p") for x in doses.split(",")]


def sample_files(work):
    """All finished sampling outputs of every round: practice shards and, with IFW_TRACES, the writing tasks."""
    return sorted(glob.glob(str(work / "gen" / "shard?.jsonl")) + glob.glob(str(work / "gen" / "r?s?.jsonl")) +
                  (glob.glob(str(work / "gen" / "ifw?.jsonl")) if IFW_TRACES else []))


def shard_counts(shards):
    """Problems per task type for `shards` sampling shards: the plan is three shards of about 720 problems; fewer
    shards drop problems rather than make each shard longer."""
    f = shards / 3
    return ["--gsm8k", round(650 * f), "--math", round(700 * f), "--mmlupro", round(420 * f),
            "--ifsynth", round(400 * f)]


def ifw(work, split):
    return work / "gen" / f"ifwrite_{split}.jsonl"


def ifw_run(queue, runs, work, case, model, which, first=""):
    """The writing tasks of the dev part ("ifw") or of the holdout part ("hoif") for one model, at the profile's
    full token limit: an instruction-following guard next to the practice sets."""
    out = runs / case / f"{model}__{which}.jsonl"
    if out.exists():
        return
    parts = [ifw(work, "dev")] if which == "ifw" else [ifw(work, f"ho{i}") for i in range(4)]
    wait = first
    if model != "base" and not first:
        wait = (f"python {HERE / 'wait_for.py'} {Path(paths()[1]) / case / 'edits' / (model + '.json')} "
                f"--jobs d_fit_{model.rsplit('_d', 1)[0]} && ")
    qadd(queue, f"{'e' if which == 'ifw' else 'f'}_{which}_{model}",
         f"{wait}python {HERE / 'sample.py'} --case {case} --edit {model} --items {' '.join(map(str, parts))} "
         f"--max-tokens 0 --n 1 --salt s0 --out {out}")


def build(case, model):
    sh([sys.executable, "-c", f"from wsp.case import model_path; print(model_path('{case}', '{model}'))"])


def cmd_setup(a):
    queue, cases, _, work = paths()
    task = WORKDIR / "task.json"
    base = json.load(open(task)).get("base", "qwen35-0.8b-think") if task.exists() else "qwen35-0.8b-think"
    parent = WORKDIR / "parent" / "edit" / "edit.json"
    stacked = task.exists() and json.load(open(task)).get("parent")
    if stacked and not parent.exists():
        raise SystemExit(f"{parent} is missing: copy the bundle's parent/ to {WORKDIR} first")
    if not (cases / a.case / "case.json").exists():
        cmd = [sys.executable, "-m", "wsp.case", "init", a.case, "--profile", base, "--practice", *PRACTICE,
               "--include", "cases/screening/sets/screen64.json",
               "--subset", "dev128=dev:128", "--subset", "holdout256=holdout:256"]
        if parent.exists():
            cmd += ["--parent", parent]
        sh(cmd)
    if stacked and not json.load(open(cases / a.case / "case.json")).get("parent"):
        raise SystemExit(f"case {a.case} has no parent edit: remove {cases / a.case} and run setup again")
    build(a.case, "base")
    sh([sys.executable, HERE / "make_shards.py", "--case", a.case, "--shards", a.shards, *shard_counts(a.shards)])
    sh([sys.executable, HERE / "split_set.py", "--case", a.case, "--set", "holdout", "--parts", 4, "--prefix", "ho"])
    tasks = [json.loads(line) for line in open(HERE.parent / "data" / "ifwrite.jsonl")]
    os.makedirs(work / "gen", exist_ok=True)
    for split in ("train", "dev", "ho0", "ho1", "ho2", "ho3"):
        with open(ifw(work, split), "w") as f:
            for r in tasks:
                if r["split"] == split:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "split"}) + "\n")
    print(f"writing tasks: {len(tasks)} -> {work / 'gen'}/ifwrite_*.jsonl")
    sh(["bash", queue, "start"])
    print("setup done: next `python run.py round`")


def queue_fit(queue, cases, runs, work, case, name, inputs, producers, extra, doses):
    """The fit of `name` on the given sampling files (it waits for them), then one dev run per dose. Repeating the
    step adds only what is missing."""
    meta = cases / case / "edits" / f"{name}.fit.json"
    if not meta.exists():
        data = work / "gen" / f"train_{name}.jsonl"
        wait = f"python {HERE / 'wait_for.py'} {' '.join(inputs)} --jobs {producers} --any && " if producers else ""
        qadd(queue, f"d_fit_{name}",
             f"{wait}python {HERE / 'select_traces.py'} {data} {' '.join(inputs)} && "
             f"python {HERE / 'fit_columns.py'} --model {work / 'models' / (case + '__base')} --data {data} "
             f"--case {case} --name {name} --doses {doses} --items {ifw(work, 'train')}{extra}")
        names = dose_names(name, doses)
    else:
        names = [s["name"] for s in json.load(open(meta))["specs"]]
        print(f"{name} is already fitted: only its missing dev runs are added (another fit needs another --name)")
    for nm in names:
        if (runs / case / f"{nm}__dev.jsonl").exists():
            continue
        spec = cases / case / "edits" / f"{nm}.json"
        wait = "" if meta.exists() else f"python {HERE / 'wait_for.py'} {spec} --jobs d_fit_{name} && "
        qadd(queue, f"e_dev_{nm}", f"{wait}python -m wsp.case run {case} {nm} dev")
    for nm in names:
        ifw_run(queue, runs, work, case, nm, "ifw", "" if not meta.exists() else " ")


def cmd_round(a):
    queue, cases, runs, work = paths()
    name = a.name or "ABDEFGH"[a.round - 1]
    tag = "shard" if a.round == 1 else f"r{a.round}s"
    if a.shards is None:                   # as many shards as setup wrote for round 1
        a.shards = max(1, len(glob.glob(str(work / "gen" / "shard[0-9].json"))))
    model = a.model or "base"
    if a.round > 1:
        if not a.model:
            raise SystemExit("--model EDIT is required after round 1: the accepted edit whose model samples the traces")
        sh([sys.executable, HERE / "make_shards.py", "--case", a.case, "--shards", a.shards, *shard_counts(a.shards),
            "--round", a.round, "--tag", tag])
    mdir = work / "models" / f"{a.case}__{model}"
    if not mdir.exists():
        build(a.case, model)
    first = ""
    if a.round == 1:
        if not (runs / a.case / "base__dev.jsonl").exists():
            qadd(queue, "a_base_dev", f"python -m wsp.case run {a.case} base dev")
            # the first vLLM start compiles kernels into a shared cache: the others start when it generates
            first = f"python {HERE / 'wait_for.py'} --generating --jobs a_base_dev && "
    outs = []
    for i in range(a.shards):
        ids, out = work / "gen" / f"{tag}{i}.json", work / "gen" / f"{tag}{i}.jsonl"
        if not ids.exists():
            raise SystemExit(f"{ids} missing: run `python run.py setup` first")
        outs.append(str(out))
        if not out.exists():
            qadd(queue, f"b{a.round}_{tag}{i}", f"{first}python {HERE / 'sample.py'} --case {a.case} --model {mdir} "
                                                f"--ids {ids} --salt {tag} --out {out}")
    if IFW_TRACES:
        out = work / "gen" / f"ifw{a.round}.jsonl"
        outs.append(str(out))
        if not out.exists():
            qadd(queue, f"b{a.round}_{tag}ifw", f"{first}python {HERE / 'sample.py'} --case {a.case} --model {mdir} "
                                                 f"--items {ifw(work, 'train')} --salt ifw{a.round} --out {out}")
    if a.round == 1:
        ifw_run(queue, runs, work, a.case, "base", "ifw", first)
    inputs = sorted(set(sample_files(work)) | set(outs))
    queue_fit(queue, cases, runs, work, a.case, name, inputs, f"b{a.round}_{tag}", "", a.doses)
    if a.round == 1:     # after the fit and its dose runs: the first upload waits for those, not for these
        for i in range(4):
            if not (runs / a.case / f"base__ho{i}.jsonl").exists():
                qadd(queue, f"c_base_ho{i}", f"python -m wsp.case run {a.case} base ho{i}")
        ifw_run(queue, runs, work, a.case, "base", "hoif", first)
    print(f"round {a.round} queued: fit {name}, doses {dose_names(name, a.doses)}. `python run.py show` follows it.")


def cmd_fit(a):
    queue, cases, runs, work = paths()
    inputs = sample_files(work)
    if not inputs:
        raise SystemExit("no finished sampling shard yet ($WSP_WORK/gen/shard?.jsonl): queue a round first")
    extra = (f" --units {a.units}" if a.units else "") + (" --random-units" if a.random_units else "") + \
        (" --random-change" if a.random_change else "")
    queue_fit(queue, cases, runs, work, a.case, a.name, inputs, "", extra, a.doses)


def cmd_holdout(a):
    queue, cases, runs, work = paths()
    if not (cases / a.case / "edits" / f"{a.edit}.json").exists():
        raise SystemExit(f"no edit {a.edit} in case {a.case}")
    for i in range(4):
        qadd(queue, f"f_ho{i}_{a.edit}", f"python -m wsp.case run {a.case} {a.edit} ho{i}")
    ifw_run(queue, runs, work, a.case, a.edit, "hoif", " ")


def cmd_diagnose(a):
    _, _, runs, _ = paths()
    f = runs / a.case / f"{a.edit or 'base'}__dev.jsonl"
    if not f.exists():
        raise SystemExit(f"{f} is not there yet")
    sh([sys.executable, HERE / "diagnose.py", f])


def cmd_show(a):
    queue, cases, runs, work = paths()
    sh(["bash", queue, "status"], check=False)
    strip = lambda f: Path(f).name.split("_", 1)[1]  # noqa: E731 - the job name without its time stamp
    redone = {strip(f) for d in ("pending", "running", "done") for f in glob.glob(str(work / "queue" / d / "*.sh"))}
    failed = [f for f in sorted(glob.glob(str(work / "queue" / "failed" / "*.sh"))) if strip(f) not in redone]
    if failed:
        print("FAILED jobs (read the .log beside each, then repeat the step: it adds only what is missing):")
        for f in failed:
            print("  ", f)
    ref = a.against or "base"
    rdir = runs / a.case
    found = {Path(p).name.split("__")[0] for p in glob.glob(str(rdir / "*__dev.jsonl")) +
             glob.glob(str(rdir / "*__ho?.jsonl"))} - {"base"}
    for e in (a.edit or sorted(found)):
        if e == ref:
            continue
        dev_b, dev_e = rdir / f"{ref}__dev.jsonl", rdir / f"{e}__dev.jsonl"
        if dev_b.exists() and dev_e.exists():
            gb, ge = rdir / f"{ref}__ifw.jsonl", rdir / f"{e}__ifw.jsonl"
            both = gb.exists() and ge.exists()
            print(f"\n== dev: {e} against {ref}" + ("" if both else "  (writing-task guard not finished yet: decision provisional)"))
            sh([sys.executable, HERE / "decide.py", "--base", dev_b, *([gb] if both else []),
                "--edit", dev_e, *([ge] if both else [])], check=False)
        hb = sorted(glob.glob(str(rdir / f"{ref}__ho?.jsonl")))
        he = sorted(glob.glob(str(rdir / f"{e}__ho?.jsonl")))
        if len(hb) == 4 and len(he) == 4:
            gb, ge = rdir / f"{ref}__hoif.jsonl", rdir / f"{e}__hoif.jsonl"
            both = gb.exists() and ge.exists()
            print(f"\n== holdout: {e} against {ref}" + ("" if both else "  (writing-task guard not finished yet: decision provisional)"))
            sh([sys.executable, HERE / "decide.py", "--base", *hb, *([gb] if both else []),
                "--edit", *he, *([ge] if both else [])], check=False)
        elif he:
            print(f"\n== holdout: {e}: {len(he)}/4 parts finished, {ref}: {len(hb)}/4 (no decision before all are)")


def cmd_upload(a):
    from wsp.upload import package, target, upload  # noqa: PLC0415
    task = a.task or str(WORKDIR / "task.json")
    os.chdir(WORKDIR)
    out, manifest = package(a.case, a.edit, a.report)
    want = "parent/edit.json" if json.load(open(task)).get("parent") else None
    got = json.load(open(out / "edit" / "edit.json")).get("parent")
    if got != want:      # collection rejects this; axonry-check does not test it
        raise SystemExit(f"packaged edit.json has parent {got!r}, the task needs {want!r}: NOTHING was uploaded")
    r = subprocess.run(["axonry-check", str(out / "edit"), "--task", task], capture_output=True, text=True, cwd=WORKDIR)
    print((r.stdout[:1200] + "\n...\n" + r.stdout[-200:]) if r.stdout else r.stderr[-1500:])
    if r.returncode != 0 or '"passed": true' not in r.stdout:
        raise SystemExit("axonry-check did not pass: NOTHING was uploaded")
    print("\n".join(upload(out, manifest, *target(task))))
    print(f"UPLOADED {a.edit}: the package that passed the check. Record it in PROGRESS.md.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="C")
    ap.add_argument("--dry", action="store_true", help="print the queue jobs instead of adding them")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--shards", type=int, default=3)
    p = sub.add_parser("round")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--name")
    p.add_argument("--model")
    p.add_argument("--shards", type=int, help="default: as many as setup wrote")
    p.add_argument("--doses", default=DOSES)
    p = sub.add_parser("fit")
    p.add_argument("--name", required=True)
    p.add_argument("--units", type=int)
    p.add_argument("--random-units", action="store_true")
    p.add_argument("--random-change", action="store_true")
    p.add_argument("--doses", default=DOSES)
    p = sub.add_parser("holdout")
    p.add_argument("--edit", required=True)
    p = sub.add_parser("diagnose")
    p.add_argument("--edit")
    p = sub.add_parser("show")
    p.add_argument("--edit", nargs="*")
    p.add_argument("--against")
    p = sub.add_parser("upload")
    p.add_argument("--edit", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--task")
    a = ap.parse_args()
    global DRY
    DRY = a.dry
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
