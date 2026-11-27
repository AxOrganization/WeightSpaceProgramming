"""Derive a reasoning-loop edit that guards productive repeated checks."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time

CASE = "b_loop_exit"
PROFILE = "qwen35-2b-think"
SOURCE = Path(__file__).resolve().parents[1]
MATH = ("math_train", "gsm8k_train")
PRACTICE = (*MATH, "ifsynth", "mmlupro")
STRENGTHS = (0.5, 1.0)
GENERATION_CAP = 16384
PROBE_CAP = 32768
FIT_MINIMUMS = (12, 12)
CALIBRATION_MINIMUMS = (8, 8)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def practice_plan(data):
    excluded = set()
    rng = random.Random("loop-exit-protect:practice")

    def draw(source, count):
        rows = sorted(data.load(f"practice:{source}"), key=lambda row: row["id"])
        rng.shuffle(rows)
        selected = []
        for row in rows:
            fp = data.fingerprint(row["problem"])
            if fp in excluded:
                continue
            excluded.add(fp)
            selected.append(row)
            if len(selected) == count:
                return selected
        raise ValueError(f"Too few distinct practice items from {source}")

    def math_draw(count):
        return [r for pair in zip(*(draw(s, count) for s in MATH), strict=True) for r in pair]

    plan = {
        "fit": math_draw(64),
        "calibration": math_draw(32),
        "holdout_math": math_draw(32),
        "holdout_instructions": draw("ifsynth", 32),
        "holdout_choice": draw("mmlupro", 32),
    }
    ids = [row["id"] for rows in plan.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Practice splits share an item ID")
    return plan


def task_parent(task, pinned):
    from wsp.edits import ARRAY_FIELDS, parent_path

    record = read_json(task)
    if record.get("schema") != "axonry.task/1" or record.get("base") != PROFILE or record.get("base_name") != pinned:
        raise ValueError("This method requires a task for the pinned 2B Thinking base")
    starts = record.get("starts")
    if not isinstance(starts, list) or any(not isinstance(start, dict) for start in starts):
        raise ValueError("The task has no valid published starts")
    parents = [start.get("sid") for start in starts if start.get("role") == "parent"]
    if len(parents) != 1 or not isinstance(parents[0], str) or not re.fullmatch(r"[0-9]{10}-[0-9]{4}", parents[0]):
        raise ValueError("The task must resolve exactly one published same-base champion as the parent")
    sid = parents[0]
    judge = record.get("judge", {})
    if not isinstance(judge, dict) or ("champion" in judge and judge["champion"] != sid):
        raise ValueError("The resolved parent disagrees with the task's kickoff champion")
    deadline = record["run"]["deadline"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("Invalid or elapsed task deadline")
    root = task.resolve().parent / sid / "edit"
    if not root.is_dir() or root.resolve() != root or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("The task must supply the champion's local published edit tree")
    spec = root / "edit.json"
    seen = set()
    location = spec
    while location is not None:
        if location in seen or not location.is_relative_to(root) or not location.is_file():
            raise ValueError("The champion edit chain is incomplete, cyclic, or outside its task directory")
        seen.add(location)
        block = read_json(location)
        if block.get("base") != PROFILE:
            raise ValueError("The champion edit chain contains a different model base")
        for op in block["ops"]:
            for field in ARRAY_FIELDS:
                ref = op.get(field)
                if isinstance(ref, str):
                    array = (location.parent / ref.partition("#")[0]).resolve()
                    if not array.is_relative_to(root) or not array.is_file():
                        raise ValueError("The champion edit references an array outside its task directory")
        location = parent_path(block, location.parent)
        if location is None and block.get("parent") != pinned:
            raise ValueError("The champion edit chain does not reach the pinned raw base")
    return sid, spec, deadline


def init_case(sid, parent_spec):
    from wsp import case

    subprocess.run([sys.executable, "-B", "-m", "wsp.case", "init", CASE, "--profile", PROFILE,
                    "--contrast", "loop", "--parent", str(parent_spec), "--practice", *PRACTICE,
                    "--dev", "0", "--holdout", "0"], check=True)
    if case.conf(CASE)["parent"] != f"{sid}/edit/edit.json":
        raise ValueError("The toolkit case did not bind the exact champion SID")
    if case.model_digest(CASE, "base") != case._sources_digest(parent_spec):
        raise ValueError("The champion's edit chain changed when copied into the case")
    return case.cdir(CASE)


def timeout(_signum, _frame):
    raise TimeoutError("The task deadline expired")


def progress(work, message):
    path = work / "PROGRESS.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(message + "\n", encoding="utf-8")
    temporary.replace(path)
    print(message, flush=True)


def read_differences(probe, pairs, minimum, want):
    import numpy as np

    if len(pairs) < minimum or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError(f"Need at least {minimum} pairs from distinct practice questions")
    pairs = sorted(pairs, key=lambda p: hashlib.sha256(p["id"].encode()).digest())
    bad = probe.read([p["bad"] for p in pairs], want=want, max_tokens=PROBE_CAP)
    good = probe.read([p["good"] for p in pairs], want=want, max_tokens=PROBE_CAP)
    valid = (np.asarray(bad["ntok"]) == 64) & (np.asarray(good["ntok"]) == 64)
    if valid.sum() < minimum:
        raise ValueError(f"Need {minimum} paired windows with exactly 64 measured reasoning tokens")
    return {name: bad[name][valid] - good[name][valid] for name in want}


def derive(champion, profile, tokenizer, plan, work):
    import gc

    import torch
    from wsp.probe import Probe

    from contrasts import collect
    from projection import choose_edit

    fit = collect(champion, profile, tokenizer, plan["fit"], scope="fit")
    calibration = collect(champion, profile, tokenizer, plan["calibration"], scope="calibration")
    progress(work, f"Found loop-exit and verification pair counts {tuple(map(len, fit))} for fit "
                  f"and {tuple(map(len, calibration))} for calibration.")
    for label, pools, minimums in (("fit", fit, FIT_MINIMUMS), ("calibration", calibration, CALIBRATION_MINIMUMS)):
        for kind, pairs, minimum in zip(("loop", "verification"), pools, minimums, strict=True):
            if len(pairs) < minimum:
                raise ValueError(f"Need {minimum} {label} {kind} pairs, obtained {len(pairs)}")
    probe = Probe(str(champion), "cuda:0")
    try:
        fit_loop, fit_check = [
            read_differences(probe, pairs, minimum, ("resid", "mlp"))
            for pairs, minimum in zip(fit, FIT_MINIMUMS, strict=True)
        ]
        cal_loop, cal_check = [
            read_differences(probe, pairs, minimum, ("resid",))
            for pairs, minimum in zip(calibration, CALIBRATION_MINIMUMS, strict=True)
        ]
        result = choose_edit(fit_loop, fit_check, cal_loop, cal_check,
                             [probe.down_proj(layer) for layer in range(len(probe.layers) - 1)])
    finally:
        del probe
        gc.collect()
        torch.cuda.empty_cache()
    result["pair_counts"] = {name: len(values["resid"]) for name, values in
                             (("fit_loop", fit_loop), ("fit_verification", fit_check),
                              ("calibration_loop", cal_loop), ("calibration_verification", cal_check))}
    return result


def write_edit(raw, directory, name, layer, cols, direction, alpha, *, parent):
    import numpy as np
    from wsp.edits import decoder_prefix, tensor_index

    index = tensor_index(raw)
    tensor = f"{decoder_prefix(index)}layers.{layer}.mlp.down_proj.weight"
    u = np.asarray(direction, dtype=np.float32)
    columns = np.asarray(cols, dtype=np.int64)
    if (tensor not in index or u.ndim != 1 or not np.isfinite(u).all()
            or not np.isclose(np.linalg.norm(u), 1.0, atol=1e-5)
            or columns.ndim != 1 or not 1 <= len(columns) <= 64
            or len(np.unique(columns)) != len(columns) or columns.min() < 0 or alpha not in STRENGTHS):
        raise ValueError("Invalid sparse direction ablation")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{name}.npz", u=u, columns=columns)
    spec = {"name": name, "base": PROFILE, "parent": os.path.relpath(parent, directory), "ops": [
        {"op": "dir_ablate", "side": "out", "tensor": tensor, "u": f"{name}.npz#u",
         "cols": f"{name}.npz#columns", "alpha": alpha},
    ]}
    write_json(directory / f"{name}.json", spec)


def scored(model, items, profile, n, salt):
    from wsp.gen import generate

    rows = generate(str(model), items, profile, n=n, max_tokens=GENERATION_CAP, salt=salt)
    need = {(item["id"], k) for item in items for k in range(n)}
    if len(rows) != len(need) or {(row["id"], row["k"]) for row in rows} != need:
        raise ValueError("Missing or duplicate practice generations")
    if any(row["kind"] == "grade_error" for row in rows):
        raise ValueError("A practice grader failed")
    return {(row["id"], row["k"]): row for row in rows}


def keep_keys(rows, keys):
    return {key: rows[key] for key in keys}


def compare(base, edited, tokenizer, prompts):
    from contrasts import has_persistent_loop
    from wsp.stats import paired

    if not base or base.keys() != edited.keys():
        raise ValueError("Comparison requires full matched practice results")
    result = paired(list(base.values()), list(edited.values()))
    for label, rows in (("base", base), ("edited", edited)):
        loops = sum(has_persistent_loop(row, tokenizer, prompt=prompts[row["id"]]) for row in rows.values())
        result[f"{label}_loop_share"] = loops / len(rows)
        result[f"{label}_other_cut_share"] = (sum(row["cut"] for row in rows.values()) - loops) / len(rows)
    return result


def accurate(result):
    return result["gain"] >= -2 * result["se"]


def check_holdout(champion, candidate, control, plan, profile, tokenizer):
    from contrasts import has_verification, prompts_for
    from wsp.stats import paired

    groups = {key: rows for key, rows in plan.items() if key.startswith("holdout_")}
    items = [row for rows in groups.values() for row in rows]
    prompts = prompts_for(items, tokenizer)
    base, edited, unprotected = [
        scored(model, items, profile, 2, "loop-exit-holdout") for model in (champion, candidate, control)
    ]
    reports = {}
    for name, rows in groups.items():
        keys = {(row["id"], k) for row in rows for k in range(2)}
        reports[name] = compare(keep_keys(base, keys), keep_keys(edited, keys), tokenizer, prompts)
    cohort = {key for key, row in base.items() if row["correct"] and not row["cut"]
              and has_verification(row["text"], tokenizer, prompt=prompts[row["id"]])}
    num_verification = len({key[0] for key in cohort})
    scores = {
        "protected_vs_champion": compare(base, edited, tokenizer, prompts),
        "groups": reports,
        "baseline_verified_check_questions": num_verification,
    }
    passed_checks = False
    if num_verification >= 8:
        b, e, u = (keep_keys(rows, cohort) for rows in (base, edited, unprotected))
        scores["verification_protected_vs_champion"] = paired(list(b.values()), list(e.values()))
        scores["verification_protected_vs_unprotected"] = paired(list(u.values()), list(e.values()))
        scores["retained_successful_checks"] = {
            label: sum(row["correct"] and not row["cut"]
                       and has_verification(row["text"], tokenizer, prompt=prompts[row["id"]])
                       for row in rows.values())
            for label, rows in (("champion", b), ("protected", e), ("unprotected", u))
        }
        scores["verification_comparison_draws"] = len(cohort)
        passed_checks = accurate(scores["verification_protected_vs_champion"])
    lengths = sorted(row["n_tokens"] for row in base.values() if row["correct"] and not row["cut"])
    if lengths:
        median = lengths[len(lengths) // 2]
        long_keys = [key for key, row in base.items() if row["correct"] and not row["cut"]
                     and row["n_tokens"] >= median]
        scores["previously_correct_long"] = {"minimum_tokens": median, "draws": len(long_keys),
                                             "retained": sum(edited[key]["correct"] for key in long_keys)}
    overall = scores["protected_vs_champion"]
    passed = (overall["edited_loop_share"] < overall["base_loop_share"]
              and accurate(overall) and all(map(accurate, reports.values())) and passed_checks)
    scores["passed"] = passed
    return scores


def deliver(raw, work, task, name, upload, *, parent_sid):
    from wsp.edits import spec_size
    from wsp.rules import RULES, payload_limit

    report = work / "REPORT.md"
    args = [sys.executable, "-B", "-m", "wsp.upload", CASE, name,
            "--start", parent_sid, "--task", str(task), "--report", str(report)]
    subprocess.run([*args, "--dry-run"], check=True)
    bundle = work / "wsp" / "deliver" / f"{CASE}__{name}"
    manifest = read_json(bundle / "manifest.json")
    entries = manifest["files"]
    wanted = {"edit/edit.json", "edit/edit.npz", "edit/REPORT.md"}
    if (manifest["schema"] != RULES["upload"]["schema"] or len(entries) != len(wanted)
            or {item["path"] for item in entries} != wanted):
        raise ValueError("Unexpected task upload manifest")
    for item in entries:
        path = bundle / item["path"]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError("Task upload contents changed")
    edit = bundle / "edit" / "edit.json"
    spec = read_json(edit)
    if (spec["base"] != PROFILE or spec["parent"] != parent_sid
            or len(spec["ops"]) != 1 or spec["ops"][0]["op"] != "dir_ablate"):
        raise ValueError("The export must carry one new direction ablation on the champion SID")
    tensors, array_bytes = spec_size(edit)
    if tensors != 1 or array_bytes > payload_limit(raw):
        raise ValueError("Task edit exceeds the format limits")
    checker = shutil.which("axonry-check")
    if not checker:
        raise ValueError("A completed task requires the official numerical admission checker")
    subprocess.run([checker, str(edit.parent), "--task", str(task), "--base-dir", str(raw)], check=True)
    result = {"manifest_sha256": sha256(bundle / "manifest.json"), "parent_sid": parent_sid,
              "numeric_admission": True, "array_bytes": array_bytes}
    write_json(work / "export.json", result)
    if upload:
        subprocess.run(args, check=True)
    return result


def run(work, task, plan, upload):
    from transformers import AutoTokenizer
    from wsp import case, profiles

    from contrasts import prompts_for

    pinned = profiles.base_name(PROFILE)
    sid, parent_spec, deadline = task_parent(task, pinned)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    raw, profile = profiles.base_dir(PROFILE), profiles.get(PROFILE)
    progress(work, f"Loading the task's published champion {sid}.")
    directory = init_case(sid, parent_spec)
    parent = directory / case.conf(CASE)["parent"]
    parent_digest = case.model_digest(CASE, "base")
    write_json(work / "parent.json", {"parent_sid": sid, "raw_base": pinned, "parent_source_digest": parent_digest})
    champion = case.model_path(CASE, "base")
    tokenizer = AutoTokenizer.from_pretrained(str(champion), local_files_only=True)
    progress(work, "Collecting natural loop exits and successful verification controls from the champion.")
    edit = derive(champion, profile, tokenizer, plan, work)
    diagnostics = {**edit["diagnostics"], "selected_layer": int(edit["layer"]),
                   "selected_columns": len(edit["cols"]), "pair_counts": edit["pair_counts"]}
    write_json(work / "derivation.json", diagnostics)
    progress(work, "Calibrating the selected sparse edit.")
    items = plan["calibration"]
    prompts = prompts_for(items, tokenizer)
    base = scored(champion, items, profile, 2, "loop-exit-calibration")
    attempts = {}
    for index, alpha in enumerate(STRENGTHS):
        name = f"protected_{index}"
        write_edit(raw, directory / "edits", name, edit["layer"], edit["cols"], edit["direction"], alpha, parent=parent)
        candidate = case.model_path(CASE, name)
        result = compare(base, scored(candidate, items, profile, 2, "loop-exit-calibration"), tokenizer, prompts)
        attempts[name] = {"strength": alpha, **result}
        write_json(work / "calibration.json", attempts)
        if result["edited_loop_share"] < result["base_loop_share"] and accurate(result):
            break
    else:
        raise ValueError("No strength reduced persistent loops while preserving calibration correctness")
    control = "unprotected"
    write_edit(raw, directory / "edits", control, edit["layer"], edit["cols"], edit["unprotected_direction"], alpha,
               parent=parent)
    case_doc = read_json(directory / "case.json")
    write_json(directory / "case.json", {**case_doc, "frozen": name})
    progress(work, "Checking the frozen edit and matched control on disjoint practice questions.")
    holdout = check_holdout(champion, candidate, case.model_path(CASE, control), plan, profile, tokenizer)
    write_json(work / "holdout.json", holdout)
    if not holdout["passed"]:
        raise ValueError("The frozen edit did not meet the practice holdout criteria")
    summary = {"parent_sid": sid, "derivation": diagnostics, "calibration": attempts, "holdout": holdout}
    (work / "REPORT.md").write_text("# Loop exit with verification protection\n\n"
                                    f"Derived and checked on the task's published champion `{sid}` using practice questions.\n\n"
                                    "```json\n" + json.dumps(summary, indent=2, allow_nan=False) + "\n```\n")
    progress(work, "Preparing the sparse champion-parent edit and checking numerical admission.")
    exported = deliver(raw, work, task, name, upload, parent_sid=sid)
    write_json(work / "run-receipt.json", {"profile": PROFILE, "raw_base": pinned, "parent_sid": sid,
                                         "parent_source_digest": parent_digest, "candidate": name,
                                         "candidate_source_digest": case.model_digest(CASE, name),
                                         "deliverable": exported, "upload_completed": upload})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if (not args.plan_only and not args.task) or (args.plan_only and args.upload):
        parser.error("An official task is required for derivation or upload")
    gpus = args.gpus.split(",")
    if any(not re.fullmatch(r"[0-9]+", gpu) for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("Use distinct numeric GPU IDs")
    work = args.work_dir.resolve()
    if work.is_relative_to(SOURCE):
        parser.error("Use a work directory outside the source package")
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(WSP_CASES=str(work / "cases"), WSP_WORK=str(work / "wsp"),
                      CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    from wsp import data

    task = args.task.resolve(strict=True) if args.task else None
    with (work / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = practice_plan(data)
        write_json(work / "practice-plan.json", {name: [row["id"] for row in rows] for name, rows in plan.items()})
        if args.plan_only:
            print({key: len(rows) for key, rows in plan.items()})
            return
        marker = work / "run-started.json"
        if marker.exists():
            raise ValueError("Use a fresh work directory for a new derivation")
        write_json(marker, {"mode": "official upload" if args.upload else "local validation"})
        run(work, task, plan, args.upload)


if __name__ == "__main__":
    main()
