"""REPORT.md draft for a packed edit of the case, from the stage logs (every job's log in $WSP_WORK/k4q/logs/, the
packing and upload lines in $WSP_WORK/k9/log.md). Numbers stay as the scripts printed them; k9/REPORT_EXTRA.md (what
you chose and why, what did not work, written by hand) is added at the end when it exists.

    python derive/report9.py CASE NAME [--out REPORT.md]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acc9  # noqa: E402
import k9  # noqa: E402

WHAT = {
    "PC_05": "Part C of submission 0009552724-0008 (#79, decisive-steps) on the start's own word-problem and math "
             "traces: per-unit and per-head gains of layers 4-21 fitted on the thinking segments whose finishing value "
             "jumps, dose 0.5",
    "K_05": "answer-span leave-one-out fit on the start's own answers to practice:ifsynth and practice:tuluif requests (each answer "
            "weighted by its grade minus its request's mean grade), with keep rows on its own finished math and word "
            "problem answers, dose 0.5",
    "MC_05": "Part C of #79 again, on hard math only (longer steps, deeper thinking, larger budget), fitted on the "
             "start, dose 0.5",
    "B035": "Part B of submission 0009559255-0006 (#94, harder-requests): answer branching (the scripts of "
            "submission 0009550031-0005, #76) on requests recombined from the practice sets, four preference fits of "
            "one neuron set averaged, dose 0.35",
    "OPR_05": "on-policy round 1: the answer-span leave-one-out fit on the packed model's own answers to "
               "practice:ifsynth and practice:tuluif requests, keep rows on its own word-problem answers, dose 0.5",
    "OP2R_05": "on-policy round 2: the same on the model with round 1, dose 0.5",
}

STAGES = [
    ("K: instruction answers (on the start)", ["Ks0", "Ks1", "Ks2", "Ks3", "Ts0", "Ts1", "Ts2", "Ts3", "Kf"], ["IFS", "TUL"]),
    ("Part C (on the start)", ["PCs0", "PCs1", "PCs2", "PCs3", "PCpick", "PCf"], ["ANS"]),
    ("Branching round (on the model with Part C and K)", ["Breq", "Bbr", "BF1", "BF14", "BF21", "BF28", "BF"], []),
    ("Math Part C (on the start)", [f"MCs{i}" for i in range(8)] + ["MCpick", "MCf"], ["MT"]),
    ("On-policy round 1", [f"OP{x}{i}" for x in "AB" for i in range(4)] + ["OPRf"], ["OPA", "OPB"]),
    ("On-policy round 2", [f"OQ{x}{i}" for x in "AB" for i in range(4)] + ["OQRf"], ["OQA", "OQB"]),
]
KEEP = re.compile(r"^(\[gen\]|shard |rloo |\{'up'|\d+ traces of|\d+ training answers|step 0 heldout|"
                  r"step \d+ heldout|\d+ head gains|\d+ unit gains|[A-Z0-9_]+: \d+ directions|  ALL|  rep|  word|  tulu|"
                  r"\[dose\]|fewer than|NO_|skip|pairs|.*neurons|.*payload|.*val loss|.*mixed)")


def stage_lines(job):
    lines = [ln.strip() for ln in k9.job_log(job).splitlines() if KEEP.match(ln.strip()) or KEEP.match(ln)]
    held = [ln for ln in lines if re.match(r"step \d+ heldout", ln)]
    lines = [ln for ln in lines if not re.match(r"step \d+ heldout", ln)]
    if held:   # the held-out loss of a fit: first and last reading only
        lines.append(f"held-out loss {held[0].split()[-1]} -> {held[-1].split()[-1]} ({held[-1].split()[1]} steps)")
    return lines[-14:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--out", default="REPORT.md")
    a = ap.parse_args()
    spec = json.loads((k9.edits(a.case) / f"{a.name}.json").read_text())
    parts = spec.get("parts") or [a.name]
    W = k9.work()
    t = [f"# {a.name}: Part C, instruction answers and branching rounds, stacked on {k9.start_sid(a.case)}", "",
         f"Start: the published edit {k9.start_sid(a.case)} (this entry's parent; it is the decided decisive-steps edit "
         "0009552724-0008 with the dead-end unit of 0009549068-0003 on top). Method: METHOD.md of this submission. "
         "Every stage was derived in this run from the toolkit's practice sets and the models' own answers to them.", "",
         "## What this edit holds (in the order the ops apply)"]
    for p in parts:
        t.append(f"- {p}: {WHAT.get(p, 'see METHOD.md')}")
    drop = spec.get("dropped_branch_layers") or []
    if drop:
        t.append(f"- branching layers left out to fit the payload limit (fewest rewritten units first): {drop}")
    t += ["", "All the gain fits are joined into one set of factors per tensor (between the branching round's "
          "outright writes), as derive/pack9.py prints below; the spec's parent is the start.", "",
          "## Readings of each stage (as the scripts printed them; practice data only)"]
    for title, jobs, files in STAGES:
        body = [ln for j in jobs for ln in stage_lines(j)]
        shards = sorted(str(p) for f in files for p in W.glob(f"{f}.*.jsonl") if re.search(r"\.\d+\.jsonl$", p.name))
        if not body and not shards:
            continue
        t += ["", f"### {title}"]
        if shards:
            t += ["Practice answers of the model the stage starts from:"] + acc9.table(shards)
        t += [f"    {ln}" for ln in body]
    logf = W / "log.md"
    if logf.exists():
        t += ["", "## Packing, cuts and uploads", logf.read_text().strip()]
    t += ["", "## Limits",
          "The payload is wsp.edits.spec_size of the spec's own ops (pack lines above); the dry-run upload and "
          "axonry-check verified the limits on the delivered edit before every upload."]
    extra = W / "REPORT_EXTRA.md"
    if extra.exists():
        t += ["", extra.read_text().strip()]
    t += ["", "## Credits",
          "Part C: submission 0009552724-0008 (#79, decisive-steps). The branching round: submission 0009559255-0006 "
          "(#94, harder-requests), on the answer-branching scripts of submission 0009550031-0005 (#76). The start: "
          "submission 0009554615-0007, with the dead-end unit of submission 0009549068-0003 (#73)."]
    Path(a.out).write_text("\n".join(t) + "\n")
    print(f"wrote {a.out} ({len(t)} lines)")


if __name__ == "__main__":
    main()
