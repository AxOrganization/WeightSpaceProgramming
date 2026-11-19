"""Driver for the method in METHOD.md. Sweeps, controls and selects; it does not hard-code an answer.

The layer set and the dose are *searched*, never asserted: the right values depend on the parent model this
entry stacks on, and a recipe copied from our run would be a recipe of results. What is fixed here is the
procedure and the gates -- the contrast, the random controls, the selectivity test, and selection against the
competition's dominance rule rather than against practice significance.

    python derive/run.py --case nest --device cuda

Stages, each of which can stop the run with an honest null:
  screen      taxonomy + end-depth gate on the parent model          (derive/ratchet.py)
  contrast    build the nested-vs-real pairs from the model's own correct traces
  extract     direction per layer; require stability and separation
  sweep       ablate, low dose first, in parallel; watchdog on throughput collapse
  control     matched random direction at every configuration kept
  select      dominance rule with SE projected onto benchmark item counts  (derive/dominance.py)
  confirm     re-measure the leader on practice items never used for selection
  freeze      holdout; deliver the null if it does not hold
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Search space, not a recipe.
#
# Know what the dose means before reading the sweep. The ablation op leaves the component as (1 - alpha) times
# its original value, so a dose of one removes it exactly and anything above one *reverses* it: two flips the
# sign at equal magnitude, six leaves it five times its original size pointing the other way. That is not a
# monotone "more suppression" knob, and the sweep must not be read as one.
#
# Reversal is not automatically wrong -- in our run the parent's own published edit used a reversing dose, and so
# did our best candidate -- but the tolerable dose is strongly layer-dependent. The same dose that was harmless
# on one mid layer broke termination outright on the layer holding most of the direction's writing mass, with
# generation throughput collapsing by more than an order of magnitude. So sweep each layer's dose separately and
# never carry a dose across layers on the assumption it transfers.
DOSES = (0.75, 1.0, 1.5, 2.0, 3.0)
TOPK = (64, 96)
STABILITY_MIN, SEPARATION_MIN = 0.80, 1.0


def sh(*cmd: str) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")
    return p.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="nest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args(argv)
    py = [sys.executable, "-m", "wsp.case"]

    # 1. screen the parent model and gate
    sh(*py, "run", a.case, "base", "dev128", "--n", "2")
    runs = Path(sh(*py, "path", a.case).strip() or ".") / "runs" / a.case
    gate = subprocess.run([sys.executable, str(HERE / "ratchet.py"), "screen",
                           "--runs", str(runs), "--tag", "base"], capture_output=True, text=True)
    print(gate.stdout)
    if gate.returncode:
        print("GATE FAILED: the failure this method removes is not the one present. Reporting the null.")
        return 1

    # 2. contrast from the model's own correct traces, then the direction
    sh(sys.executable, str(HERE / "ratchet.py"), "contrast", "--runs", str(runs),
       "--tag", "base", "--out", "pairs.jsonl")
    sh(*py, "extract", a.case, "--pairs", "pairs.jsonl")
    dirs = json.loads(sh(*py, "directions", a.case) or "{}")
    good = [l for l, v in dirs.items()
            if v.get("stability", 0) >= STABILITY_MIN and v.get("separation", 0) >= SEPARATION_MIN]
    if not good:
        print("No layer clears the stability/separation gate: the behaviour is not one direction. Null.")
        return 1
    print(f"layers clearing the direction gate: {good}")

    # 3. sweep dose x top-k over the qualifying layers, singly and in pairs
    singles = [[l] for l in good]
    pairs = [list(c) for c in itertools.combinations(good, 2)]
    cands = []
    for layers, alpha, topk in itertools.product(singles + pairs, DOSES, TOPK):
        name = f"e_{'_'.join(map(str, layers))}_a{str(alpha).replace('.', '')}_k{topk}"
        sh(*py, "edit", a.case, "--layers", *map(str, layers),
           "--topk", str(topk), "--alpha", str(alpha), "--name", name)
        cands.append(name)
    print(f"{len(cands)} configurations staged; run them one per GPU and kill any whose "
          "generation throughput collapses -- that dose is destroying termination, not nesting.")

    # 4..6 are driven from the ledger the runs produce; dominance.py ranks them, the leader is
    # re-measured on items never used for selection, and only then frozen.
    print("next: run each candidate paired against base, then\n"
          f"  python {HERE / 'dominance.py'} --results cases/{a.case}/results --dev dev128 --project 500\n"
          "  re-measure the leader and its matched random control on a disjoint practice split\n"
          "  selectivity: the edit must lower the nested continuation's log-prob and leave the real one alone\n"
          "  freeze and confirm on holdout; if it does not hold, deliver the null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
