"""Step 0 - one minute of checks before any GPU work: the toolkit, the case and its inherited start, the practice
sources, the GPUs, the units the start's edit chain already owns, and that an edit with inline arrays survives the
upload packaging unchanged. Builds the start model once (later jobs reuse it).

  python derive/smoke.py <case>
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    a = ap.parse_args()
    out, bad = [], []
    import wsp  # noqa: PLC0415
    from wsp import data  # noqa: PLC0415
    from wsp.edits import export, load_chain  # noqa: PLC0415
    out.append(f"toolkit: {list(wsp.__path__)[0]}; WSP_CASES={os.environ.get('WSP_CASES')} "
               f"WSP_WORK={os.environ.get('WSP_WORK')}")
    c = D.conf(a.case)
    if not c.get("parent"):
        bad.append("the case has no parent: init it with --parent parent/edit/edit.json (the start is the entry's "
                   "parent)")
    else:
        chain = load_chain(D.case_dir(a.case) / c["parent"])
        out.append(f"start: {len(chain)} published edits in its chain, {sum(len(s['ops']) for s, _ in chain)} ops")
    for s in c.get("practice", []):
        out.append(f"practice:{s}: {len(data.load(f'practice:{s}'))} problems")
    missing = [s for s in ("gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif")
               if s not in c.get("practice", [])]
    if missing:
        bad.append(f"case splits lack the practice sources {missing}: init with --practice gsm8k_train math_train "
                   "mmlupro ifsynth tuluif")
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        out.append(f"GPUs ({len(g)}): " + " | ".join(g))
    except (OSError, subprocess.SubprocessError):
        bad.append("nvidia-smi not readable")
    t0 = time.time()
    mdir = D.start_model(a.case)
    info = D.model_info(mdir)
    out.append(f"start model {mdir} ready in {time.time() - t0:.0f}s; {info['n_layers']} layers, full attention at "
               f"{info['full_attn']}, MLP {info['inter']} units")
    written, touched = D.chain_units(a.case, info["last"])
    out.append(f"last-layer units the start's chain set: {sorted(written)[:40] or 'none'}; touched in any way: "
               f"{len(touched)} (never used as spare units)")
    tmp = D.kdir(a.case) / "smoke"
    tmp.mkdir(exist_ok=True)
    spec = {"name": "probe", "base": c["profile"], "parent": "../" + c["parent"] if c.get("parent") else None,
            "ops": D.unit_ops(info["prefix"], info["last"], 0, [0.5] * info["hidden"], [0.25] * info["hidden"],
                              [0.125] * info["hidden"])}
    if spec["parent"] is None:
        spec.pop("parent")
    probe = D.case_dir(a.case) / "edits" / "_smoke_probe.json"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text(json.dumps(spec))
    try:
        try:
            p = export(probe, tmp / "deliver", "edit", parents=False)
        except TypeError:                              # newer toolkits: export() has no `parents` argument
            p = export(probe, tmp / "deliver", "edit")
        got = json.loads(p.read_text())
        same = all(x.get("values") == y.get("values") and x.get("u") == y.get("u")
                   for x, y in zip(spec["ops"], got["ops"]))
        out.append(f"upload packaging keeps inline arrays: {'yes' if same else 'NO'}; parent reference "
                   f"{got.get('parent')}")
        if not same:
            bad.append("the packager changed inline arrays: report it and check every delivered edit.json by hand")
    except Exception as ex:  # noqa: BLE001 - the probe only mirrors the packager; the dry-run upload checks the real one
        out.append(f"note: packaging probe skipped ({type(ex).__name__}: {str(ex)[:120]}); the dry-run upload of each "
                   "edit checks the real packager")
    finally:
        probe.unlink(missing_ok=True)
    for f in ("jobs.py", "sets.py", "sample.py", "startcheck.py", "reads.py", "gates.py", "merge.py", "wouldclose.py",
              "devcheck.py", "report.py"):
        if not (D.HERE / f).exists():
            bad.append(f"derive/{f} missing: copy the submission's derive/ directory whole")
    if not Path("task.json").exists():
        out.append("note: no task.json in the working directory (needed for axonry-check and the upload)")
    D.log(a.case, "smoke", "\n".join(out) + ("\nPROBLEMS:\n" + "\n".join(bad) if bad else "\nall checks passed"))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
