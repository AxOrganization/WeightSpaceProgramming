"""Drive the row screen, then the instruction-gain dose.

  python derive/chain.py prep --task task.json
  python derive/chain.py screen
  python derive/chain.py dose --model MERGE

prep is CPU. screen and dose need the GPUs. Practice sets only. Run from the task directory.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CASE = "sc"
HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prep(task):
    info = json.loads(subprocess.check_output([PY, str(HERE / "rows.py"), "plan", "--task", task], text=True))
    print(json.dumps(info))
    Path("mode.json").write_text(json.dumps(info))
    if not info.get("start_edit"):
        raise SystemExit(info.get("why", "no start"))
    run([PY, "-m", "wsp.case", "init", CASE, "--profile", "qwen35-0.8b-think", "--parent", info["start_edit"],
         "--practice", "gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth",
         "--dev", "64", "--holdout", "256", "--seed", "11"])
    splits = [
        ("math96", "96", ["math_train"], "21"),
        ("pz96", "96", ["bbhsynth"], "22"),
        ("mc96", "96", ["mmlupro"], "23"),
        ("gsm96", "96", ["gsm8k_train"], "24"),
        ("req96", "96", ["ifsynth", "tuluif"], "25"),
        ("math256", "256", ["math_train"], "31"),
        ("mathguard", "128", ["math_train"], "34"),
        ("reqfit", "256", ["ifsynth"], "32"),
        ("reqhold", "256", ["tuluif"], "33"),
    ]
    for name, n, sources, seed in splits:
        run([PY, "-m", "wsp.case", "split", CASE, name, "--n", n, "--practice", *sources, "--seed", seed])
    if info["mode"] != "screen":
        print("[chain] the start already holds the parent layer. next: python derive/chain.py dose --model base")
        return
    for name, piece in (("FRESH", "fresh"), ("OVER", "over"),
                        ("OVER_0_10", "over:0-10"), ("OVER_11_16", "over:11-16"), ("OVER_17_40", "over:17-40")):
        run([PY, str(HERE / "rows.py"), "build", "--task", task, "--case", CASE, "--name", name, "--piece", piece, "--k", "0.5"])
    print("[chain] prep done. next: python derive/chain.py screen")


def _runs(models, splits, n):
    for model in models:
        for split in splits:
            run([PY, "-m", "wsp.case", "run", CASE, model, split, "--n", str(n)])


def screen():
    small = ["math96", "pz96", "mc96", "gsm96", "req96"]
    _runs(["base", "FRESH", "OVER"], small, 2)
    run([PY, str(HERE / "pick.py"), "--case", CASE, "--pieces", "FRESH", "OVER",
         "--math", "math96", "--sets", "pz96", "mc96", "gsm96", "req96", "--out", "kept.json"])
    kept = json.loads(Path("kept.json").read_text())["kept"]
    if "OVER" not in kept:
        bands = ["OVER_0_10", "OVER_11_16", "OVER_17_40"]
        _runs(bands, small, 2)
        run([PY, str(HERE / "pick.py"), "--case", CASE, "--pieces", *bands,
             "--math", "math96", "--sets", "pz96", "mc96", "gsm96", "req96", "--out", "kept_bands.json"])
        kept = [n for n in kept if n != "OVER"] + json.loads(Path("kept_bands.json").read_text())["kept"]
    Path("kept.json").write_text(json.dumps({"kept": kept}))
    if not kept:
        print("[chain] no piece passed the screen. Upload a null own edit.")
        return
    run([PY, str(HERE / "stack.py"), "join", "--case", CASE, "--pieces", *kept, "--name", "MERGE"])
    _runs(["base", "MERGE"], ["math256", "pz96", "mc96", "gsm96", "req96"], 2)
    run([PY, str(HERE / "pick.py"), "--case", CASE, "--pieces", "MERGE",
         "--math", "math256", "--sets", "pz96", "mc96", "gsm96", "req96", "--math-min", "0", "--floor", "-1",
         "--out", "merge_ok.json"])
    ok = json.loads(Path("merge_ok.json").read_text())["kept"]
    if not ok:
        print("[chain] MERGE failed the confirm. Upload a null own edit.")
        return
    print("[chain] MERGE passed. Upload it, then: python derive/chain.py dose --model MERGE")


def dose(model):
    run([PY, "-m", "wsp.case", "run", CASE, model, "reqfit", "--n", "4"])
    run([PY, str(HERE / "mix.py"), "--case", CASE, "--run", f"{model}__reqfit", "--min-constraints", "2"])
    run([PY, str(HERE / "reqgain.py"), "--case", CASE, "--model", model])
    if (C_dose_skip()):
        print("[chain] dose skipped. The upload stays", model)
        return
    for label, value in (("D05", "0.5"), ("D10", "1.0")):
        run([PY, "-m", "wsp.fit", "spec", CASE, label, "--fits", "g1", "g2", "--dose", value])
        if model != "base":
            run([PY, str(HERE / "stack.py"), "point", "--case", CASE, "--name", label, "--parent", model])
    for edit in (model, "D05", "D10"):
        run([PY, "-m", "wsp.case", "run", CASE, edit, "reqhold", "--n", "2"])
        run([PY, "-m", "wsp.case", "run", CASE, edit, "mathguard", "--n", "2"])
    for edit in ("D05", "D10"):
        for split in ("reqhold", "mathguard"):
            run([PY, "-m", "wsp.case", "compare", CASE, model, edit, split])
    print("[chain] upload D05 or D10 only when reqhold gain versus", model, "is >= +1.0 "
          "and mathguard gain versus", model, "is >= -0.2. Otherwise the earlier upload stands. "
          "Prefer the dose with the higher reqhold gain when both pass.")


def C_dose_skip():
    from wsp import case as C
    return (C.cdir(CASE) / "dose_skip").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--task", default="task.json")
    sub.add_parser("screen")
    d = sub.add_parser("dose")
    d.add_argument("--model", default="MERGE")
    args = ap.parse_args()
    {"prep": lambda: prep(args.task), "screen": screen, "dose": lambda: dose(args.model)}[args.cmd]()


if __name__ == "__main__":
    main()
