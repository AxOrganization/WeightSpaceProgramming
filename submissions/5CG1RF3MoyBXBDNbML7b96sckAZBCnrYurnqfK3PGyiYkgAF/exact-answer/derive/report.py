"""Write REPORT.md from the run's own records (no hand-copied numbers).

  python derive/report.py --case S --chosen NAME --diag /workspace/work/fit/train.jsonl.diag.json \
      --results /workspace/work/res/*.json [--fits /workspace/work/fit/D1 ...] [--notes /workspace/work/notes.md] \
      --out /workspace/work/REPORT.md

--results: files written by `derive/exact.py paired ... --out` (one per set). --notes: the agent's own paragraphs
(start chosen and why, what did not work); included verbatim.
"""
import argparse
import glob
import json
import os
from pathlib import Path


def name_of(path):
    return Path(path).name.split("__")[0]


def set_of(path):
    return Path(path).name.split("__")[-1].replace(".jsonl", "")


def fmt(s):
    return "-" if not s else f"{s['gain']:+.1f} ± {s['se']:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--chosen", required=True)
    ap.add_argument("--diag")
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--fits", nargs="*", default=[])
    ap.add_argument("--notes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = [f"# Exact answers under the real sampler (case {a.case}) - uploaded edit: {a.chosen}", ""]
    L += ["Method: submission/METHOD.md. A layer of learned dir_ablate directions (one alpha per matrix and direction)",
          "fitted to the start model's own practice answers: exact-form positives, answers repaired to the reference",
          "form, loop tails; every log-probability under the sampler's presence penalty. Practice data only.", ""]
    if a.notes and os.path.exists(a.notes):
        L += [Path(a.notes).read_text().strip(), ""]
    if a.diag and os.path.exists(a.diag):
        d = json.load(open(a.diag))
        L += ["## Diagnosis on the fit problems (start model, 4 samples, sampling cap)", "",
              "| source | problems | lenient acc | exact acc | form misses | other equivalent forms | repaired kinds | "
              "cut | cut in a loop | kept |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for src, t in d["by_source"].items():
            L.append(f"| {src} | {t['problems']} | {t['lenient_acc']} | {t['exact_acc']} | {t['form_miss']}% | "
                     f"{t.get('equivalent', 0)}% | "
                     f"{', '.join(f'{k} {v}' for k, v in t['form_kinds'].items()) or '-'} | {t['cut']}% | "
                     f"{t['cut_loop']}% | {', '.join(f'{k} {v}' for k, v in t['kept'].items())} |")
        L += ["", f"Training examples: {d['examples']}", ""]
    probe = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "results" / "presence_probe.json"
    if probe.exists():
        p = json.load(open(probe))
        L += ["## Presence probe (repaired answers, first departing token)", "",
              f"- examples: {p['fix_examples']}; reference token preferred without the penalty: "
              f"{p['reference_preferred_plain']}, with it: {p['reference_preferred_penalised']}; "
              f"flipped by the penalty: {p['flipped_by_penalty']}",
              f"- mean log-prob reference / original (plain, penalised): {p['mean_logp_reference']} / "
              f"{p['mean_logp_original']}", ""]
    for f in a.fits:
        st = Path(f) / "log.jsonl"
        if st.exists():
            recs = [json.loads(x) for x in st.read_text().splitlines() if x.strip()]
            vals = [r for r in recs if "val_nll" in r]
            last = recs[-1] if recs else {}
            L.append(f"- fit {Path(f).name}: {last.get('step')} steps, {last.get('min')} min; val NLL "
                     f"{vals[0]['val_nll'] if vals else '-'} -> {vals[-1]['val_nll'] if vals else '-'}; "
                     f"alpha rms {vals[-1].get('alpha_rms') if vals else '-'}")
    if a.fits:
        L.append("")
    paths = sorted({p for pat in a.results for p in glob.glob(pat)})
    if paths:
        L += ["## Paired results against the start (same problems and seeds; SE clustered by problem)", "",
              "| edit | set | exact gain | lenient gain | cut start -> edit | exact gain by source |",
              "|---|---|---|---|---|---|"]
        for p in paths:
            r = json.load(open(p))
            for ep, v in r["edits"].items():
                ex, le = v["exact"]["all"], v["lenient"]["all"]
                by = "; ".join(f"{s} {x['gain']:+.1f}" for s, x in v["exact"]["by_source"].items() if x)
                L.append(f"| {name_of(ep)} | {set_of(ep)} | {fmt(ex)} | {fmt(le)} | "
                         f"{ex['cut'][0] if ex else '-'}% -> {ex['cut'][1] if ex else '-'}% | {by} |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n")
    print(a.out, len(L), "lines")


if __name__ == "__main__":
    main()
