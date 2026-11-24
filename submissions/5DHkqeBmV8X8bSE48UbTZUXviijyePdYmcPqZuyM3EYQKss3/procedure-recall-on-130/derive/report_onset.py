"""report_onset.py NAME: REPORT.md of the delivered edit (stdout), from this run's own logs (practice data only)."""
import json, os, sys
from pathlib import Path
name = sys.argv[1]
W = Path(os.environ["WSP_WORK"]) / "onset"
log = (W / "status.log").read_text() if (W / "status.log").exists() else ""
pairs = ""
f = W / "fit_201.log"
if f.exists():
    pairs = next((l for l in f.read_text().splitlines() if "[pref]" in l and "pairs" in l), "")
onset = next((l for l in log.splitlines() if l.startswith("[onset]")), "")
print(f"# Violation-onset credit: delivered {name}\n")
print("A layer of per-unit and per-head gains (layers 4-21) fitted on the start's own answers to practice instruction")
print("requests: at the point where an answer first breaks its request (found with the request's own checker on growing")
print("prefixes of the visible answer), the start continues eight times; continuations that keep the request are preferred")
print("to those that break it again and to the original answer, with the loss on the continuation only. See METHOD.md.\n")
print("## This run (practice data)\n")
for l in (onset, pairs):
    if l:
        print(f"- `{l.strip()}`")
print("\n## Steps\n")
print("```")
print("\n".join(l for l in log.splitlines() if l[:2].isdigit()))
print("```")
