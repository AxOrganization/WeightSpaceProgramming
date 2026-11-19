"""Split a run (jsonl of one case run) by practice source: <run>.jsonl -> <out_dir>/<stem>__<source>.jsonl.
Usage: split_runs.py <run.jsonl> <out_dir>"""
import json
import sys
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
groups = {}
for line in open(run):
    r = json.loads(line)
    groups.setdefault(r["id"].split("_")[0], []).append(line)  # practice ids: math_..., gsm8k_..., mmlupro_..., ifsynth_...
for src, lines in groups.items():
    (out / f"{run.stem}__{src}.jsonl").write_text("".join(lines))
    print(src, len(lines))
