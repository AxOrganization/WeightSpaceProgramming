"""Drop every request of the stage's whose text has a benchmark fingerprint (the toolkit's data/bench_fingerprints.txt.gz,
wsp.data.fingerprint), as derive/reqgen.py does for its requests. Rewrites the files in place.

  python derive/inst/fp_filter.py REQ.json [REQ2.json ...]
"""
import json
import sys

from wsp import data

fps = data.bench_fingerprints()
for path in sys.argv[1:]:
    rows = json.load(open(path))
    keep = [r for r in rows if data.fingerprint(r["problem"]) not in fps]
    json.dump(keep, open(path, "w"))
    print(f"{path}: {len(rows)} requests, {len(rows) - len(keep)} dropped for a benchmark fingerprint")
