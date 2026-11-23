"""Extra practice-derived items (R2's synthetic instruction items) visible to wsp.data.by_id: set WSP_EXTRA_ITEMS to a
jsonl file. Import before using data.by_id."""
import json
import os

from wsp import data

_orig = data.by_id


def by_id(spec="practice:all"):
    d = dict(_orig(spec))
    p = os.environ.get("WSP_EXTRA_ITEMS")
    if p and os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            d[r["id"]] = r
    return d


data.by_id = by_id
