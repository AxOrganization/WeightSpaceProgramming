"""derive/hold_sparse.py (the published trainer of 0009550031-0005 with hold rows) with the recombined requests
(REQ_ITEMS: the request file of derive/mkreq.py) added to the toolkit's item index first, as derive/train_req.py of
0009565562-0012 does for its fits.

  REQ_ITEMS=REQ.json torchrun --standalone --nproc_per_node 4 derive/train_hold_req.py <arguments of hold_sparse.py ...>
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp import data  # noqa: E402

extra = {it["id"]: it for it in json.load(open(os.environ["REQ_ITEMS"]))}
_by_id = data.by_id
data.by_id = lambda *a, **k: {**_by_id(*a, **k), **extra}
import hold_sparse  # noqa: E402

if __name__ == "__main__":
    hold_sparse.main()
