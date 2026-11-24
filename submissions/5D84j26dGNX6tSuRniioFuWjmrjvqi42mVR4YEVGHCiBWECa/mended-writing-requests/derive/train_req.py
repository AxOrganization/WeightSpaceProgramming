"""The published preference fit (derive/train_sparse.py, unchanged) on pairs of recombined requests (REQ_ITEMS: the
request file is added to the toolkit's item index first).

  REQ_ITEMS=REQ.json torchrun --standalone --nproc_per_node 2 derive/train_req.py <arguments of train_sparse.py ...>
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp import data  # noqa: E402

extra = {it["id"]: it for it in json.load(open(os.environ["REQ_ITEMS"]))}
_by_id = data.by_id
data.by_id = lambda *a, **k: {**_by_id(*a, **k), **extra}
import train_sparse  # noqa: E402

if __name__ == "__main__":
    train_sparse.main()
