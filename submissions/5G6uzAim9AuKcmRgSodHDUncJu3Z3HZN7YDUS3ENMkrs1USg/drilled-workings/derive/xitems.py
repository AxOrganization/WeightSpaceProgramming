"""Requests written by derive/reqgen.py are not in the toolkit's item index. Importing this module adds them: the file
named by the environment variable REQ_ITEMS (one JSON item per line, the toolkit's own item format) is merged into
`wsp.data.by_id` and `wsp.data.load("practice:all")`. The published scripts rollouts.py, branch.py and train_sparse.py
(unchanged) are started through x_rollouts.py, x_branch.py and x_train.py, which import this module first; the GPU
workers those scripts start with their own file name come back through the same entry points."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp import data  # noqa: E402


def read_items(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def register(path=None):
    path = path or os.environ.get("REQ_ITEMS")
    if not path or getattr(data, "_req_items", None) == path:
        return
    extra = read_items(path)
    by, _by_id, _load = {it["id"]: it for it in extra}, data.by_id, data.load
    data.by_id = lambda spec="practice:all", *a, **k: {**_by_id(spec, *a, **k), **(by if spec == "practice:all" else {})}
    data.load = lambda spec, *a, **k: _load(spec, *a, **k) + (extra if spec == "practice:all" else [])
    data._req_items = path


register()
