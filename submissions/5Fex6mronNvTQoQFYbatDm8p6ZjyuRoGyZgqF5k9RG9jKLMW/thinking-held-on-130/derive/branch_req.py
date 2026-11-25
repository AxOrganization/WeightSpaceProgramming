"""The published branching script (derive/branch.py, unchanged) on recombined requests: they are not in the toolkit's
item index, so this adds them (REQ_ITEMS: the request file) and hands over; the script's GPU workers start through
this file as well.

  REQ_ITEMS=REQ.json python derive/branch_req.py plan|run|pairs <arguments of branch.py ...>
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp import data  # noqa: E402

extra = {it["id"]: it for it in json.load(open(os.environ["REQ_ITEMS"]))}
_by_id = data.by_id
data.by_id = lambda *a, **k: {**_by_id(*a, **k), **extra}
import branch  # noqa: E402

if __name__ == "__main__":
    branch.__file__ = os.path.abspath(__file__)
    branch.main()
