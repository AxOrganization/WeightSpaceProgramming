"""Run a script with our own instruction requests in the toolkit's item index and our own checker as their grader.

  REQ_ITEMS=A.json[:B.json] python own_run.py SCRIPT.py ARGS...

The items (derive/owninst.py: task "owninst", each with its instructions and their parameters) are added to
wsp.data.by_id (the idea of 0009576302-0002's derive/branch_req.py, as in our reqrun.py), relabelled with source
"ifsynth" so that the request-gain scripts (branch2.py --kinds if, pref_gains2.py) treat them as instruction requests,
and graded by owninst.check_item through the toolkit's grader table (wsp.grade.GRADERS["owninst"]). The practice pools
are pinned (derive/pools.py of 0009576553-0024).
"""
import json
import os
import runpy
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import pools  # noqa: E402
pools.pin()
import owninst  # noqa: E402
from wsp import data  # noqa: E402
from wsp import grade  # noqa: E402


def _own(item, answer):
    per, ok = owninst.check_item(item, answer)
    return [bool(x) for x in per], bool(ok)


grade.GRADERS["owninst"] = _own
data.PROMPTS["owninst"] = data.PROMPTS["ifeval"]     # the request text as it is, like every instruction task
extra = {}
for f in filter(None, os.environ.get("REQ_ITEMS", "").split(":")):
    for it in json.load(open(f)):
        it = dict(it, kind="owninst", source="ifsynth")
        extra[it["id"]] = it
_by_id = data.by_id
data.by_id = lambda *a, **k: {**_by_id(*a, **k), **extra}
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
