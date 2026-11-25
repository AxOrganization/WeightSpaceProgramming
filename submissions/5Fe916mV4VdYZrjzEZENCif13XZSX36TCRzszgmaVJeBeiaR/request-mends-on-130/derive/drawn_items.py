"""The puzzles of derive/draw_puzzles.py are not in the toolkit's item index. Importing this module adds them: the
files named by the environment variable DRAWN_ITEMS (separated by ':'; one item per line, the toolkit's own item
format) are merged into `wsp.data.by_id` and `wsp.data.load("practice:all")`. The fit is started through
fit_drawn.py and the generation through roll_drawn.py, which import this module first. The mechanism is that of
derive/xitems.py of the finished entry 0009570269-0024."""
import json
import os

from wsp import data


def register(paths=None):
    paths = paths or os.environ.get("DRAWN_ITEMS")
    if not paths or getattr(data, "_drawn_items", None) == paths:
        return
    extra = [json.loads(line) for p in paths.split(":") if p and os.path.exists(p) for line in open(p) if line.strip()]
    by, _by_id, _load = {it["id"]: it for it in extra}, data.by_id, data.load
    data.by_id = lambda spec="practice:all", *a, **k: {**_by_id(spec, *a, **k), **(by if spec == "practice:all" else {})}
    data.load = lambda spec, *a, **k: _load(spec, *a, **k) + (extra if spec == "practice:all" else [])
    data._drawn_items = paths


register()
