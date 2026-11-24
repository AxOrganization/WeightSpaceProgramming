"""Puzzles drawn in the run (more_items.py) are not in the toolkit's practice files. install() adds the rows of the
files named in LOCAL_ITEMS (':'-separated jsonl) to wsp.data.by_id, so a script that looks a problem up by id finds
them. Nothing changes when LOCAL_ITEMS is unset."""
import json
import os


def install():
    from wsp import data
    if getattr(data, "_local_items_installed", False):
        return
    extra = {}
    for p in filter(None, os.environ.get("LOCAL_ITEMS", "").split(":")):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                extra[r["id"]] = r
    if not extra:
        return
    orig = data.by_id

    def by_id(*args, **kw):
        d = dict(orig(*args, **kw))
        d.update(extra)
        return d
    data.by_id = by_id
    data._local_items_installed = True
