"""The requests and puzzles this method builds (make_requests.py, puzzles.py) are not in the toolkit's practice
files, so the scripts that look a problem up by id add them here: the file named by REQUESTS_FILE and every file
named in LOCAL_ITEMS (separated by ':')."""
import json
import os


def extend(items, path=None):
    paths = [path or os.environ.get("REQUESTS_FILE", "requests.jsonl")]
    paths += [p for p in os.environ.get("LOCAL_ITEMS", "").split(":") if p]
    for p in paths:
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    items[r["id"]] = r
    return items
