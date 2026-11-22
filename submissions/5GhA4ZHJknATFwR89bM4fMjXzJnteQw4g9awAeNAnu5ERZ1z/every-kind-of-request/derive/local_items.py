"""The requests this method builds from shipped practice requests (make_requests.py -> requests.jsonl) are not in
the toolkit's practice files, so the scripts that look a problem up by id add them here."""
import json
import os


def extend(items, path=None):
    path = path or os.environ.get("REQUESTS_FILE", "requests.jsonl")
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                items[r["id"]] = r
    return items
