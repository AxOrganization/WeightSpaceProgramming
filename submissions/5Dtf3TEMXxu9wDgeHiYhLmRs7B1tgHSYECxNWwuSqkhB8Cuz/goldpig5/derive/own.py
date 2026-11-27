"""own.py : this entry's own practice requests made known to the toolkit, and the toolkit's fitter run with them.

  import own; own.register([files])       the requests of the given jsonl files (derive/reqs.py) join wsp.data.by_id(),
                                          their prompt is the request text as it stands (task "ifx": "{problem}") and
                                          their grader is the request's own checkers (derive/ifx.py check): a sample
                                          is right when its visible answer follows every instruction of its request.
  python derive/own.py fit <wsp.fit arguments>     wsp.fit with the requests of $OWN_ITEMS (comma-separated files)
  python derive/own.py stats <wsp.stats arguments> wsp.stats likewise

Nothing of the toolkit's fitter, sampler or grader is changed: the requests are only made known to them."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load(files):
    items = {}
    for f in files:
        if f and os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        it = json.loads(line)
                        items[it["id"]] = it
    return items


def register(files=None):
    """-> {id: request} of the files (default: $OWN_ITEMS), now known to wsp.data / wsp.grade in this process."""
    from wsp import data, grade
    import ifx
    files = files if files is not None else [f for f in os.environ.get("OWN_ITEMS", "").split(",") if f]
    items = load(files)
    data.PROMPTS["ifx"] = "{problem}"

    def _ifx(item, answer):
        per, ok = ifx.check(item, answer)
        return per, bool(ok)

    grade.GRADERS["ifx"] = _ifx
    if not getattr(data.by_id, "_own", None):
        orig = data.by_id

        def by_id(spec="practice:all"):
            return {**orig(spec), **by_id._own}

        by_id._own = {}
        data.by_id = by_id
    data.by_id._own.update(items)
    return items


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    register()
    if cmd == "fit":
        from wsp import fit
        sys.argv = ["wsp.fit"] + rest
        fit.main()
    elif cmd == "stats":
        from wsp import stats
        sys.argv = ["wsp.stats"] + rest
        stats.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
