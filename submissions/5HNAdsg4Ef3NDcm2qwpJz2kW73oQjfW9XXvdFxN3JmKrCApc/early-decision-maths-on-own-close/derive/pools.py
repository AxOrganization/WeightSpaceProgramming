"""pools.py: the practice rows this method uses, pinned.

The toolkit's practice files can grow (rows appended to a set, new sets added). Every list this method builds by
enumerating a practice set uses the pinned rows only, in file order: the first 1,024 rows of practice:ifsynth and of
practice:tuluif (the case splits of sets.py and the request check problems), the 7,274 rows of practice:math_train
(the case splits and the maths answer bank of step bank) and the 2,432 rows of practice:bbhsynth (the puzzle check
problems), as the toolkit held them when this entry was frozen. Lookups by id in the other steps (their own
processes) are not affected. Sets it does not list here are used as the toolkit ships them; sets it never names are
not used.

  python derive/pools.py -m MODULE ARGS...     run a module with the pools pinned   (e.g. -m wsp.case init ...)
  python derive/pools.py SCRIPT.py ARGS...     run a script with the pools pinned   (e.g. derive/sets.py st)
  import pools; pools.pin()                    pin them in this process
  import pools; pools.check()                  the pin holds (stops if not); -> the content line for the log

check() also compares the ids of the pinned rows with the digests below (sha256 of the ids joined by newlines, in
file order), written when this entry was frozen from the toolkit as it was then. A different digest is reported, not
a stop: every split, bank and check list is built in the run from the same pinned rows, so they stay disjoint
whatever the rows hold.
"""
import hashlib
import runpy
import sys

POOL_ROWS = {"ifsynth": 1024, "tuluif": 1024, "math_train": 7274, "bbhsynth": 2432}   # the first rows, in file order
POOL_SHA = {"ifsynth": "c4cb18f42a162072876acee326b4d12af4e7bcc059e61f8166ede4802671bb45",
            "tuluif": "d5c6c150f10df1c228d7d4db71ac39d1b4d77f3ef62858d8c880b94f5c4dc5a5",
            "math_train": "ae387b2d29784d2b57e71b78c2e23996bfde90ea2762af2e84434d6195b521db",
            "bbhsynth": "12d4de0ba747295e94e26d7b6a31105e0a54a1112e5b986b695953231c7cf2d9"}


def pin():
    import wsp.data as data  # noqa: PLC0415
    if getattr(data, "_pools_pinned", False):
        return
    orig = data._practice

    def _practice(name):
        rows = orig(name)
        return rows[: POOL_ROWS[name]] if name in POOL_ROWS else rows
    data._practice, data._pools_pinned = _practice, True


def check():
    """the pin holds: every set of POOL_ROWS enumerates exactly POOL_ROWS rows (else stop: SystemExit).
    -> one line saying whether the pinned rows' ids still match POOL_SHA (information only)"""
    pin()
    import wsp.data as data  # noqa: PLC0415
    rows = {n: data.load(f"practice:{n}") for n in POOL_ROWS}
    got = {n: len(r) for n, r in rows.items()}
    if got != POOL_ROWS:
        raise SystemExit(f"pools.py: the practice pools are not pinned as this method needs ({got}, want {POOL_ROWS}): "
                         "the toolkit's practice loading changed; the maths layer does not run (upload 1 is not "
                         "affected)")
    sha = {n: hashlib.sha256("\n".join(r["id"] for r in rows[n]).encode()).hexdigest() for n in POOL_ROWS}
    diff = [n for n in POOL_ROWS if sha[n] != POOL_SHA[n]]
    return ("pools: pinned " + ", ".join(f"{n} {POOL_ROWS[n]}" for n in POOL_ROWS) + "; content "
            + ("as at freeze" if not diff else "differs from the pinned digest on " + ", ".join(diff)
               + " (information: every list is built in this run from the same pinned rows)"))


def main():
    check()
    if sys.argv[1] == "-m":
        mod, sys.argv = sys.argv[2], [sys.argv[2]] + sys.argv[3:]
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    else:
        path, sys.argv = sys.argv[1], sys.argv[1:]
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
