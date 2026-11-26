"""pools.py: the practice rows this method uses, pinned.

The toolkit's practice files can grow (rows appended to a set, new sets added). This method was developed and timed on
the first 1,024 rows of practice:ifsynth and of practice:tuluif; every list it builds by enumerating a practice set
(the splits of sets.py, the case splits, the request list of step 5) uses those rows only, in file order. Lookups by id
in the other steps (their own processes) are not affected. Sets it does not list here are used as the toolkit ships them; sets it never names are not used.

  python derive/pools.py -m MODULE ARGS...     run a module with the pools pinned   (e.g. -m wsp.case init ...)
  python derive/pools.py SCRIPT.py ARGS...     run a script with the pools pinned   (e.g. derive/sets.py e1)
  import pools; pools.pin()                    pin them in this process
"""
import runpy
import sys

POOL_ROWS = {"ifsynth": 1024, "tuluif": 1024}   # the first rows, in file order, of these practice sets


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
    """the pin holds: practice:ifsynth and practice:tuluif enumerate exactly POOL_ROWS rows (else stop the run)"""
    pin()
    import wsp.data as data  # noqa: PLC0415
    got = {n: len(data.load(f"practice:{n}")) for n in POOL_ROWS}
    if got != POOL_ROWS:
        raise SystemExit(f"pools.py: the practice pools are not pinned as this method needs ({got}, want {POOL_ROWS}): "
                         "the toolkit's practice loading changed; stop and fix derive/pools.py")


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
