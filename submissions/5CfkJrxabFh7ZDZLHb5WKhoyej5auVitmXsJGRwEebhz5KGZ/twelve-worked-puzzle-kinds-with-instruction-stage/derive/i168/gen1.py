"""gen1.py: the toolkit's generation and grading (wsp.gen) on ONE GPU, rows written before any engine teardown.

  python derive/gen1.py --model DIR --ids IDS.json --n N --salt SALT --out OUT.jsonl [--max-tokens T] [--k0 K]

Same function as `python -m wsp.gen` (wsp.gen._generate: the same prompts, sampling, per-request seeds from (id, k,
salt) and grading); only the process around it differs. wsp.gen runs each GPU in a worker process and writes its
--out file after every worker has returned, which is after each worker's vLLM engine has been torn down; a teardown
that never ends then loses every row of the call. Here the engine object is kept referenced, so nothing is torn down
before the rows exist; they are written to OUT.part, flushed to disk and renamed to OUT (so OUT exists only
complete). Then every process this call started (the engine process and its helpers) is ended with SIGKILL and the
call exits, as the toolkit ends its workers' process groups with _kill_groups after they return. Run one call per GPU
(CUDA_VISIBLE_DEVICES), as run_e1.py's slots() does (it also ends each call's process group and waits for the GPU's
memory to be released before the next call on that GPU).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k0", type=int, default=0)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--profile", default=os.environ.get("WSP_PROFILE", "qwen35-0.8b-think"))
    a = ap.parse_args()

    import vllm  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import read_json, write_jsonl  # noqa: PLC0415
    from wsp.gen import _generate  # noqa: PLC0415

    keep = []

    class _KeptLLM(vllm.LLM):   # held in `keep`: its teardown cannot run before the rows are written
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            keep.append(self)
    vllm.LLM = _KeptLLM

    items = [data.by_id()[i] for i in read_json(a.ids)]
    rows = _generate(a.model, items, profiles.get(a.profile), n=a.n, k0=a.k0, salt=a.salt, max_tokens=a.max_tokens)
    write_jsonl(a.out + ".part", rows)
    with open(a.out + ".part", "rb") as f:
        os.fsync(f.fileno())
    os.replace(a.out + ".part", a.out)
    print(f"[gen1] {len(rows)} rows -> {a.out}", flush=True)
    sys.stderr.flush()
    _kill_descendants()
    os._exit(0)


def _kill_descendants():
    """SIGKILL every live descendant of this process (read from /proc), deepest first"""
    import signal  # noqa: PLC0415
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [os.getpid()]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    for pid in reversed(out):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


if __name__ == "__main__":
    main()
