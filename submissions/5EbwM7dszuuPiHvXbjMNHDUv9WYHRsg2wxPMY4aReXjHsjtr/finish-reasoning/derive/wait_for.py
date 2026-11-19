"""First line of a queued job that depends on another queued job: wait until that job has delivered.

usage: python wait_for.py FILE [FILE ...] --jobs NAME [--any] [--timeout 10800]
         exit 0 as soon as every FILE exists (run files, sample files and <name>.fit.json are written when their
         job is complete, so "exists" means "finished"). With --any: also exit 0 when no producing job is left
         and at least one FILE exists (a part of the work failed; go on with what is there).
       python wait_for.py --generating --jobs NAME [--timeout 3600]
         exit 0 as soon as a queue job whose name contains NAME shows "Processed prompts" in its log or has
         finished: the first vLLM start on a machine compiles kernels into a shared cache, and several first
         starts at once can fail.
exit 1 when no queue job whose name contains NAME is pending or running any more and the condition does not hold
(the producing job failed or was never added), or after --timeout seconds: the waiting job then fails at once
instead of holding its GPU. The waiting job's own name must not contain NAME.
"""
import argparse
import glob
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--jobs", required=True, help="part of the name of the queue job(s) to wait for")
    ap.add_argument("--generating", action="store_true")
    ap.add_argument("--any", action="store_true")
    ap.add_argument("--timeout", type=int, default=10800)
    a = ap.parse_args()
    q = os.path.join(os.environ["WSP_WORK"], "queue")
    t0 = time.time()

    def done():
        if a.generating:
            if glob.glob(os.path.join(q, "done", f"*{a.jobs}*.sh")):
                return True
            for f in glob.glob(os.path.join(q, "running", f"*{a.jobs}*.log")):
                try:
                    if "Processed prompts" in open(f, errors="ignore").read():
                        return True
                except OSError:
                    pass
            return False
        return all(os.path.exists(f) for f in a.files)

    while True:
        if done():
            return 0
        alive = [p for d in ("pending", "running") for p in glob.glob(os.path.join(q, d, f"*{a.jobs}*.sh"))]
        if not alive:
            time.sleep(5)   # a job that just ended is moved out of running/ after its files are complete
            if done():
                return 0
            have = [f for f in a.files if os.path.exists(f)]
            if a.any and have:
                print(f"going on with {len(have)} of {len(a.files)} files (no job named *{a.jobs}* is left)", flush=True)
                return 0
            print(f"no pending or running job named *{a.jobs}*, and the condition does not hold "
                  f"(missing: {[f for f in a.files if not os.path.exists(f)]})", flush=True)
            return 1
        if time.time() - t0 > a.timeout:
            print(f"timeout after {a.timeout}s", flush=True)
            return 1
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
