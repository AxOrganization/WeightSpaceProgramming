"""Step 0 - generate the start model's own practice runs (vLLM through the toolkit, the profile's sampling).

  python derive/sample.py <case> <set> <chunk>     one chunk of a fit set of sets.py
                                                    -> $WSP_WORK/runs/<case>/base__<set>c<chunk>.jsonl
derive/plan.py queues every chunk. A chunk that exists is not generated again. Each chunk appends its output rate to
$WSP_WORK/k4/<case>/rates.jsonl (output tokens per second per GPU, wall time including the vLLM start-up).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def run_chunk(case, name, chunk):
    from wsp import profiles  # noqa: PLC0415
    from wsp.common import RUNS, write_jsonl  # noqa: PLC0415
    from wsp.data import by_id  # noqa: PLC0415
    from wsp.gen import generate  # noqa: PLC0415

    out = Path(RUNS) / case / f"base__{name}c{chunk}.jsonl"
    if out.exists():
        print(f"[sample] {out.name} exists")
        return
    plan = json.loads((D.kdir(case) / "sets.json").read_text())[name]
    ch = plan["chunks"][chunk]
    idx = by_id()
    items = [idx[i] for i in ch["ids"] if i in idx]
    prof = profiles.get(D.conf(case)["profile"])
    model = D.start_model(case)
    t0 = time.time()
    rows = generate(model, items, prof, n=1, salt=f"fit-{name}-{ch['k']}", max_tokens=plan["cap"])
    dt = time.time() - t0
    for r in rows:
        r["k"] = ch["k"]
    write_jsonl(out, rows)
    ntok = sum(r["n_tokens"] for r in rows)
    rate = {"job": f"{name}c{chunk}", "answers": len(rows), "tokens": ntok, "seconds": round(dt),
            "tok_s_per_gpu": ntok / max(dt, 1)}
    with open(D.kdir(case) / "rates.jsonl", "a") as f:
        f.write(json.dumps(rate) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[sample] {out.name}: {len(rows)} answers, {ntok} tokens in {dt / 60:.1f} min "
          f"({rate['tok_s_per_gpu']:.0f} tok/s per GPU incl. start-up); kinds {kinds}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("set")
    ap.add_argument("chunk", type=int)
    a = ap.parse_args()
    run_chunk(a.case, a.set, a.chunk)


if __name__ == "__main__":
    main()
