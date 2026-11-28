"""gen_x.py --case CASE --model base|EDIT --ids IDS.json --items REQ.jsonl --out RUN.jsonl [--n 2] [--salt s0]
Answers of a model of the case to requests of derive/reqgen.py (which are not in the toolkit's item index), sampled and
graded by the toolkit's own generator (wsp.gen.generate: the profile's sampling, seeds from (id, sample, salt), rows in
its format). Uses the GPUs in CUDA_VISIBLE_DEVICES."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--model", default="base")
    ap.add_argument("--ids", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--salt", default="s0")
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args()
    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.gen import generate  # noqa: PLC0415
    by = {}
    for line in open(a.items):
        if line.strip():
            it = json.loads(line)
            by[it["id"]] = it
    items = [by[i] for i in json.load(open(a.ids))]
    rows = generate(C.model_path(a.case, a.model), items, profiles.get(C.conf(a.case)["profile"]), n=a.n, out=a.out,
                    salt=a.salt, max_tokens=a.max_tokens)
    rows = rows or [json.loads(line) for line in open(a.out)]
    print(f"{a.model}: {len(rows)} answers, followed {100 * sum(bool(r['correct']) for r in rows) / max(1, len(rows)):.1f}% "
          f"-> {a.out}")
    print("GENX_DONE", flush=True)


if __name__ == "__main__":
    main()
