"""stack.py --specs A.json B.json [...] --out S.json

One spec whose ops are those of several specs of the same start, in order (each op keeps the array file it names;
the files stay beside the specs). Used to build the model that carries the directions and a neuron layer together:
they change different matrices, so the order does not matter."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = [json.loads(Path(p).read_text()) for p in a.specs]
    out = Path(a.out)
    spec = dict(specs[0], name=out.stem, ops=[op for s in specs for op in s["ops"]])
    out.write_text(json.dumps(spec))
    print(f"[stack] {out}: {len(spec['ops'])} ops of {len(specs)} specs")


if __name__ == "__main__":
    main()
