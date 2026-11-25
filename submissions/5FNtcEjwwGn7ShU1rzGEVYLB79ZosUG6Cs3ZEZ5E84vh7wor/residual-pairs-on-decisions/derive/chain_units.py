#!/usr/bin/env python3
"""chain_units.py: the MLP neurons that the start's whole chain writes as units, as a spec for the trainer's
--exclude-neurons-from. (This entry's file.)

  python derive/chain_units.py --edit cases/<case>/<start sid>/edit/edit.json --out X.json [--max 15000]

A neuron is (layer, index): row `index` of that layer's mlp.gate_proj and mlp.up_proj, column `index` of its
mlp.down_proj. The start is a chain of published edits (wsp.edits.load_chain: the root first, the start's own edit
last). Every link of the chain is read, and a neuron counts when an op of that link writes it as a unit:

  set_rows on gate_proj / up_proj (its rows), set_cols on down_proj (its cols), and dir_ablate restricted to selected
  rows of gate_proj / up_proj or to selected columns of down_proj (the int8 form of a neuron layer is written so).

Not counted: scale_rows / scale_cols (a gain multiplies whatever the row holds: a layer fitted on the start's weights
composes with it, and the gain composes with the layer), ops without a selection (whole matrices), round, ops on any
other matrix. A new layer that wrote over a counted neuron would replace a published layer's unit; on the others it
only adds its own change to the start's weights.

Only indices are read; no value of any edit is copied anywhere. The output holds one set_rows op per layer on
gate_proj with the indices and no values: the form derive/train_sparse.py reads for --exclude-neurons-from. Which ops
count is the rule of derive/exclude_neurons.py of 0009575744-0005, applied to every link of the chain and without
the gain ops.

--max: when more neurons than this are counted over the whole chain (published neuron layers have about 650 neurons
each, so a chain of a few such layers has a few thousand; many more would mean that an op kind is counted that should
not be), the walk falls back to the units of the start's own link alone (the last link: its set_rows / set_cols and
restricted dir_ablate ops, counted by the same rule), prints `[units] FALLBACK ...` and goes on. The rule is fixed
here; the run does not stop on it.
"""
import argparse
import collections
import json
import re
from pathlib import Path

MLP = re.compile(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
UNIT_OPS = ("set_rows", "set_cols", "dir_ablate")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edit", required=True, help="the start's edit.json (its parents are found from it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=15000, help="stop when more neurons than this are counted")
    a = ap.parse_args()
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415  (the toolkit's own chain walk and array reader)

    by, own = collections.defaultdict(set), collections.defaultdict(set)
    chain = load_chain(a.edit)
    base = chain[-1][0].get("base")
    for i, (spec, d) in enumerate(chain):
        kinds, before = collections.Counter(op["op"] for op in spec["ops"]), sum(len(v) for v in by.values())
        skipped = collections.Counter()
        for op in spec["ops"]:
            m = MLP.search(op.get("tensor", ""))
            if not m:
                continue
            key = "cols" if m.group(2) == "down_proj" else "rows"
            if op["op"] not in UNIT_OPS or op.get(key) is None:
                skipped[op["op"]] += 1
                continue
            idx = [int(x) for x in _resolve(op[key], d).ravel()]
            by[int(m.group(1))].update(idx)
            if i == len(chain) - 1:
                own[int(m.group(1))].update(idx)
        name = Path(d).parent.name if Path(d).name == "edit" else Path(d).name
        print(f"[units] link {i} {name}: {len(spec['ops'])} ops {dict(sorted(kinds.items()))}; MLP ops not counted "
              f"{dict(sorted(skipped.items()))}; neurons +{sum(len(v) for v in by.values()) - before}", flush=True)
    total = sum(len(v) for v in by.values())
    if total > a.max:
        n_own = sum(len(v) for v in own.values())
        print(f"[units] FALLBACK: {total} neurons counted over {len(chain)} links, more than --max {a.max}: the units "
              f"of the start's own link alone are kept ({n_own} neurons in {len(own)} layers)", flush=True)
        by, total = own, n_own
    ops = [{"op": "set_rows", "tensor": f"model.language_model.layers.{L}.mlp.gate_proj.weight", "rows": sorted(v)}
           for L, v in sorted(by.items())]
    per = ", ".join(f"L{L}:{len(v)}" for L, v in sorted(by.items()))
    Path(a.out).write_text(json.dumps({"name": Path(a.out).stem, "base": base, "ops": ops}))
    print(f"[units] {total} unit neurons in {len(by)} layers over {len(chain)} links ({per}) -> {a.out}")


if __name__ == "__main__":
    main()
