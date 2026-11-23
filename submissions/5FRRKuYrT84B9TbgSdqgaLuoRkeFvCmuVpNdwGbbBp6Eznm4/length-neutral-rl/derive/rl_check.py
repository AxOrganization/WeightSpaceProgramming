"""Check that the exported spec builds (wsp's own builder) to the model the RL lane trained (fast build):
relative difference of every gained tensor (quantisation of unit / kv gains to 32 levels per tensor is the only expected
difference).
    python derive/rl_check.py FAST_DIR OFFICIAL_DIR"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from wsp.edits import tensor_index

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ia, ib = tensor_index(a), tensor_index(b)
ta = {sh: load_file(str(a / sh)) for sh in set(ia.values())}
tb = {sh: load_file(str(b / sh)) for sh in set(ib.values())}
worst, n, diff = 0.0, 0, 0
for name, sh in ia.items():
    x, y = ta[sh][name].float(), tb[ib[name]][name].float()
    if not torch.equal(x, y):
        diff += 1
        rel = float((x - y).norm() / max(float(y.norm()), 1e-12))
        worst = max(worst, rel)
    n += 1
print(json.dumps({"tensors": n, "differing": diff, "worst_rel_diff": worst}))
