"""Build self-contained edit operations from vectors derived during the run."""
import math


def inline_gate_ops(prefix, layer, unit, gate, up, down):
    """Return gate/up rows and a down column as inline JSON values, without NPZ references."""
    vectors = [[float(value) for value in vector] for vector in (gate, up, down)]
    if not vectors[0] or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("gate, up and down must be nonempty vectors with the same width")
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("gate values must be finite")
    gate, up, down = vectors
    tensor = f"{prefix}layers.{layer}.mlp."
    return [
        {"op": "set_rows", "tensor": tensor + "gate_proj.weight", "rows": [unit], "values": [gate]},
        {"op": "set_rows", "tensor": tensor + "up_proj.weight", "rows": [unit], "values": [up]},
        {"op": "set_cols", "tensor": tensor + "down_proj.weight", "cols": [unit],
         "values": [[value] for value in down]},
    ]
