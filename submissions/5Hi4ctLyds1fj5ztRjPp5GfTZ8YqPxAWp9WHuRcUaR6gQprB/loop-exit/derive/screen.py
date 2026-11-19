"""Candidate edits from a directions file, cheap screens, and one table to choose from.

  build <case> --dirs directions_branch.npz --layers 8 10 12 | --auto 4 [--topk 64] [--alpha 6]
        [--mlp/--no-mlp] [--attn] [--random] --name NAME
                      dir_ablate spec: the file's direction of each layer ablated from the MLP down_proj (top-k
                      writer columns) and/or the layer's attention output matrix (self_attn.o_proj on full-attention
                      layers, linear_attn.out_proj on Gated-DeltaNet layers; all columns). --auto K picks the K
                      layers with the largest MLP writer mass among those with split-half stability >= 0.8, never
                      the last layer. --random: a random direction orthogonal to each layer's (same scope: control).
        --band LO HI restricts --auto to layers LO..HI. --invert: instead of a direction ablation, scale the top-k
        writer columns by --factor (e.g. -1: sign inversion of those neurons' output). --heads N [--head-factor F]:
        scale the attention output columns of the N heads writing most of the direction (head_writer in the
        directions file) by F (0 silences a head; 0.5 halves it), on top of --mlp/--attn when given; with
        --head-ablate [--head-alpha A] the direction is instead removed from those heads' columns (ThinkEdit form).
  heads <case> --name NAME L:H[:F] [...]           fixed head edit: scale the output columns of head H of layer L by F
        (default 0); also L:mlp:J[:F] scales MLP neuron J's down_proj column (reported edits, for comparison)
  combine <case> --name NAME EDIT1 EDIT2 [...]      one spec with the ops of several (applied in order)
  screen <case> EDIT [...] [--no-rollout]           per edit: selectivity on branch_pairs_heldout and loop_pairs_heldout
                      (needs the base's values: screen base first), the held-out loop rate paired with the base
                      (rollout), payload and relative change -> results/<edit>__screen.json
  table <case>                                      every screened edit, with its DEV / HOLDOUT paired gains if run
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))

from wsp import profiles  # noqa: E402
from wsp.case import cdir, conf, scratch  # noqa: E402
from wsp.common import read_json, write_json  # noqa: E402
from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: E402

MIN_STABILITY = 0.8


def _attn_tensor(names, prefix, layer):
    for cand in (f"{prefix}layers.{layer}.self_attn.o_proj.weight", f"{prefix}layers.{layer}.linear_attn.out_proj.weight"):
        if cand in names:
            return cand
    raise SystemExit(f"no attention output matrix for layer {layer}")


def auto_layers(d, k, band=None):
    n = len(d["U"])
    lo, hi = band if band else (0, n - 2)
    rows = []
    for l in range(max(lo, 0), min(hi, n - 2) + 1):          # never the last layer
        w = d["writer"][l]
        if d["stability"][l] >= MIN_STABILITY:
            rows.append((float(w[w > 0].sum()), l))
    rows.sort(reverse=True)
    return sorted(l for _, l in rows[:k])


def top_heads(d, n):
    """[(layer, head, score)] of the n heads with the largest writer score (never the last layer)."""
    hw = d["head_writer"]
    rows = [(float(hw[l, h]), l, h) for l in range(len(hw) - 1) for h in range(hw.shape[1]) if hw[l, h] == hw[l, h]]
    rows.sort(reverse=True)
    return [(l, h, sc) for sc, l, h in rows[:n]]


def head_cols(meta, layer, head):
    m = next(x for x in meta if x["layer"] == layer)
    return m["tensor"], list(range(head * m["head_dim"], (head + 1) * m["head_dim"]))


def cmd_build(a):
    c = conf(a.case)
    d = np.load(scratch(a.case) / a.dirs)
    layers = a.layers if a.layers is not None else auto_layers(d, a.auto, a.band)
    if not layers and not a.heads:
        raise SystemExit("no layer passes the stability rule; look at the table and choose by hand")
    names = tensor_index(profiles.base_dir(c["profile"]))
    prefix = decoder_prefix(names)
    rng, arrays, ops = np.random.default_rng(0), {}, []
    for l in layers:
        u = d["U"][l].astype(np.float32)
        if a.random:
            r = rng.standard_normal(u.shape).astype(np.float32)
            r -= (r @ u) * u
            u = r / np.linalg.norm(r)
        arrays[f"u{l}"] = u
        if a.mlp:
            arrays[f"cols{l}"] = np.argsort(-d["writer"][l])[:a.topk].astype(np.int64)
            if a.invert:
                ops.append({"op": "scale_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight",
                            "cols": f"{a.name}.npz#cols{l}", "factor": a.factor})
            else:
                ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "side": "out",
                            "u": f"{a.name}.npz#u{l}", "alpha": a.alpha, "cols": f"{a.name}.npz#cols{l}"})
        if a.attn:
            ops.append({"op": "dir_ablate", "tensor": _attn_tensor(names, prefix, l), "side": "out",
                        "u": f"{a.name}.npz#u{l}", "alpha": a.attn_alpha if a.attn_alpha is not None else a.alpha})
    heads = []
    if a.heads:
        meta = json.loads(str(d["head_meta"]))
        for l, h, sc in top_heads(d, a.heads):
            tname, cols = head_cols(meta, l, h)
            arrays[f"head{l}_{h}"] = np.array(cols, dtype=np.int64)
            if a.head_ablate:      # ThinkEdit form: the direction removed from this head's output columns only
                arrays.setdefault(f"u{l}", d["U"][l].astype(np.float32))
                ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.{tname}", "side": "out",
                            "u": f"{a.name}.npz#u{l}", "alpha": a.head_alpha, "cols": f"{a.name}.npz#head{l}_{h}"})
            else:
                ops.append({"op": "scale_cols", "tensor": f"{prefix}layers.{l}.{tname}",
                            "cols": f"{a.name}.npz#head{l}_{h}", "factor": a.head_factor})
            heads.append([l, h, round(sc, 4)])
    out = cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops,
            "about": {"dirs": a.dirs, "layers": layers, "topk": a.topk, "alpha": a.alpha, "mlp": a.mlp, "attn": a.attn,
                      "random": a.random, "invert": a.invert and a.factor, "heads": heads, "head_factor": a.head_factor}}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    t, payload = spec_size(out / f"{a.name}.json")
    print(f"{a.name}: layers {layers}{' heads ' + str(heads) if heads else ''}, {t} tensors, payload {payload} bytes")


def cmd_heads(a):
    """Fixed head / neuron scalings: L:H[:F] (attention head H of layer L) or L:mlp:J[:F] (down_proj column J)."""
    from wsp.probe import Probe  # noqa: PLC0415
    c = conf(a.case)
    names = tensor_index(profiles.base_dir(c["profile"]))
    prefix = decoder_prefix(names)
    probe = Probe(str(profiles.base_dir(c["profile"])), "cpu")        # layout only: no forward pass
    cfg = probe.model.config.get_text_config()
    ops, arrays = [], {}
    for i, item in enumerate(a.items):
        parts = item.split(":")
        l = int(parts[0])
        if parts[1] == "mlp":
            j, f = int(parts[2]), float(parts[3]) if len(parts) > 3 else 0.0
            ops.append({"op": "scale_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "cols": [j], "factor": f})
            continue
        h, f = int(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0
        full = f"{prefix}layers.{l}.self_attn.o_proj.weight" in names
        tname = f"{prefix}layers.{l}.self_attn.o_proj.weight" if full else f"{prefix}layers.{l}.linear_attn.out_proj.weight"
        n = cfg.num_attention_heads if full else cfg.linear_num_value_heads
        mod = next(m for k, m in probe.layers[l].named_modules() if k.split(".")[-1] in ("o_proj", "out_proj"))
        hd = mod.in_features // n
        arrays[f"h{i}"] = np.arange(h * hd, (h + 1) * hd, dtype=np.int64)
        ops.append({"op": "scale_cols", "tensor": tname, "cols": f"{a.name}.npz#h{i}", "factor": f})
    out = cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops, "about": {"fixed": a.items}}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    t, payload = spec_size(out / f"{a.name}.json")
    print(f"{a.name}: {len(ops)} ops, {t} tensors, payload {payload} bytes")


def cmd_combine(a):
    out = cdir(a.case) / "edits"
    specs = [read_json(out / f"{e}.json") for e in a.edits]
    spec = {"name": a.name, "base": specs[0]["base"], "ops": [op for s in specs for op in s["ops"]],
            "about": {"combine": a.edits}}
    if specs[0].get("parent"):
        spec["parent"] = specs[0]["parent"]
    write_json(out / f"{a.name}.json", spec)
    t, payload = spec_size(out / f"{a.name}.json")
    print(f"{a.name}: {len(spec['ops'])} ops, {t} tensors, payload {payload} bytes")


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def cmd_screen(a):
    from wsp.case import model_path  # noqa: PLC0415
    from wsp.directions import selectivity  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415

    R = cdir(a.case) / "results"
    sel_sets = {"branch": "branch_pairs_heldout", "loop": "loop_pairs_heldout"}
    for edit in a.edits:
        res = {"edit": edit}
        path = model_path(a.case, edit)                      # builds the model on first use
        if edit != "base":
            man = read_json(f"{path}/wsp_edit_manifest.json")
            own = set(man.get("own_tensors", []))
            res["relative_change"] = {t["name"].split("layers.")[-1]: round(t["relative_change_vs_parent"], 4)
                                      for t in man.get("tensors", []) if t["name"] in own}
            res["payload"] = spec_size(cdir(a.case) / "edits" / f"{edit}.json")[1]
        probe = Probe(path, a.device)
        for key, name in sel_sets.items():
            pairs = read_json(scratch(a.case) / f"{name}.json")
            if not pairs:
                continue
            s = selectivity(probe, pairs)
            res[f"sel_{key}"] = {"logp_good": round(s["logp_good"], 4), "logp_bad": round(s["logp_bad"], 4), "n": len(pairs)}
        del probe
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
        if a.rollout:
            here = os.path.dirname(os.path.abspath(__file__))
            _run([sys.executable, os.path.join(here, "branch.py"), "rollout", a.case, edit, "--part", "heldout"])
            roll = read_json(R / f"{edit}__rollout_heldout.json")
            res["rollout"] = {"labels": roll["labels"], "points_with_both": roll["points_with_both"],
                              **({"paired": roll["paired_with_base"]} if "paired_with_base" in roll else {})}
        write_json(R / f"{edit}__screen.json", res)
        print(json.dumps(res), flush=True)
    cmd_table(a)


def _paired(R, edit, tag):
    p = R / f"{edit}__{tag}.json"
    if not p.exists():
        return ""
    d = read_json(p).get("paired")
    if not d:
        return ""
    return f"{100 * d['gain']:+.1f}±{100 * d['se']:.1f} cut {100 * d['base_cut']:.0f}→{100 * d['edit_cut']:.0f}"


def cmd_table(a):
    R = cdir(a.case) / "results"
    base = read_json(R / "base__screen.json") if (R / "base__screen.json").exists() else None
    rows = sorted(R.glob("*__screen.json"))
    hdr = ["edit", "Δbad(branch)", "Δgood(branch)", "Δgood(loop)", "Δbad(loop)", "loop rate held-out (Δ±se)", "DEV128", "HOLDOUT256", "payload"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for p in rows:
        r = read_json(p)
        e = r["edit"]
        def delta(key, side):
            if not base or key not in r or key not in base:
                return ""
            return f"{r[key][side] - base[key][side]:+.3f}"
        roll = r.get("rollout", {})
        pr = roll.get("paired", {}).get("loop")
        rr = (f"{pr['base']:.2f}→{pr['model']:.2f} ({100 * pr['diff']:+.0f}±{100 * pr['se']:.0f} pts)" if pr else
              (f"{roll['labels'].get('loop', 0) / max(sum(roll['labels'].values()), 1):.2f}" if roll else ""))
        print(f"| {e} | {delta('sel_branch', 'logp_bad')} | {delta('sel_branch', 'logp_good')} | "
              f"{delta('sel_loop', 'logp_good')} | {delta('sel_loop', 'logp_bad')} | {rr} | "
              f"{_paired(R, e, 'dev128')} | {_paired(R, e, 'holdout256')} | {r.get('payload', '')} |")
    if base:
        print(f"\nbase selectivity: branch good/bad {base.get('sel_branch', {}).get('logp_good')} / "
              f"{base.get('sel_branch', {}).get('logp_bad')}; loop good/bad {base.get('sel_loop', {}).get('logp_good')} / "
              f"{base.get('sel_loop', {}).get('logp_bad')}. A good edit: Δbad(branch) clearly negative, Δgood(branch) "
              f"and Δgood(loop) near 0 (a drop there predicts lost answers), loop rate down beyond 2 se, random control flat.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("case")
        for a in args:
            s.add_argument(*a[0], **a[1])
        return s

    b = p("build", (["--dirs"], {"required": True}), (["--layers"], {"type": int, "nargs": "*"}),
          (["--auto"], {"type": int, "default": 4}), (["--topk"], {"type": int, "default": 64}),
          (["--alpha"], {"type": float, "default": 6.0}), (["--attn"], {"action": "store_true"}),
          (["--attn-alpha"], {"type": float}), (["--random"], {"action": "store_true"}),
          (["--band"], {"type": int, "nargs": 2}), (["--invert"], {"action": "store_true"}),
          (["--factor"], {"type": float, "default": -1.0}), (["--heads"], {"type": int, "default": 0}),
          (["--head-factor"], {"type": float, "default": 0.0}), (["--head-ablate"], {"action": "store_true"}),
          (["--head-alpha"], {"type": float, "default": 1.0}),
          (["--name"], {"required": True}), help="write an edit spec from a directions file")
    b.add_argument("--mlp", action=argparse.BooleanOptionalAction, default=True)
    p("heads", (["--name"], {"required": True}), (["items"], {"nargs": "+"}), help="fixed head / neuron scalings")
    p("combine", (["--name"], {"required": True}), (["edits"], {"nargs": "+"}), help="one spec from several")
    s = p("screen", (["edits"], {"nargs": "+"}), (["--device"], {"default": "cuda"}), help="cheap screens per edit")
    s.add_argument("--rollout", action=argparse.BooleanOptionalAction, default=True)
    p("table", help="table of every screened edit")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
