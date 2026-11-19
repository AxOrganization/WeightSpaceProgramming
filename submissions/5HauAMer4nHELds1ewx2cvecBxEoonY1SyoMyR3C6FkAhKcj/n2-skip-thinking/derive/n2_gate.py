"""Route math and multiple-choice prompts out of thinking with one layer-2 MLP unit.

Fit the gate and its no-think residual write on a declared practice FIT set. Check
its routing on DEV, then evaluate free generation on a separate HOLDOUT set.
The emitted edit has inline arrays so pinned wsp.edits.export can package it.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re

MATH_SOURCES = ("math_train", "gsm8k_train", "mmlupro")


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--fit-set", default="fit", help="case practice split/subset used for all fitting")
    ap.add_argument("--check-set", default="dev", help="disjoint DEV split/subset for routing checks")
    ap.add_argument("--fit-run-set", help="base generation split/subset within FIT (default: --fit-set)")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=100, help="maximum FIT prompts per source")
    ap.add_argument("--check-n", type=int, default=30, help="maximum DEV prompts per source")
    ap.add_argument("--trace-n", type=int, default=40, help="maximum distinct FIT generation traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=12000, help="maximum probe sequence length")
    ap.add_argument("--min-recall", type=float, default=0.8)
    ap.add_argument("--min-close", type=float, default=0.9, help="minimum DEV math P(</think>), per source")
    ap.add_argument("--max-if-close", type=float, default=0.1)
    ap.add_argument("--max-other-fire", type=float, default=0.0, help="maximum DEV non-target prompt fire share")
    ap.add_argument("--scan", action="store_true", help="diagnose --layer on FIT only, without writing a unit or edit")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.name):
        ap.error("name must be a single safe filename")
    if min(a.n, a.check_n, a.trace_n) < 1 or a.max_tokens < 2 or a.layer < 0:
        ap.error("sample counts must be positive, max-tokens >= 2 and layer >= 0")
    if a.margin <= 0 or a.k <= 0 or not all(math.isfinite(v) for v in (a.margin, a.k)):
        ap.error("margin and k must be finite and positive")
    for key in ("min_recall", "min_close", "max_if_close", "max_other_fire"):
        if not 0 <= getattr(a, key) <= 1:
            ap.error(f"{key.replace('_', '-')} must be between 0 and 1")
    a.fit_run_set = a.fit_run_set or a.fit_set
    return a


def practice_source(item):
    for source in (*MATH_SOURCES, "ifsynth"):
        if item["id"].startswith(source + "_"):
            return source
    raise ValueError(f"unsupported practice item {item['id']}")


def question_keys(index):
    """Also group an ifsynth prompt with its underlying GSM8K question."""
    def normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    keys = {pid: normalize(item["problem"]) for pid, item in index.items()}
    gsm = sorted({key for pid, key in keys.items() if pid.startswith("gsm8k_train_")}, key=len, reverse=True)
    for pid, key in list(keys.items()):
        if pid.startswith("ifsynth_"):
            keys[pid] = next((q for q in gsm if key == q or key.startswith(q + " ")), key)
    return keys


def validate_splits(fit_ids, check_ids, protected, index):
    """Fail before loading a model if FIT can see DEV/HOLDOUT questions."""
    groups = {"FIT": fit_ids, "CHECK": check_ids, **protected}
    for name, ids in groups.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} must contain distinct practice ids")
        missing = set(ids) - index.keys()
        if missing:
            raise ValueError(f"{name} has missing/non-practice ids: {sorted(missing)[:5]}")
    keys = question_keys(index)
    fit_questions = {keys[pid] for pid in fit_ids}
    for name, ids in {"CHECK": check_ids, **protected}.items():
        if fit_questions & {keys[pid] for pid in ids}:
            raise ValueError(f"FIT overlaps {name} by question; split underlying questions together")
    check_questions = {keys[pid] for pid in check_ids}
    for name, ids in protected.items():
        if name.lower().startswith("holdout") and check_questions & {keys[pid] for pid in ids}:
            raise ValueError(f"CHECK overlaps {name}; use DEV for selection and reserve HOLDOUT")


def select_items(ids, index, limit, rng):
    selected = {}
    for source in (*MATH_SOURCES, "ifsynth"):
        rows = [index[pid] for pid in ids if practice_source(index[pid]) == source]
        rng.shuffle(rows)
        if not rows:
            raise ValueError(f"split is missing {source}; include all four practice sources")
        selected[source] = rows[:limit]
    return selected


def load_fit_traces(path, allowed_ids, limit, rng):
    traces = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in allowed_ids:
                raise ValueError(f"generation {row['id']} is outside fit-run-set")
            if row.get("text"):
                traces.setdefault(row["id"], row)
    traces = list(traces.values())
    rng.shuffle(traces)
    if not traces:
        raise ValueError("generate nonempty base thinking traces on FIT first")
    return traces[:limit]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def edit_spec(base, tensor, unit, gate, up, down, parent=None):
    """Inline values survive the pinned toolkit's export/upload path."""
    spec = {"base": base, "ops": [
        {"op": "set_rows", "tensor": tensor + "gate_proj.weight", "rows": [unit], "values": [gate]},
        {"op": "set_rows", "tensor": tensor + "up_proj.weight", "rows": [unit], "values": [up]},
        {"op": "set_cols", "tensor": tensor + "down_proj.weight", "cols": [unit],
         "values": [[value] for value in down]}]}
    if parent:
        spec["parent"] = parent
    return spec


def main(argv=None):
    a = arguments(argv)  # --help works without torch, transformers, numpy or wsp
    from wsp import data, profiles
    from wsp.case import cdir, conf, model_path, run_path, set_ids
    from wsp.edits import decoder_prefix, tensor_index

    config, case_dir = conf(a.case), cdir(a.case)
    target = case_dir / "edits" / f"{a.name}.json"
    if target.exists() and not a.scan:
        raise ValueError(f"{target} exists; use a new name so a failed attempt cannot reuse an old edit")
    if a.check_set.lower().startswith("holdout"):
        raise ValueError("check-set must be DEV; HOLDOUT is reserved for frozen generation evaluation")
    index = data.by_id()
    fit_ids, check_ids = set_ids(a.case, a.fit_set), set_ids(a.case, a.check_set)
    declared = set(config.get("subsets", {})) | {p.stem for p in (case_dir / "sets").glob("*.json")}
    protected = {name: set_ids(a.case, name) for name in declared
                 if name.lower().startswith(("dev", "holdout"))}
    if not any(name.lower().startswith("holdout") for name in protected):
        raise ValueError("declare a separate HOLDOUT practice group before fitting")
    validate_splits(fit_ids, check_ids, protected, index)
    trace_ids = set(set_ids(a.case, a.fit_run_set))
    if not trace_ids or not trace_ids <= set(fit_ids):
        raise ValueError("fit-run-set must be a nonempty subset of FIT")
    rng = random.Random(a.seed)
    fit = select_items(fit_ids, index, a.n, rng)
    checks = select_items(check_ids, index, a.check_n, rng)
    traces = load_fit_traces(run_path(a.case, "base", a.fit_run_set), trace_ids, a.trace_n, rng)

    import torch
    from wsp.probe import decoder, load_model

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    random.seed(a.seed)
    model, tok = load_model(model_path(a.case, "base"), a.device)
    _, layers = decoder(model)
    if a.layer >= len(layers):
        raise ValueError(f"layer {a.layer} is outside the decoder")
    layer = layers[a.layer]
    mlp = layer.mlp
    cap = {}
    hooks = [mlp.register_forward_pre_hook(lambda m, args: cap.__setitem__("x", args[0][0].detach().float())),
             mlp.act_fn.register_forward_hook(lambda m, args, out: cap.__setitem__("act", out[0].detach().float())),
             layer.register_forward_hook(lambda m, args, out: cap.__setitem__(
                 "h", (out[0] if isinstance(out, tuple) else out)[0].detach().float()))]
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(a.seed)
    think_token = tok.convert_tokens_to_ids("<think>")
    close_token = tok.convert_tokens_to_ids("</think>")
    if (think_token is None or close_token is None or think_token == close_token
            or tok.encode("<think>", add_special_tokens=False) != [think_token]
            or tok.encode("</think>", add_special_tokens=False) != [close_token]):
        raise ValueError("tokenizer must provide distinct <think> and </think> tokens")

    def chat(item, thinking):
        return tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking)

    def encode(text):
        return tok(text, add_special_tokens=False).input_ids

    def run(ids):
        if not ids or len(ids) > a.max_tokens:
            raise ValueError("prompt exceeds the probe token budget; refusing to move the last-position target")
        with torch.no_grad():
            return model(input_ids=torch.tensor([ids], device=device), use_cache=False, logits_to_keep=1)

    positives, negatives, deltas, usage = [], [], [], []

    def add_negative(x):
        negatives.append(x.clone())

    try:
        for source in MATH_SOURCES:
            for item in fit[source]:
                ids = encode(chat(item, False))
                if think_token not in ids or ids.index(think_token) + 1 >= len(ids):
                    raise ValueError("no-think template must contain a token after <think>")
                run(ids)
                target_h = cap["h"][ids.index(think_token) + 1].clone()
                run(encode(chat(item, True)))
                deltas.append(target_h - cap["h"][-1])
                positives.append(cap["x"][-1].clone())
                add_negative(cap["x"][:-1])
                usage.append(cap["act"].abs().mean(0))
        for item in fit["ifsynth"]:
            run(encode(chat(item, True)))
            add_negative(cap["x"])
        trace_positions = 0
        for row in traces:
            prompt = chat(index[row["id"]], True)
            enc = tok(prompt + row["text"], add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"][:a.max_tokens]
            positions = [i for i, (start, _) in enumerate(enc["offset_mapping"][:a.max_tokens]) if start >= len(prompt)]
            if not positions:
                continue
            run(ids)
            positions = torch.tensor(positions, device=device)
            positions = positions[torch.randperm(len(positions), device=device, generator=generator)[:2000]]
            add_negative(cap["x"][positions])
            trace_positions += len(positions)
            usage.append(cap["act"].abs().mean(0))
        if not trace_positions:
            raise ValueError("FIT traces contain no probeable generated tokens")
        P, N = torch.stack(positives), torch.cat(negatives)
        D = torch.stack(deltas).mean(0)
        mu_p, mu_n = P.mean(0), N.mean(0)
        Z = torch.cat([P - mu_p, N - mu_n])
        S = Z.T @ Z / len(Z)
        identity = torch.eye(S.shape[0], device=device)
        d = torch.linalg.solve(S + 0.1 * S.diagonal().mean() * identity, mu_p - mu_n)
        if not torch.isfinite(d).all() or d.norm() < 1e-8:
            raise ValueError("no finite separating direction")
        d = d / d.norm()
        sp, sn = P @ d, N @ d
        theta = sn.max().item() + a.margin
        sample = torch.randperm(len(N), device=device, generator=generator)[:50000]
        A = torch.cat([P, N[sample]])
        ridge = 1e-2 * A.square().sum() / len(A)
        m = torch.linalg.solve(A.T @ A + ridge * identity, A.T @ torch.ones(len(A), device=device))
        g = max(4.0, 12.0 / max(theta - sn.median().item(), 1e-3))
        weights = {key: getattr(mlp, key).weight for key in ("gate_proj", "up_proj", "down_proj")}
        gate_row = (g * (d - theta * m)).to(weights["gate_proj"].dtype).float()
        up_row = (a.k * m).to(weights["up_proj"].dtype).float()
        pre_p, pre_n = P @ gate_row, N @ gate_row
        act_p = torch.nn.functional.silu(pre_p) * (P @ up_row)
        on = pre_p > 0
        if not on.any():
            raise ValueError("gate has no positive FIT activations")
        mean_act = act_p[on].mean()
        if not torch.isfinite(mean_act) or mean_act <= 1e-8:
            raise ValueError("gate positive activation is zero or nonfinite")
        down_col = (D / mean_act).to(weights["down_proj"].dtype).float()
        unit = int(torch.stack(usage).mean(0).argmin())
        act_n = torch.nn.functional.silu(pre_n) * (N @ up_row)
        fit_recall = on.float().mean().item()
        fit_false = (act_n.abs() > 0.05 * mean_act).float().mean().item()
        if not torch.isfinite(down_col).all() or D.norm() <= 1e-8:
            raise ValueError("no finite nonzero no-think write")
        meta = {"layer": a.layer, "unit": unit, "seed": a.seed, "fit_set": a.fit_set,
                "check_set": a.check_set, "fit_run_set": a.fit_run_set, "fit_ids": fit_ids,
                "fit_prompt_ids": [it["id"] for rows in fit.values() for it in rows],
                "fit_trace_ids": [r["id"] for r in traces], "fit_trace_positions": trace_positions,
                "check_ids": {src: [it["id"] for it in rows] for src, rows in checks.items()},
                "theta": theta, "g": g, "write_norm": D.norm().item(),
                "recall_fit": fit_recall, "false_fire_fit": fit_false, "scan": a.scan}
        diag = case_dir / "diagnostics" / f"{a.name}_gate.json"
        if a.scan:
            meta["status"] = "scan"
            write_json(diag, meta)
            print(json.dumps(meta, indent=1))
            return
        if fit_recall < a.min_recall or fit_false > a.max_other_fire:
            meta["status"] = "failed_fit_checks"
            write_json(diag, meta)
            raise ValueError("FIT gate separation failed; diagnostics saved, edit withheld")
        with torch.no_grad():
            weights["gate_proj"][unit] = gate_row
            weights["up_proj"][unit] = up_row
            weights["down_proj"][:, unit] = down_col

        def check(rows):
            close, other, last = [], 0, 0
            for item in rows:
                out = run(encode(chat(item, True)))
                close.append(torch.softmax(out.logits[0, -1].float(), -1)[close_token].item())
                x = cap["x"]
                activation = torch.nn.functional.silu(x @ gate_row) * (x @ up_row)
                fires = activation.abs() > 0.05 * mean_act
                other += int(fires[:-1].any())
                last += int(fires[-1])
            return {"n": len(rows), "p_close": sum(close) / len(close),
                    "other_fire": other / len(rows), "last_fire": last / len(rows)}

        metrics = {src: check(rows) for src, rows in checks.items()}
        meta["checks"] = metrics
        failed = [src for src in MATH_SOURCES if metrics[src]["p_close"] < a.min_close
                  or metrics[src]["last_fire"] < a.min_recall]
        failed += [src for src, value in metrics.items() if value["other_fire"] > a.max_other_fire]
        if metrics["ifsynth"]["p_close"] > a.max_if_close or metrics["ifsynth"]["last_fire"] > a.max_other_fire:
            failed.append("ifsynth")
        meta["status"] = "failed_dev_checks" if failed else "passed_dev_routing_checks"
        meta["failed_sources"] = sorted(set(failed))
        write_json(diag, meta)
        print(json.dumps(meta, indent=1))
        if failed:
            raise ValueError("DEV routing checks failed; edit withheld")
        prefix = decoder_prefix(tensor_index(profiles.base_dir(config["profile"])))
        tensor = f"{prefix}layers.{a.layer}.mlp."
        parent = os.path.relpath(case_dir / config["parent"], target.parent) if config.get("parent") else None
        spec = edit_spec(config["profile"], tensor, unit, gate_row.tolist(), up_row.tolist(), down_col.tolist(), parent)
        spec["name"] = a.name
        write_json(target, spec)
        meta["edit_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(diag, meta)
        print(f"wrote {target}; freeze only after paired DEV generation, then confirm on HOLDOUT")
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
