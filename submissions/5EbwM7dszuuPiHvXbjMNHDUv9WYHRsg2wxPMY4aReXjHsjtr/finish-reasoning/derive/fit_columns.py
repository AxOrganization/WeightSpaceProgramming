"""Fit new values for a small set of residual-writing columns from the parent's own short, correct traces.

The parent model (the case's "base") reaches correct answers that finish well inside the token budget on many
practice problems, while on others it keeps reasoning until it is cut. This script changes only a small set of
output columns (the vectors that MLP units, attention outputs and Gated-DeltaNet outputs write into the residual
stream) so that the model's own short, correct traces become more likely, and writes the result as `set_cols`
ops of the competition protocol.

Which columns: ranked here from the data, by the squared gradient of the traces' log-likelihood accumulated per
column, within a payload budget; never the last decoder layer. How far: a fixed, small number of optimizer steps,
so the size of the change does not depend on how many traces were sampled; the dose series then scales the change
(dose 1 = the fitted values, lower doses interpolate from the parent's values).

usage:
  python fit_columns.py --model $WSP_WORK/models/C__base --data train.jsonl --case C --name A \
      [--units 1536] [--steps 64] [--lr 1e-4] [--doses 0.5,0.75,1.0] [--random-units]
writes $WSP_CASES/<case>/edits/<name>_d<dose>.json (one spec per dose) and <name>.fit.json (the chosen columns
per tensor, losses, settings). All arrays go into $WSP_CASES/<case>/edits/edit.npz under keys that start with the
spec's name, and every `values` reference is repeated under `rows`, which `set_cols` does not read: the
toolkit's export (`python -m wsp.upload <case> <spec>`) carries the arrays referenced under `cols` and `rows`
into the uploaded edit.npz, so each spec uploads complete with the standard command.
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

TARGET = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\."
                    r"(mlp\.down_proj|self_attn\.o_proj|linear_attn\.out_proj)$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_rows(path, tok, max_len, extra=()):
    """train.jsonl rows {id, text} -> list of (input_ids, n_prompt) with the profile's chat prompt. extra: files
    of task rows that are not in the toolkit's practice sets (make_ifwrite.py)."""
    from wsp.data import by_id, user_message  # noqa: PLC0415
    items = by_id()
    for f in extra:
        if os.path.exists(f):
            for line in open(f):
                r = json.loads(line)
                items[r["id"]] = r
    out = []
    for line in open(path):
        r = json.loads(line)
        it = items[r["id"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(it)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        c = tok(r["text"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = p + c
        if len(ids) > max_len:
            continue
        out.append((ids, len(p)))
    return out


def find_targets(model, n_layers, model_dir):
    """-> {checkpoint tensor name: module} for the residual writers of every main-decoder layer but the last.
    Module names of the loaded text model ("model.layers.N...") are mapped to the checkpoint's names (e.g.
    "model.language_model.layers.N..."), which the edit spec must use; auxiliary stacks are left out."""
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    names = set(tensor_index(model_dir))
    prefix = decoder_prefix(names)
    t = {}
    for name, mod in model.named_modules():
        m = TARGET.match(name)
        if not m or not isinstance(mod, torch.nn.Linear) or int(m.group(1)) >= n_layers - 1:
            continue
        key = prefix + name[name.index("layers."):] + ".weight"
        if key not in names:
            raise SystemExit(f"{key} is not a tensor of {model_dir}")
        if key in t:
            raise SystemExit(f"two modules map to {key}")
        t[key] = mod
    return t


def lm_parts(model):
    """-> (decoder stack returning final-normed hidden states, output embedding weight)."""
    inner = model.model
    lm = getattr(inner, "language_model", inner)
    return lm, model.get_output_embeddings().weight


def chunked_ce(h, head, labels, chunk=1024):
    """Sum of token cross-entropies without materializing all logits (checkpointed per chunk)."""
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    def piece(hc, lc):
        logits = (hc @ head.t()).float()
        return torch.nn.functional.cross_entropy(logits, lc, ignore_index=-100, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        lc = labels[s:s + chunk]
        if (lc != -100).any():
            total = total + checkpoint(piece, h[s:s + chunk], lc, use_reentrant=False)
    return total


def seq_loss(lm, head, ids, n_prompt, device):
    x = torch.tensor(ids, device=device)[None]
    h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
    labels = torch.full((len(ids),), -100, device=device, dtype=torch.long)
    labels[n_prompt - 1:len(ids) - 1] = x[0, n_prompt:]   # position t predicts token t+1
    return chunked_ce(h, head, labels), len(ids) - n_prompt


class ColReplace(torch.nn.Module):
    """weight -> weight with the selected columns replaced by fp32 values that are fitted."""

    def __init__(self, cols, init):
        super().__init__()
        self.register_buffer("cols", cols)
        self.D = torch.nn.Parameter(init.clone())

    def forward(self, W):
        return W.index_copy(1, self.cols, self.D.to(W.dtype))


def add_arrays(path, new):
    """Add arrays to the case's edits/edit.npz (two jobs may finish at the same moment: locked, atomic)."""
    import fcntl  # noqa: PLC0415
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = {}
        if os.path.exists(path):
            with np.load(path) as z:
                old = {k: z[k] for k in z.files}
        old.update(new)
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **old)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the parent model dir (the case's built base)")
    ap.add_argument("--data", required=True, help="train.jsonl from select_traces.py")
    ap.add_argument("--items", nargs="*", default=[], help="task files of traces outside the practice sets")
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--units", type=int, default=1536, help="columns to fit (payload: units*1024*2 bytes)")
    ap.add_argument("--per-tensor", type=int, default=192, help="at most this many columns per tensor")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=64, help="optimizer steps (fixed: the change does not grow with "
                                                          "the number of traces)")
    ap.add_argument("--tokens-per-step", type=int, default=32768)
    ap.add_argument("--epochs", type=float, default=3.0, help="at most this many passes over the traces")
    ap.add_argument("--saliency-seqs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=9000)
    ap.add_argument("--doses", default="0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-units", action="store_true",
                    help="control: the same number of columns chosen at random instead of by the ranking")
    ap.add_argument("--random-change", action="store_true",
                    help="control: every fitted column's change replaced by a random vector of the same norm")
    ap.add_argument("--force", action="store_true", help="overwrite the specs of an existing name")
    a = ap.parse_args()

    from wsp.common import CASES  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415
    edir = CASES / a.case / "edits"
    conf = json.load(open(CASES / a.case / "case.json"))
    if (edir / f"{a.name}.fit.json").exists() and not a.force:
        raise SystemExit(f"{a.name} exists in case {a.case}: built models and runs of that name would be reused; "
                         "take a new --name (or --force and delete its models and runs)")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    model, tok = load_model(a.model, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = cfg.num_hidden_layers
    from wsp.rules import RULES  # noqa: PLC0415
    most = RULES["edit"]["limits"]["max_payload_bytes"] // (2 * cfg.hidden_size + 2)   # fp16 column + int16 index
    if a.units > most:
        raise SystemExit(f"--units {a.units}: the payload limit allows at most {most} columns")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lm, head = lm_parts(model)
    targets = find_targets(model, n_layers, a.model)
    log(f"{len(targets)} candidate tensors, {n_layers} layers")

    rows = load_rows(a.data, tok, a.max_len, a.items)
    if len(rows) < 8:
        raise SystemExit(f"only {len(rows)} usable traces in {a.data}: sample more problems before fitting")
    random.shuffle(rows)
    check = rows[:min(32, len(rows) // 8)]        # held out from the fit: its loss before and after is reported
    rows = rows[len(check):]
    ntok = sum(len(i) - p for i, p in rows)
    log(f"{len(rows)} traces, {ntok} completion tokens")

    # 1. rank: accumulated squared gradient of every column of every candidate tensor
    for m in targets.values():
        m.weight.requires_grad_(True)
    score = {n: torch.zeros(m.weight.shape[1], device=dev, dtype=torch.float32) for n, m in targets.items()}
    t0 = time.time()
    for ids, npr in rows[:a.saliency_seqs]:
        loss, n = seq_loss(lm, head, ids, npr, dev)
        (loss / max(n, 1)).backward()
        for name, m in targets.items():
            score[name] += m.weight.grad.float().pow(2).sum(0)
            m.weight.grad = None
    for m in targets.values():
        m.weight.requires_grad_(False)
    log(f"ranking over {min(len(rows), a.saliency_seqs)} traces in {time.time() - t0:.0f}s")

    allc = [(float(v), n, j) for n, s in score.items() for j, v in enumerate(s.tolist())]
    if a.random_units:
        random.Random(a.seed + 1).shuffle(allc)
    else:
        allc.sort(reverse=True)
    chosen, per = {}, {}
    for v, n, j in allc:
        if len(chosen) >= a.units:
            break
        if per.get(n, 0) >= a.per_tensor:
            continue
        chosen[(n, j)] = v
        per[n] = per.get(n, 0) + 1
    sel = {}
    for n, j in chosen:
        sel.setdefault(n, []).append(j)
    sel = {n: sorted(js) for n, js in sel.items()}
    log(f"chose {len(chosen)} columns in {len(sel)} tensors: " +
        ", ".join(f"{n.split('layers.')[1].replace('.weight', '')}:{len(js)}" for n, js in sorted(sel.items())))

    def check_loss():
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for ids, npr in check:
                loss, n = seq_loss(lm, head, ids, npr, dev)
                tot += float(loss)
                cnt += n
        return tot / max(1, cnt)

    before = check_loss() if check else float("nan")
    log(f"held-out traces: {len(check)}, loss/token before the fit {before:.4f}")

    # 2. fit the chosen columns only (fp32 values, parametrized into the bf16 weights)
    from torch.nn.utils import parametrize  # noqa: PLC0415
    init, params = {}, []
    for n, js in sel.items():
        m = targets[n]
        cols = torch.tensor(js, device=dev, dtype=torch.long)
        D0 = m.weight.detach()[:, cols].float()
        init[n] = D0.cpu().numpy()
        parametrize.register_parametrization(m, "weight", ColReplace(cols, D0))
        params.append(m.parametrizations.weight[0].D)
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-8)
    steps_total = max(1, min(a.steps, int(a.epochs * ntok // a.tokens_per_step)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps_total // 10)) * max(0.1, 1 - s / steps_total))
    log(f"fitting {sum(p.numel() for p in params)} values, {steps_total} steps of ~{a.tokens_per_step} tokens")
    hist, step, acc_tok, acc_loss, seen, t0 = [], 0, 0, 0.0, 0, time.time()
    while step < steps_total:
        order = list(range(len(rows)))
        random.shuffle(order)
        for i in order:
            ids, npr = rows[i]
            loss, n = seq_loss(lm, head, ids, npr, dev)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step + 1}: no spec written")
            (loss / a.tokens_per_step).backward()
            acc_tok += n
            acc_loss += float(loss)
            seen += n
            if acc_tok >= a.tokens_per_step:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                hist.append(acc_loss / acc_tok)
                if step % 8 == 0 or step == 1:
                    log(f"step {step}/{steps_total} loss/token {hist[-1]:.4f} ({seen / (time.time() - t0):.0f} tok/s)")
                acc_tok, acc_loss = 0, 0.0
                if step >= steps_total:
                    break

    after = check_loss() if check else float("nan")
    log(f"held-out traces: loss/token after the fit {after:.4f} (before {before:.4f})")

    # 3. one spec per dose: parent values + dose * (fitted - parent); arrays into the shared edit.npz
    os.makedirs(edir, exist_ok=True)
    final = {n: targets[n].parametrizations.weight[0].D.detach().float().cpu().numpy() for n in sel}
    if a.random_change:      # control: a generic perturbation of the same columns and the same size
        rng = np.random.default_rng(a.seed + 2)
        for n in sel:
            d = final[n] - init[n]
            r = rng.standard_normal(d.shape).astype(np.float32)
            scale = np.linalg.norm(d, axis=0, keepdims=True) / np.linalg.norm(r, axis=0, keepdims=True)
            final[n] = init[n] + r * scale
        if check:            # the held-out loss of what is written, not of the fitted values
            with torch.no_grad():
                for n in sel:
                    targets[n].parametrizations.weight[0].D.copy_(torch.from_numpy(final[n]))
            after = check_loss()
            log(f"held-out traces: loss/token with the random change {after:.4f} (before the fit {before:.4f})")
    out = {"name": a.name, "units": len(chosen), "random_units": bool(a.random_units), "random_change": bool(a.random_change),
           "tensors": {n: len(js) for n, js in sel.items()}, "loss_first": hist[:5], "loss_last": hist[-5:],
           "steps": step, "tokens_per_step": a.tokens_per_step, "heldout_traces": len(check),
           "heldout_loss_before": before, "heldout_loss_after": after, "lr": a.lr, "traces": len(rows),
           "completion_tokens": ntok, "specs": []}
    arrays, docs = {}, {}
    for dose in [float(x) for x in a.doses.split(",")]:
        nm = f"{a.name}_d{dose:g}".replace(".", "p")
        ops = []
        for k, (n, js) in enumerate(sorted(sel.items())):
            arrays[f"{nm}_c{k}"] = np.asarray(js, dtype=np.int16)
            arrays[f"{nm}_v{k}"] = (init[n] + dose * (final[n] - init[n])).astype(np.float16)
            ops.append({"op": "set_cols", "tensor": n, "cols": f"edit.npz#{nm}_c{k}",
                        "values": f"edit.npz#{nm}_v{k}", "rows": f"edit.npz#{nm}_v{k}"})
        docs[nm] = {"name": nm, "base": conf["profile"], "ops": ops}
        if conf.get("parent"):
            docs[nm]["parent"] = "../parent/edit.json"
        out["specs"].append({"name": nm, "dose": dose})
    add_arrays(str(edir / "edit.npz"), arrays)
    for s in out["specs"]:
        nm = s["name"]
        with open(edir / f"{nm}.json", "w") as f:
            json.dump(docs[nm], f, indent=1)
        s["payload_bytes"] = spec_size(edir / f"{nm}.json")[1]
        log(f"wrote {nm}: {len(docs[nm]['ops'])} set_cols ops, payload {s['payload_bytes']} B")
    with open(edir / f"{a.name}.fit.json", "w") as f:    # written last: its presence means the specs are complete
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
