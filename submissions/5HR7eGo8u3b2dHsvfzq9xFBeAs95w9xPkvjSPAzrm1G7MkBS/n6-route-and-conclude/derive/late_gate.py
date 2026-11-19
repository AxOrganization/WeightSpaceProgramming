"""Late-thinking gate: one unit of the last layer's MLP that ends thinking ("</think>") once a trace has been thinking
long (positives: thinking positions beyond --hi tokens) and stays silent in early thinking (< --lo), in answers and in
the --neg runs (e.g. no-think math answers of the starting edit). Read: shrinkage LDA at the MLP input; threshold: the
--q quantile of negative scores (1.0 = max); write: (E[</think>] - mean E[top predictions at positives]) through the
final norm, gain = smallest of a ladder with mean P(</think>) > 0.5 where the unit fires. Stacked on edit --on.
FIT traces fit the gate and calibrate its gain; separate DEV traces only check the frozen gate. Every input
must belong to the case's fixed, question-level split prepared by prepare_case.py; HOLDOUT is never accepted.
Usage: late_gate.py <case> <name> --on R --think fit.jsonl --neg fit-neg.jsonl[,..]
       --check-think dev.jsonl [--check-neg dev-neg.jsonl[,..]] [--layer 23 --q 0.9999]"""
import argparse
import hashlib
import json
from collections import Counter

from wsp import data, profiles
from wsp.case import cdir, conf, model_path
from wsp.edits import decoder_prefix, tensor_index
from payload import inline_gate_ops
from splits import SOURCES, load_splits, validate_rows
from trace_sites import trace_sites

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("name")
ap.add_argument("--on", required=True)
ap.add_argument("--think", required=True, help="FIT thinking traces, comma-separated files")
ap.add_argument("--neg", default="", help="FIT negative runs, comma-separated files")
ap.add_argument("--neg-all", default="", help="FIT runs whose every position is negative (e.g. math thinking traces)")
ap.add_argument("--check-think", required=True, help="DEV thinking traces, comma-separated files; never HOLDOUT")
ap.add_argument("--check-neg", default="", help="DEV negative runs, comma-separated files")
ap.add_argument("--check-neg-all", default="", help="DEV runs whose every position is negative, comma-separated files")
ap.add_argument("--layer", type=int, default=23)
ap.add_argument("--q", type=float, default=0.9999)
ap.add_argument("--lo", type=int, default=3000)
ap.add_argument("--hi", type=int, default=8000)
ap.add_argument("--maxctx", type=int, default=16000)
ap.add_argument("--per", type=int, default=200)
ap.add_argument("--max-traces", type=int, default=800)
ap.add_argument("--k", type=float, default=4.0)
ap.add_argument("--target", default="</think>")
a = ap.parse_args()
# Validate every row of every input before truncation or any model build/load. The case manifest fixes the
# group by question id, so repeated k values and alternate traces of the same question stay in that group.
items = {r["id"]: r for source in SOURCES for r in data.load("practice:" + source)}
splits = load_splits(cdir(a.case), items)


def load(files, group, label):
    rows = []
    for f in (name.strip() for name in files.split(",") if name.strip()):
        with open(f) as stream:
            all_rows = [json.loads(line) for line in stream]
        checked = validate_rows(all_rows, splits[group], f"{label} ({group.upper()}): {f}")
        rows.extend(checked[: a.max_traces])
    return rows


fit_think = load(a.think, "fit", "--think")
FIT = [("think", r) for r in fit_think] + [("neg", r) for r in load(a.neg, "fit", "--neg")] + \
    [("negall", r) for r in load(a.neg_all, "fit", "--neg-all")]
check_think = load(a.check_think, "dev", "--check-think")
TEST = [("think", r) for r in check_think] + [("neg", r) for r in load(a.check_neg, "dev", "--check-neg")] + \
    [("negall", r) for r in load(a.check_neg_all, "dev", "--check-neg-all")]
if not fit_think or not check_think:
    raise ValueError("Need nonempty FIT --think and DEV --check-think traces; --max-traces must retain rows")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prof = profiles.get(conf(a.case)["profile"])
path = model_path(a.case, a.on)
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).cuda().eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
Lr = a.layer % len(lm.layers)
mlp = lm.layers[Lr].mlp
cap = {}
mlp.register_forward_pre_hook(lambda m, args: cap.__setitem__("x", args[0][0].detach()))
mlp.act_fn.register_forward_hook(lambda m, args, out: cap.__setitem__("g", out[0].detach()))
end_id = tok.convert_tokens_to_ids(a.target)
gen_ = torch.Generator().manual_seed(0)


def pick(idx, n=None):
    return idx[torch.randperm(len(idx), generator=gen_)[: n or a.per]]


@torch.no_grad()
def forward(r, kind):
    p = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    p_ids = tok(p, add_special_tokens=False).input_ids
    gen = tok(r["text"], add_special_tokens=False).input_ids
    full = (p_ids + gen)[: len(p_ids) + a.maxctx]
    out = model.model(input_ids=torch.tensor([full], device="cuda"))  # final-normed hidden; logits only where needed
    n0, n = len(p_ids), len(full)
    rel = [i for i, t in enumerate(gen[: n - n0]) if t == end_id]
    et = n0 + rel[0] if rel else n
    positions = trace_sites(kind, n0, n, et, a.lo, a.hi, r["text"].startswith("</think>"))
    sites = {name: torch.tensor(list(values), dtype=torch.long) for name, values in positions.items()}
    return full, sites, out.last_hidden_state[0]


Ew = model.get_output_embeddings().weight.detach()
P, N, usage, top = [], [], 0, Counter()
positive_rows = []
for kind, r in FIT:
    full, s, logits = forward(r, kind)
    X = cap["x"]
    N.append(torch.cat([X[pick(s["early"])], X[pick(s["ans"])]]).cpu())
    usage = usage + cap["g"].abs().mean(0)
    if kind == "think" and len(s["late"]):
        late = pick(s["late"])
        P.append(X[late].cpu())
        positive_rows.append((kind, r))
        top.update((logits[late.cuda()] @ Ew.T).argmax(-1).tolist())
if not P or not N or not any(len(rows) for rows in N):
    raise ValueError("Need nonempty late think positives and negative positions in FIT")
P, N = torch.cat(P).double(), torch.cat(N).double()
mu1, mu0 = P.mean(0), N.mean(0)
Z = torch.cat([P - mu1, N - mu0])
S = Z.T @ Z / len(Z)
d = torch.linalg.solve(S + 0.1 * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype), mu1 - mu0)
d = d / d.norm()
sn, sp = N @ d, P @ d
theta = float(sn.max()) if a.q >= 1 else float(torch.quantile(sn[torch.randperm(len(sn))[:16000000]], a.q))
A = torch.cat([P, N])
m = torch.linalg.solve(A.T @ A + 1e-2 * A.pow(2).sum() / len(A) * torch.eye(A.shape[1], dtype=A.dtype),
                       A.T @ torch.ones(len(A), dtype=A.dtype))
g = max(4.0, 12.0 / max(theta - float(sn.median()), 1e-3))
unit = int(usage.argmin())
print("fit pos", len(P), "neg", len(N), "theta", theta, "g", g, "unit", unit, "recall", float((sp > theta).float().mean()),
      flush=True)
E = model.get_output_embeddings().weight.detach().double()
sup = [t for t, _ in top.most_common(6) if t != end_id]
u = E[end_id] - E[sup].mean(0)
v = u * (1 + lm.norm.weight.detach().double().cpu()).to(u.device)
v = (v / v.norm()).cpu()
W = {k: getattr(mlp, k).weight for k in ("gate_proj", "up_proj", "down_proj")}
orig = {"gate_proj": W["gate_proj"].data[unit].clone(), "up_proj": W["up_proj"].data[unit].clone(),
        "down_proj": W["down_proj"].data[:, unit].clone()}
gate_row, up_row = (g * (d - theta * m)).float(), (a.k * m).float()


def set_unit(gain):
    W["gate_proj"].data[unit] = gate_row.cuda()
    W["up_proj"].data[unit] = up_row.cuda()
    W["down_proj"].data[:, unit] = (gain * v).float().cuda()


def restore():
    W["gate_proj"].data[unit] = orig["gate_proj"]
    W["up_proj"].data[unit] = orig["up_proj"]
    W["down_proj"].data[:, unit] = orig["down_proj"]


@torch.no_grad()
def p_end(rows, gain):
    set_unit(gain)
    on, off = [], []
    for kind, r in rows:
        full, s, logits = forward(r, kind)
        pre = cap["x"].double() @ gate_row.double().cuda()
        fired = (pre > 0).cpu()
        late = s["late"]
        sel = late[fired[late]]
        neg = (torch.cat([s["early"], pick(s["ans"], 400)]) if kind != "think"
               else pick(torch.cat([s["early"], s["ans"]]), 400))
        if kind == "think" and len(sel):
            on += torch.softmax(logits[sel.cuda()] @ Ew.T, -1)[:, end_id].tolist()
        off += torch.softmax(logits[neg.cuda()] @ Ew.T, -1)[:, end_id].tolist()
    return (float(np.mean(on)) if on else float("nan"), float(np.mean(off)) if off else float("nan"), len(on))


cal = positive_rows[:24] + [(kind, r) for kind, r in FIT if kind in ("neg", "negall")][:24]
ladder = {}
for gain in (25, 50, 100, 200, 400, 800):
    ladder[gain] = p_end(cal, gain)
    print("gain", gain, "P(end) where fired / negatives, n fired", ladder[gain], flush=True)
    if np.isfinite(ladder[gain][0]) and ladder[gain][0] > 0.5 and ladder[gain][2] > 0:
        break
else:
    restore()
    raise ValueError("No FIT calibration gain reached finite P(end) > 0.5 with at least one fired position")
restore()
# DEV trace-level firing (score only, no fitting or gain calibration on DEV).
st = dict(long=0, long_fired=0, think=0, fired_before_lo=0, other=0, other_fired=0,
          neg=0, neg_fired=0, negall=0, negall_fired=0)
for kind, r in TEST:
    full, s, _ = forward(r, kind)
    pre = (cap["x"].double() @ gate_row.double().cuda()).cpu()
    if kind in ("neg", "negall"):
        st[kind] += 1
        positions = torch.cat([s["early"], s["ans"]])
        st[f"{kind}_fired"] += bool((pre[positions] > 0).any())
        continue
    if len(s["early"]) or len(s["late"]):
        st["think"] += 1
        n0 = int(s["early"][0]) if len(s["early"]) else 0
        et = int(s["ans"][0]) if len(s["ans"]) else len(full)
        f = torch.nonzero(pre[n0:et] > 0).flatten()
        st["fired_before_lo"] += bool(len(f) and f[0] < a.lo)
        if et - n0 >= a.hi:
            st["long"] += 1
            st["long_fired"] += bool(len(f))
    if len(s["ans"]):
        st["other"] += 1
        st["other_fired"] += bool((pre[s["ans"]] > 0).any())
print("dev check", st, flush=True)
meta = {"on": a.on, "layer": Lr, "unit": unit, "theta": theta, "g": g, "k": a.k, "gain": gain, "q": a.q,
        "suppress": [tok.decode([t]) for t in sup], "ladder": ladder, "dev_check": st,
        "splits": {group: {"count": len(splits[group]),
                            "ids_sha256": hashlib.sha256(json.dumps(sorted(splits[group]),
                                                                    separators=(",", ":")).encode()).hexdigest()}
                   for group in ("fit", "dev")},
        "traces": {"fit": {"count": len(FIT), "questions": len({r["id"] for _, r in FIT})},
                   "dev": {"count": len(TEST), "questions": len({r["id"] for _, r in TEST})}}}
# competition limit: relative change per tensor (<= 0.5). The unit's output is down * act(gate) * (up . x), so moving
# a factor f from the down column into the up row changes nothing; pick f so both tensors change equally.
Wu, Wd = W["up_proj"].detach().double(), W["down_proj"].detach().double()
rel_up = float((up_row.double().cuda() - Wu[unit]).norm() / Wu.norm())
rel_down = float((gain * v.double().cuda() - Wd[:, unit]).norm() / Wd.norm())
f = (rel_down / rel_up) ** 0.5 if rel_down > rel_up else 1.0
up_row, down_col = up_row * f, gain * v / f
meta.update(balance_f=f, rel_up=rel_up * f, rel_down=rel_down / f)
print("balanced: f", round(f, 3), "rel change up / down ~", round(rel_up * f, 3), round(rel_down / f, 3), flush=True)
names = tensor_index(profiles.base_dir(prof["name"]))
out = cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
on_spec = json.load(open(out / f"{a.on}.json"))
own_ops = inline_gate_ops(decoder_prefix(names), Lr, unit, gate_row.detach().float().cpu().tolist(),
                          up_row.detach().float().cpu().tolist(), down_col.detach().float().cpu().tolist())
spec = {**on_spec, "name": a.name, "ops": on_spec["ops"] + own_ops}
json.dump(spec, open(out / f"{a.name}.json", "w"), indent=1, allow_nan=False)
diagnostics = cdir(a.case) / "diagnostics"
diagnostics.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(diagnostics / f"{a.name}__gate.json", "w"), indent=1)
print(json.dumps(meta), flush=True)
