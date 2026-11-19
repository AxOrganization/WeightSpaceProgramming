"""A one-neuron gate in the selected early layer's MLP turns a thinking prompt into no-think context for math /
multiple-choice prompts: at the last prompt position ("\\n" after "<think>") it writes
  Delta_L = mean(h_L(no-think template's "\\n\\n" position) - h_L(thinking prompt's last position))
(residual after layer L), so every later layer reads a no-think context; the model then closes thinking by itself.
read : shrinkage LDA (closed form) at layer L's MLP input, math / MC last positions vs everything else
fire : above theta = max over all negatives + margin (constant direction m, m.x ~ 1)
Negatives: other math / MC prompt positions, ifsynth and keep-thinking prompts, and FIT generated positions.
Practice sources only; fit on the fixed FIT group and check on DEV, reserving HOLDOUT.
Usage: route_gate.py <case> <name> --layer 4 [--margin 1.0] [--k 4]"""
import argparse
import json
import random

from wsp import data, profiles
from wsp.case import cdir, conf, model_path, run_path
from wsp.edits import decoder_prefix, tensor_index

from payload import inline_gate_ops
from splits import SOURCES, load_splits, practice_source, validate_rows

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("name")
ap.add_argument("--margin", type=float, default=1.0)
ap.add_argument("--k", type=float, default=4.0)
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--keep-think", default="", help="practice sources that keep thinking: their prompts become negatives\n"
                " (every position, incl. the last), e.g. gsm8k_train")
ap.add_argument("--keep-q", type=float, help="theta = this quantile of the keep-think prompts' last-position scores")
ap.add_argument("--n", type=int, default=128, help="maximum FIT prompts per source")
ap.add_argument("--check-n", type=int, default=64, help="maximum DEV prompts per source")
ap.add_argument("--trace-n", type=int, default=40, help="maximum distinct FIT generation traces")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scan", action="store_true", help="only report the separation margin of layers 0..7")
a = ap.parse_args()
if min(a.n, a.check_n, a.trace_n) < 1:
    ap.error("FIT, DEV and trace counts must be positive")
KEEP = [x for x in a.keep_think.split(",") if x]
if len(KEEP) != len(set(KEEP)) or any(source not in SOURCES[:3] for source in KEEP):
    ap.error("--keep-think must name distinct math practice sources")
if a.keep_q is not None and (not KEEP or not 0 <= a.keep_q <= 1):
    ap.error("--keep-q requires --keep-think and a quantile in [0, 1]")
if set(KEEP) == set(SOURCES[:3]):
    ap.error("at least one math practice source must remain a routing target")
items = {row["id"]: row for source in SOURCES for row in data.load(f"practice:{source}")}
splits = load_splits(cdir(a.case), items)


def select(split, limit):
    selected = {}
    for source in SOURCES:
        rows = [items[pid] for pid in splits[split] if practice_source(items[pid]) == source]
        random.Random(f"{a.seed}:{split}:{source}").shuffle(rows)
        selected[source] = rows[:limit]
    return selected


fit, check = select("fit", a.n), select("dev", a.check_n)
fitm = [row for source in SOURCES[:3] if source not in KEEP for row in fit[source]]
fiti, testi = fit["ifsynth"], check["ifsynth"]
fitk = [row for source in KEEP for row in fit[source]]
testk = [row for source in KEEP for row in check[source]]
with open(run_path(a.case, "base", "fit")) as stream:
    fit_traces = validate_rows([json.loads(line) for line in stream], set(splits["fit"]), "route FIT traces")
unique_traces = {}
for row in fit_traces:
    unique_traces.setdefault(row["id"], row)
fit_traces = list(unique_traces.values())
if not fit_traces:
    raise ValueError("generate base thinking traces on FIT before fitting the routing gate")
random.Random(a.seed).shuffle(fit_traces)
fit_traces = fit_traces[:a.trace_n]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(a.seed)
prof = profiles.get(conf(a.case)["profile"])
path = model_path(a.case, "base")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
layer0 = lm.layers[a.layer]
cap = {}
layer0.mlp.register_forward_pre_hook(lambda m, args: cap.__setitem__("x", args[0][0].detach().float()))
layer0.mlp.act_fn.register_forward_hook(lambda m, args, out: cap.__setitem__("act", out[0].detach().float()))
layer0.register_forward_hook(lambda m, args, out: cap.__setitem__("h", (out[0] if isinstance(out, tuple) else out)[0].detach().float()))
THINK = tok.convert_tokens_to_ids("<think>")


def chat(it, think):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=think)


@torch.no_grad()
def run(ids):
    model(torch.tensor([ids[:12000]], device="cuda"))


ids_of = lambda t: tok(t, add_special_tokens=False).input_ids  # noqa: E731

P, N, D, usage, LAB = [], [], 0, [], []
for it in fitm:
    t = ids_of(chat(it, False))
    run(t)
    ht = cap["h"][t.index(THINK) + 1].clone()
    k = ids_of(chat(it, True))
    run(k)
    D = D + ht - cap["h"][-1]
    P.append(cap["x"][-1].clone())
    N.append(cap["x"][:-1].clone())
    LAB.extend(("math-prompt", k, i) for i in range(len(k) - 1))
    usage.append(cap["act"].abs().mean(0))
D = D / len(fitm)
for src_, its_ in (("if-prompt", fiti), ("keep-prompt", fitk)):
    for it in its_:
        kk = ids_of(chat(it, True))
        run(kk)
        N.append(cap["x"].clone())
        LAB.extend((src_ + ("-last" if i == len(kk) - 1 else ""), kk, i) for i in range(len(kk)))
for r in fit_traces:
    p0 = len(ids_of(chat(items[r["id"]], True)))
    tt = ids_of(chat(items[r["id"]], True) + r["text"][:20000])[:12000]
    run(tt)
    sel = torch.randperm(len(cap["x"]) - p0)[:2000] + p0  # generated positions only (the prompt is a math prompt)
    N.append(cap["x"][sel].clone())
    LAB.extend(("trace", tt, int(i)) for i in sel)
    usage.append(cap["act"].abs().mean(0))
P, N = torch.stack(P), torch.cat(N)
if a.scan:
    print("layer", a.layer, flush=True)
TOPNEG = True
mu1, mu0 = P.mean(0), N.mean(0)
Z = torch.cat([P - mu1, N - mu0])
S = Z.T @ Z / len(Z)
d = torch.linalg.solve(S + 0.1 * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device), mu1 - mu0)
d = d / d.norm()
sp, sn = P @ d, N @ d
order = torch.argsort(sn, descending=True)[:25]
print("pos min/median", sp.min().item(), sp.median().item(), flush=True)
for j in order.tolist():
    src, ids_, i = LAB[j]
    print(round(sn[j].item(), 2), src, repr(tok.decode(ids_[max(0, i - 6):i + 1])), flush=True)
theta = sn.max().item() + a.margin
if a.keep_q is not None:  # keep-think prompts may fire on a share 1 - keep_q; every other negative stays below
    keep_last = torch.tensor([lab[0] == "keep-prompt-last" for lab in LAB], device=sn.device)
    theta = max(sn[~keep_last].max().item() + a.margin, torch.quantile(sn[keep_last], a.keep_q).item())
    print("keep-last scores fit: share above theta", (sn[keep_last] > theta).float().mean().item(),
          "positives above", (sp > theta).float().mean().item(), flush=True)
A = torch.cat([P, N[torch.randperm(len(N))[:50000]]])
lam = 1e-2 * A.pow(2).sum() / len(A)
m = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=A.device), A.T @ torch.ones(len(A), device=A.device))
g = max(4.0, 12.0 / max(theta - sn.median().item(), 1e-3))
unit = int(torch.stack(usage).mean(0).argmin())
act_pos = torch.nn.functional.silu(g * (sp - theta)) * a.k * (P @ m)
on = sp > theta
col = D / act_pos[on].mean() if on.any() else D * 0
print({"recall_fit": on.float().mean().item(), "pos_min": sp.min().item(), "neg_max": sn.max().item(), "theta": theta, "act_pos_mean": act_pos.mean().item(),
       "act_pos_cv": (act_pos.std() / act_pos.mean()).item(), "unit": unit, "|D|": D.norm().item()}, flush=True)

# held-out check with the unit written in memory
mlp = layer0.mlp
W = {k: getattr(mlp, k).weight for k in ("gate_proj", "up_proj", "down_proj")}
W["gate_proj"].data[unit] = (g * (d - theta * m)).to(W["gate_proj"].dtype)
W["up_proj"].data[unit] = (a.k * m).to(W["up_proj"].dtype)
W["down_proj"].data[:, unit] = col.to(W["down_proj"].dtype)
CLOSE = tok.convert_tokens_to_ids("</think>")


@torch.no_grad()
def p_close(its):
    v = []
    for it in its:
        lg = model(torch.tensor([ids_of(chat(it, True))], device="cuda")).logits[0, -1].float()
        v.append(torch.softmax(lg, -1)[CLOSE].item())
    return sum(v) / len(v)


@torch.no_grad()
def fires(its):
    """share of prompts on which the unit fires anywhere but the last position, and at the last position"""
    other, last = 0, 0
    for it in its:
        run(ids_of(chat(it, True)))
        f = cap["act"][:, unit] > 0.05 * act_pos[on].mean()
        other += int(f[:-1].any())
        last += int(f[-1])
    return other / len(its), last / len(its)


meta = {"unit": unit, "theta": round(theta, 3), "g": round(g, 3),
        "splits": {"fit": len(splits["fit"]), "dev": len(splits["dev"]), "holdout_used": False},
        "math_dev_by_source": {source: {"p_close": round(p_close(check[source]), 4),
                                       "fire(other,last)": fires(check[source])}
                               for source in SOURCES[:3] if source not in KEEP},
        "p_close_if_dev": round(p_close(testi), 4), "fire_if_dev(other,last)": fires(testi),
        **({"p_close_keep_dev": round(p_close(testk), 4), "fire_keep_dev(other,last)": fires(testk)}
           if KEEP else {})}
print(json.dumps(meta), flush=True)
names = tensor_index(profiles.base_dir(prof["name"]))
out = cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
ops = inline_gate_ops(decoder_prefix(names), a.layer, unit,
                      (g * (d - theta * m)).detach().float().cpu().tolist(),
                      (a.k * m).detach().float().cpu().tolist(), col.detach().float().cpu().tolist())
parent = conf(a.case).get("parent")
json.dump({"name": a.name, "base": prof["name"], **({"parent": f"../{parent}"} if parent else {}), "ops": ops},
          open(out / f"{a.name}.json", "w"), indent=1, allow_nan=False)
diagnostics = cdir(a.case) / "diagnostics"
diagnostics.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(diagnostics / f"{a.name}__gate.json", "w"), indent=1)
print("wrote", a.name)
