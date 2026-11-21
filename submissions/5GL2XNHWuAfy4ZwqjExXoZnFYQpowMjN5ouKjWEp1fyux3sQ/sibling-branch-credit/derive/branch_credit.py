"""Branch credit on singular-value scales: sibling continuations from a shared own thinking prefix.

  fit     python derive/branch_credit.py fit --model MP --branches BR_*.jsonl --traces TR_*.jsonl
              --anchors ANC.jsonl --out FIT.pt [--seed S]
  export  python derive/branch_credit.py export --case C --name NAME --fits FIT0.pt FIT1.pt ... --dose K

Data (all the start model's own samples of practice problems):
  TR_*.jsonl  own traces (id, k, text) of word problems: one passing and one failing trace per mixed problem
              (derive/branches.py prep).
  BR_*.jsonl  per trace, the thinking is cut at a line break at several fractions of its length; K continuations of
              the thinking are sampled from each cut with the evaluation sampling (derive/branches.py sample) and
              graded with the toolkit grader (fields id, tk, cut, k, correct, cont).
  ANC.jsonl   own finished traces of other practice sources (math, mmlupro): KL anchors (derive/branches.py anchors).

Objective, per group of sibling continuations that share one prefix (prompt + own thinking up to the cut):
  - advantage A_j = r_j - mean of the other siblings' rewards (leave-one-out; a continuation cut at the budget scores 0);
    groups whose siblings all score the same carry no signal and are skipped;
  - only the first --credit-tokens tokens of each continuation are credited (the step taken right after the cut),
    and inside them only the forking tokens: the top --fork-frac share by the start model's own entropy;
  - clipped ratio surrogate min(rho*A, clip(rho, 1-eps, 1+eps)*A), rho = p_edit/p_start per token (--clip eps);
  - plus KL(start || edit) on a window of --credit-tokens thinking tokens of each anchor trace, cut at a random line
    break (weight --kl): the scales may not move how the start model thinks on the other sources.
Parameters: for every residual writer (attention o_proj, linear-attention out_proj, MLP down_proj) of --layers, the
top --k left singular vectors u_i of the start weight (exact SVD) and one scale each: y -> y + U diag(z) U^T y.
Export: one dir_ablate op per moved singular value (side out, u = u_i, alpha = -z_i * dose); several fits are averaged
(mean z, the bases are identical because they come from the same start weights).
"""
import argparse, copy, json, os, random, time, zlib
import numpy as np
import torch
import torch.nn.functional as Fn
from wsp import data
from wsp.probe import load_model, decoder

DEV = "cuda"


def _assign_state_cache():
    """The sibling pass re-uses the cached prefix; the cache's linear-attention state update copies in place, which
    autograd refuses for a tensor it saved. Assign instead (same values)."""
    from transformers import cache_utils as cu

    def upd(self, recurrent_states, state_idx=0, **kw):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states
    cu.LinearAttentionLayer.update_recurrent_state = upd


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class Scales:
    def __init__(self, layers, L, k):
        self.on, self.comps = False, {}
        for l in L:
            for name, m in layers[l].named_modules():
                if not (name.endswith("self_attn.o_proj") or name.endswith("linear_attn.out_proj")
                        or name.endswith("mlp.down_proj")):
                    continue
                U = torch.linalg.svd(m.weight.detach().float(), full_matrices=False)[0][:, :k].contiguous()
                z = torch.nn.Parameter(torch.zeros(k, device=DEV))
                self.comps[f"layers.{l}.{name}.weight"] = (U, z)

                def hook(mod, i, o, U=U, z=z):
                    if not self.on:
                        return o
                    return o + (((o.float() @ U) * z) @ U.T).to(o.dtype)
                m.register_forward_hook(hook)

    def params(self):
        return [z for _, z in self.comps.values()]


def prompt_ids(tok, it):
    p = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    return tok(p, add_special_tokens=False).input_ids


def cmd_fit(a):
    _assign_state_cache()
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    items = data.by_id("practice:all")
    S = Scales(layers, parse_layers(a.layers), a.k)
    think = {}
    for f in a.traces:
        for line in open(f):
            r = json.loads(line)
            think[(r["id"], r["k"])] = r["text"].split("</think>", 1)[0]
    by = {}
    for f in a.branches:
        for line in open(f):
            r = json.loads(line)
            if (r["id"], r["tk"]) in think:
                by.setdefault((r["id"], r["tk"], r["cut"]), []).append(r)
    work = []
    for key, rs in by.items():
        rw = [float(r["correct"]) for r in rs]
        if len(rs) < 2 or min(rw) == max(rw):
            continue
        m = sum(rw)
        work.append(("pg", key, rs, [x - (m - x) / (len(rw) - 1) for x in rw]))
    npg = len(work)
    anc = [json.loads(l) for l in open(a.anchors)]
    random.Random(a.seed).shuffle(anc)
    for r in anc[: a.n_anchors]:
        th = r["text"].split("</think>", 1)[0]
        c = th.rfind("\n", 0, int(len(th) * random.Random(zlib.crc32(r["id"].encode())).uniform(0.1, 0.8))) + 1
        think[(r["id"], r["k"])] = th
        work.append(("kl", (r["id"], r["k"], c), None, None))
    print(f"[fit] {npg} sibling groups, {len(work) - npg} anchors, {len(S.comps)} writers x {a.k}", flush=True)
    opt = torch.optim.AdamW(S.params(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(work) // a.accum + 1
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps))

    def prefix(key):
        ids = prompt_ids(tok, items[key[0]]) + tok(think[key[:2]][: key[2]], add_special_tokens=False).input_ids
        with torch.no_grad():
            S.on = False
            o = owner(input_ids=torch.tensor([ids], device=DEV), use_cache=True)
        return o.past_key_values, o.last_hidden_state[0, -1:]

    def logits(cache, last_h, y):
        o = owner(input_ids=torch.tensor([y[:-1]], device=DEV), past_key_values=copy.deepcopy(cache), use_cache=True)
        return head(torch.cat([last_h, o.last_hidden_state[0]], 0)).float() / a.temperature

    t0, acc = time.time(), {"pg": [0.0, 0], "kl": [0.0, 0]}
    for ep in range(a.epochs):
        random.Random(a.seed + 7 + ep).shuffle(work)
        for j, (kind, key, rs, adv) in enumerate(work):
            cache, last_h = prefix(key)
            if kind == "pg":
                for r, A in zip(rs, adv):
                    y = tok(r["cont"], add_special_tokens=False).input_ids[: a.credit_tokens]
                    if len(y) < 2 or A == 0:
                        continue
                    yt = torch.tensor(y, device=DEV)
                    with torch.no_grad():
                        S.on = False
                        lg0 = logits(cache, last_h, y)
                        lp0 = -Fn.cross_entropy(lg0, yt, reduction="none")
                        q = torch.log_softmax(lg0, -1)
                        ent = -(q.exp() * q).sum(-1)
                        mask = (ent >= torch.quantile(ent, 1 - a.fork_frac)).float() if a.fork_frac < 1 else torch.ones_like(ent)
                    S.on = True
                    lp = -Fn.cross_entropy(logits(cache, last_h, y), yt, reduction="none")
                    rho = torch.exp(lp - lp0)
                    sur = torch.minimum(rho * A, torch.clamp(rho, 1 - a.clip, 1 + a.clip) * A)
                    loss = -(sur * mask).sum() / mask.sum().clamp(min=1) / len(rs)
                    loss.backward()
                    acc["pg"][0] += loss.item(); acc["pg"][1] += 1
            else:
                y = tok(think[key[:2]][key[2]:], add_special_tokens=False).input_ids[: a.credit_tokens]
                if len(y) >= 2:
                    with torch.no_grad():
                        S.on = False
                        lq = torch.log_softmax(logits(cache, last_h, y), -1)
                    S.on = True
                    lp = torch.log_softmax(logits(cache, last_h, y), -1)
                    kl = (lq.exp() * (lq - lp)).sum(-1).mean()
                    (a.kl * kl).backward()
                    acc["kl"][0] += kl.item(); acc["kl"][1] += 1
            S.on = False
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(S.params(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (j + 1) % 200 == 0:
                zr = torch.cat([z.detach() for z in S.params()])
                print(f"[fit] ep {ep} {j + 1}/{len(work)} pg {acc['pg'][0] / max(acc['pg'][1], 1):+.4f} "
                      f"kl {acc['kl'][0] / max(acc['kl'][1], 1):.5f} z rms {zr.pow(2).mean().sqrt():.4f} "
                      f"max {zr.abs().max():.4f} {time.time() - t0:.0f}s", flush=True)
                acc = {"pg": [0.0, 0], "kl": [0.0, 0]}
    torch.save({"comps": {n: (U.cpu().half(), z.detach().cpu()) for n, (U, z) in S.comps.items()}, "args": vars(a)},
               a.out)
    print(f"[fit] done {time.time() - t0:.0f}s -> {a.out}", flush=True)


def cmd_export(a):
    from wsp import case as C
    from wsp.edits import decoder_prefix, tensor_index
    from wsp import profiles
    fits = [torch.load(f) for f in a.fits]
    cdir = C.cdir(a.case)
    conf = C.conf(a.case)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(conf["profile"])))
    zbar = {}
    for F in fits:
        for n, (U, z) in F["comps"].items():
            zbar.setdefault(n, [U, 0.0])
            zbar[n][1] = zbar[n][1] + z.numpy() / len(fits)
    arrays, ops = {}, []
    for n, (U, z) in sorted(zbar.items()):
        for i in range(len(z)):
            zi = float(z[i]) * a.dose
            if abs(zi) < a.min_change:
                continue
            key = f"u{len(arrays)}"
            arrays[key] = U[:, i].numpy().astype(np.float16)
            ops.append({"op": "dir_ablate", "tensor": f"{prefix}{n}", "u": f"{a.name}.npz#{key}",
                        "alpha": round(-zi, 5), "side": "out"})
    out = cdir / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": conf["profile"], "ops": ops}
    if conf.get("parent"):
        spec["parent"] = os.path.relpath(cdir / conf["parent"], out)
    json.dump(spec, open(out / f"{a.name}.json", "w"), indent=1)
    zs = np.concatenate([v[1] for v in zbar.values()])
    if len(fits) > 1:
        Z = [np.concatenate([F["comps"][n][1].numpy() for n in sorted(zbar)]) for F in fits]
        cc = np.corrcoef(Z)[np.triu_indices(len(Z), 1)]
        print(f"[export] fit z correlation: mean {cc.mean():.2f}, min {cc.min():.2f}")
    print(f"[export] {a.name}: {len(ops)} ops (dose {a.dose}, z rms {np.sqrt((zs ** 2).mean()) * a.dose:.4f}) "
          f"-> {out / a.name}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "export"])
    ap.add_argument("--model"); ap.add_argument("--branches", nargs="*", default=[])
    ap.add_argument("--traces", nargs="*", default=[]); ap.add_argument("--anchors"); ap.add_argument("--out")
    ap.add_argument("--layers", default="4-21"); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--credit-tokens", type=int, default=384); ap.add_argument("--fork-frac", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--n-anchors", type=int, default=300); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--accum", type=int, default=8); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case"); ap.add_argument("--name"); ap.add_argument("--fits", nargs="*", default=[])
    ap.add_argument("--dose", type=float, default=2.0); ap.add_argument("--min-change", type=float, default=0.002)
    a = ap.parse_args()
    {"fit": cmd_fit, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
