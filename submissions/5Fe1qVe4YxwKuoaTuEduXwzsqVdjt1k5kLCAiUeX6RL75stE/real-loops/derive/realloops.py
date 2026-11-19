"""Real loops: the loops the model actually falls into, as the contrast for a second loop removal (see ../METHOD.md).

python derive/realloops.py <command> <case> ...   (the case is a wsp.case case stacked on the parent; every step
reads and writes $WSP_WORK/cases/<case>/realloops/; edits go to the case's edits/ like any other edit)

Main path (METHOD.md):
  gen         a toolkit run (wsp.case run) split over several GPUs: shards of a set, merged into the same run file
  label       the starting model's own traces -> in-loop positions (the repeated suffix of a cut thinking) and normal
              positions (traces scored right); fit / held-out split by problem
  collect     teacher-forced statistics of the residual state and MLP units in loops and in normal traces
  directions  the real-loop direction per layer and its MLP writers, in the toolkit's directions.npz format, for
              `python -m wsp.case edit` (dir_ablate on the top writers)
  check       teacher-forced selectivity of any edit on held-out traces: log-prob change of the loops' own continuations
              and of whole correct traces against the starting model (for gate edits also: fires, P(</think>))
  merge       one spec from stacked edits (a further round derived on an earlier round's model), on the published parent
  subset      a set file: one split restricted to some practice sources (the decision set of long problems)

Other contrasts, for diagnosis (not the main path; see METHOD.md "What does not work"):
  escape      matched on-policy pairs at real loop states, in the toolkit's pairs format (wsp.case extract / selectivity):
              the model's own loop continuation vs its own continuation with the loop's next token blocked
  restack     a direction orthogonal to the parent's directions in the layers it edited, its units left out of the
              writers (rewrites the case's directions.npz; the original is kept)

Measured not to help, kept for diagnosis (METHOD.md "What does not work"):
  scan / refine / choose / build / replay   the loop-exit gate: an LDA read direction and threshold per layer, a gate
              edit (wsp.gate unit, or a relay from an early detector to a late unit), and where it fired in generation
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import wsp  # noqa: F401
except ImportError:  # the toolkit checkout next to the submission, or $WSP_REPO
    for cand in (os.environ.get("WSP_REPO"), "/workspace/WeightSpaceProgramming",
                 str(Path(__file__).resolve().parents[4])):
        if cand and (Path(cand) / "wsp").is_dir():
            sys.path.insert(0, cand)
            break
from wsp import case as C  # noqa: E402
from wsp import data, profiles  # noqa: E402
from wsp.common import read_json, read_jsonl, stable_seed, write_json, write_jsonl  # noqa: E402

THINK_END = "</think>"


def sdir(case):
    d = C.scratch(case) / "realloops"
    d.mkdir(parents=True, exist_ok=True)
    return d


def heldout(pid, frac):
    return stable_seed(pid, "realloops-split") % 1000 < frac * 1000


# ----------------------------------------------------------------------------------------------------- gen
def cmd_gen(a):
    """Run `model` on `set` with one process per GPU, then write the toolkit's run file and results."""
    name, _ = C.split_model(a.model)
    C.model_path(a.case, name)  # build the edit once, before the shards start
    tag, items = C.set_items(a.case, a.set)
    out = C.run_path(a.case, a.model, tag)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (--force to regenerate)")
    gpus = a.gpus or [str(i) for i in range(a.shards)]
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append(subprocess.Popen([sys.executable, __file__, "_shard", a.case, a.model, a.set, str(i), str(len(gpus)),
                                       str(a.n)], env=env))
    if any(p.wait() for p in procs):
        raise SystemExit("a shard failed")
    order = {it["id"]: j for j, it in enumerate(items)}
    rows = []
    for i in range(len(gpus)):
        part = out.with_name(out.name + f".part{i}")
        rows += read_jsonl(part)
        part.unlink()
    rows.sort(key=lambda r: (order[r["id"]], r["k"]))
    write_jsonl(out, rows)
    res = C._write_result(a.case, a.model, tag)
    if a.model == C.BASE:
        for p in (C.RUNS / a.case).glob(f"*__{tag}.jsonl"):
            if not p.name.startswith(C.BASE + "__"):
                C._write_result(a.case, p.name.split("__")[0], tag)
    print(json.dumps(res["paired"] or res["summary"], indent=1))


def cmd_shard(a):
    from wsp.gen import generate  # noqa: PLC0415
    prof = profiles.get(C.conf(a.case)["profile"])
    tag, items = C.set_items(a.case, a.set)
    name, sampling = C.split_model(a.model)
    out = C.run_path(a.case, a.model, tag)
    generate(C.model_path(a.case, name), items[a.i::a.of], prof, n=a.n, out=out.with_name(out.name + f".part{a.i}"),
             sampling=sampling)


# ----------------------------------------------------------------------------------------------------- label
def ngram_counts(ids, n):
    """cnt[t] = how many times the n-gram ending at t occurred before t (0 for t < n - 1)."""
    seen, cnt = {}, np.zeros(len(ids), np.int32)
    for t in range(n - 1, len(ids)):
        g = tuple(ids[t - n + 1:t + 1])
        c = seen.get(g, 0)
        cnt[t], seen[g] = c, c + 1
    return cnt


def terminal_loop(ids, n=8, win=256, frac=0.7):
    """Start of the repeated suffix of a token list (every window of `win` tokens in it has >= frac positions whose
    n-gram occurred before), or None. -> (onset, cnt)"""
    cnt = ngram_counts(ids, n)
    if len(ids) < 2 * win:
        return None, cnt
    rep = (cnt >= 1).astype(np.float32)
    m = np.convolve(rep, np.ones(win) / win, mode="valid")  # m[j] = mean of rep[j:j+win]
    if m[-1] < frac:
        return None, cnt
    bad = np.flatnonzero(m < frac)
    onset = int(bad[-1] + win) if len(bad) else 0  # first position after the last non-repeating window
    return onset, cnt


def cmd_label(a):
    from transformers import AutoTokenizer  # noqa: PLC0415
    prof = profiles.get(C.conf(a.case)["profile"])
    model = C.model_path(a.case, C.BASE)
    tok = AutoTokenizer.from_pretrained(model)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    items = data.by_id()
    # tokens that start a new paragraph: the positions before them are where a normal reasoning step ends ("break")
    para = np.zeros(len(tok), bool)
    for s, j in tok.get_vocab().items():
        if j < len(para) and "\n\n" in tok.convert_tokens_to_string([s]):
            para[j] = True
    rng = np.random.default_rng(0)
    rows = [r for s in a.sets for r in read_jsonl(C.run_path(a.case, C.BASE, s)) if r["id"] in items]
    meta, arrays, kinds = [], {}, {}
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        g_ids = tok(r["text"], add_special_tokens=False).input_ids
        if len(p_ids) + len(g_ids) > a.max_tokens:
            g_ids = g_ids[: a.max_tokens - len(p_ids)]
        ends = [i for i, t in enumerate(g_ids) if t == end_id]
        think_end = ends[0] if ends else None
        n_p, T = len(p_ids), len(p_ids) + len(g_ids)
        pos, neg, close = np.zeros(T, bool), np.zeros(T, bool), -1
        kind = "finished" if not r["cut"] else "cut_other"
        onset = None
        # normal (negative) positions: where a fire costs something. "correct": traces scored right (a fire could
        # lose them); "finished": also finished wrong ones; "all": every non-loop position (also the doomed parts
        # of traces that end cut, where an exit can only help)
        keep = {"correct": r["correct"], "finished": r["correct"] or not r["cut"], "all": True}[a.neg_from]
        if r["cut"]:
            onset, cnt = terminal_loop(g_ids, a.ngram, a.window, a.frac)
            if onset is not None and len(g_ids) - onset >= a.min_loop:
                stop = think_end if think_end is not None else len(g_ids)
                if onset < stop:
                    kind = "loop"
                    sel = np.zeros(len(g_ids), bool)
                    sel[onset:stop] = cnt[onset:stop] >= a.min_prev  # from the (min_prev + 1)-th occurrence on
                    pos[n_p:] = sel
                    if a.neg_from == "all":
                        neg[: n_p + max(onset - a.guard, 0)] = True
                else:
                    kind = "answer_loop"  # loops after the thinking closed: not a target; its thinking is normal
                    neg[: n_p + think_end] = keep
            else:
                onset = None
                neg[:] = keep
        else:
            neg[:] = keep
            if think_end is not None and think_end > 0:
                close = n_p + think_end - 1  # the position whose next token is </think>
        brk = np.zeros(0, np.int64)
        if kind == "finished" and think_end is not None:  # step boundaries inside a normal thinking
            g = np.asarray(g_ids[:think_end])
            b = np.flatnonzero(para[np.clip(g[1:], 0, len(para) - 1)]) + n_p  # position t whose next token starts one
            brk = np.sort(rng.choice(b, min(32, len(b)), replace=False)) if len(b) else brk
        i = len(meta)
        arrays[f"brk{i}"] = brk
        arrays[f"ids{i}"] = np.asarray(p_ids + g_ids, np.int32)
        arrays[f"pos{i}"], arrays[f"neg{i}"] = np.packbits(pos), np.packbits(neg)
        meta.append({"i": i, "id": r["id"], "k": r["k"], "kind": kind, "task": items[r["id"]].get("task", "math"),
                     "source": r.get("source"), "correct": r["correct"], "T": T, "n_prompt": n_p,
                     "onset": None if onset is None else n_p + onset, "think_end": None if think_end is None else n_p + think_end,
                     "close": close, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                     "heldout": heldout(r["id"], a.heldout_frac)})
        kinds[kind] = kinds.get(kind, 0) + 1
    np.savez_compressed(sdir(a.case) / "labels.npz", **arrays)
    write_json(sdir(a.case) / "labels.json", meta)
    loops = [m for m in meta if m["kind"] == "loop"]
    print(f"{len(meta)} traces: {kinds}; loop traces fit / held-out: "
          f"{sum(not m['heldout'] for m in loops)} / {sum(m['heldout'] for m in loops)}; "
          f"in-loop tokens {sum(m['n_pos'] for m in loops)}; median onset at "
          f"{int(np.median([m['onset'] - m['n_prompt'] for m in loops])) if loops else None} generated tokens")


def load_labels(case):
    meta = read_json(sdir(case) / "labels.json")
    z = np.load(sdir(case) / "labels.npz")
    def get(m, brk=False):
        T = m["T"]
        out = (z[f"ids{m['i']}"], np.unpackbits(z[f"pos{m['i']}"])[:T].astype(bool),
               np.unpackbits(z[f"neg{m['i']}"])[:T].astype(bool))
        return out + (z[f"brk{m['i']}"],) if brk else out
    return meta, get


# ----------------------------------------------------------------------------------------------------- forward
class Taps:
    """Hooks on every decoder layer: MLP input x (pre-hook of the MLP), residual after the layer, down_proj input
    (unit activations). `fn(layer, kind, tensor[T, dim])` is called inside the forward pass."""

    def __init__(self, model, fn, kinds=("x", "resid", "act")):
        from wsp.probe import _child, decoder  # noqa: PLC0415
        self.owner, self.layers = decoder(model)
        self.hooks = []
        for li, layer in enumerate(self.layers):
            mlp = _child(layer, "mlp")
            if "x" in kinds:
                self.hooks.append(mlp.register_forward_pre_hook(lambda m, i, li=li: fn(li, "x", i[0][0])))
            if "act" in kinds:
                self.hooks.append(_child(mlp, "down_proj").register_forward_pre_hook(
                    lambda m, i, li=li: fn(li, "act", i[0][0])))
            if "resid" in kinds:
                self.hooks.append(layer.register_forward_hook(
                    lambda m, i, o, li=li: fn(li, "resid", (o[0] if isinstance(o, tuple) else o)[0])))

    def remove(self):
        for h in self.hooks:
            h.remove()


def forward(model, ids, device):
    import torch  # noqa: PLC0415
    with torch.no_grad():
        model(input_ids=torch.as_tensor(ids, dtype=torch.long, device=device)[None], logits_to_keep=1, use_cache=False)


# ----------------------------------------------------------------------------------------------------- collect
def cmd_collect(a):
    import torch  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415
    meta, get = load_labels(a.case)
    model, _ = load_model(C.model_path(a.case, C.BASE), a.device)
    fit = [m for m in meta if not m["heldout"] and m["kind"] in ("loop", "finished", "cut_other", "answer_loop")]
    L = len(Taps(model, lambda *x: None, ()).layers)
    D = model.config.get_text_config().hidden_size
    f64 = dict(dtype=torch.float64, device=a.device)
    acc = {k: torch.zeros(L, D, **f64) for k in ("sp", "sn", "rp", "rc", "rb", "rn")}
    acc.update({k: torch.zeros(L, D, D, **f64) for k in ("ssp", "ssn")})
    usage, acts = None, {}
    cnt = {"np": 0, "nn": 0, "nc": 0, "nb": 0}
    samples = {"pos": [[] for _ in range(L)], "neg": [[] for _ in range(L)]}
    rng = np.random.default_rng(0)
    cur = {}

    def fn(li, kind, t):
        nonlocal usage
        pos, neg, close, pick_p, pick_n = cur["pos"], cur["neg"], cur["close"], cur["pick_p"], cur["pick_n"]
        if kind == "x":
            x = t.float()
            if pos.any():
                xp = x[pos].double()
                acc["sp"][li] += xp.sum(0)
                acc["ssp"][li] += xp.T @ xp
                samples["pos"][li].append(x[pick_p].half().cpu().numpy())
            xn = x[neg]
            acc["sn"][li] += xn.double().sum(0)
            acc["ssn"][li] += (xn.T @ xn).double()
            samples["neg"][li].append(x[pick_n].half().cpu().numpy())
        elif kind == "resid":
            r = t.float()
            acc["rn"][li] += r[neg].double().sum(0)
            if pos.any():
                acc["rp"][li] += r[pos].double().sum(0)
            if close is not None:
                acc["rc"][li] += r[close].double()
            if len(cur["brk"]):
                acc["rb"][li] += r[cur["brk"]].double().sum(0)
        elif kind == "act":
            at = t.float()
            u = at[neg].abs().sum(0).double()
            if usage is None:
                usage = torch.zeros(L, u.shape[0], **f64)
                acts["an"] = torch.zeros(L, u.shape[0], **f64)
                acts["ap"] = torch.zeros(L, u.shape[0], **f64)
            usage[li] += u
            acts["an"][li] += at[neg].sum(0).double()
            if pos.any():
                acts["ap"][li] += at[pos].sum(0).double()

    taps = Taps(model, fn)
    try:
        for j, m in enumerate(fit):
            ids, pos, neg, brk = get(m, brk=True)
            pi, ni = np.flatnonzero(pos), np.flatnonzero(neg)
            cur.update(brk=torch.as_tensor(brk, dtype=torch.long, device=a.device), pos=torch.as_tensor(pos, device=a.device), neg=torch.as_tensor(neg, device=a.device),
                       close=m["close"] if m["close"] >= 0 else None,
                       pick_p=torch.as_tensor(rng.choice(pi, min(a.rows, len(pi)), replace=False) if len(pi) else pi,
                                              device=a.device),
                       pick_n=torch.as_tensor(rng.choice(ni, min(a.rows, len(ni)), replace=False), device=a.device))
            forward(model, ids, a.device)
            cnt["np"] += int(pos.sum())
            cnt["nn"] += int(neg.sum())
            cnt["nc"] += int(m["close"] >= 0)
            cnt["nb"] += len(brk)
            if j % 20 == 0:
                print(f"[collect] {j + 1}/{len(fit)} traces", flush=True)
    finally:
        taps.remove()
    out = {k: v.cpu().numpy() for k, v in acc.items()}
    out.update(usage=usage.cpu().numpy(), **{k: v.cpu().numpy() for k, v in acts.items()},
               **{k: np.asarray(v) for k, v in cnt.items()})
    for k in ("pos", "neg"):
        out[f"rows_{k}"] = np.stack([np.concatenate(s) if s else np.zeros((0, D), np.float16) for s in samples[k]])
    np.savez(sdir(a.case) / "stats.npz", **out)
    print(f"in-loop positions {cnt['np']}, normal positions {cnt['nn']}, close states {cnt['nc']} "
          f"({len(fit)} fit traces)")


# ----------------------------------------------------------------------------------------------------- scan
def directions(st, shrink):
    """Per layer LDA read direction (in-loop minus normal, pooled within-class covariance, ridge)."""
    L = st["sp"].shape[0]
    out = []
    for li in range(L):
        mp, mn = st["sp"][li] / st["np"], st["sn"][li] / st["nn"]
        S = (st["ssp"][li] - st["np"] * np.outer(mp, mp) + st["ssn"][li] - st["nn"] * np.outer(mn, mn)) / (st["np"] + st["nn"])
        d = np.linalg.solve(S + shrink * np.trace(S) / len(S) * np.eye(len(S)), mp - mn)
        out.append(d / np.linalg.norm(d))
    return np.asarray(out, np.float32)


def tagged(case, stem, tag, ext):
    return sdir(case) / f"{stem}{'_' + tag if tag else ''}{ext}"


def scan_scores(case, Dirs, device):
    """{f"s{i}": [T, L] scores x . d_l} for every labelled trace."""
    import torch  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415
    meta, get = load_labels(case)
    model, _ = load_model(C.model_path(case, C.BASE), device)
    Dt = torch.as_tensor(Dirs, device=device)
    cur, out = {}, {}

    def fn(li, kind, t):
        cur[li] = (t.float() @ Dt[li]).cpu().numpy()

    taps = Taps(model, fn, ("x",))
    try:
        for j, m in enumerate(meta):
            ids, _, _ = get(m)
            forward(model, ids, device)
            out[f"s{m['i']}"] = np.stack([cur[li] for li in range(len(Dirs))], 1).astype(np.float32)
            if j % 20 == 0:
                print(f"[scan] {j + 1}/{len(meta)} traces", flush=True)
    finally:
        taps.remove()
    return out


def cmd_scan(a):
    if a.dirs_tag is None:
        st = dict(np.load(sdir(a.case) / "stats.npz"))
        Dirs = directions(st, a.shrink)
        np.save(tagged(a.case, "dirs", a.tag, ".npy"), Dirs)
    else:
        Dirs = np.load(tagged(a.case, "dirs", a.dirs_tag, ".npy"))
    out = scan_scores(a.case, Dirs, a.device)
    np.savez_compressed(tagged(a.case, "scores", a.tag or a.dirs_tag, ".npz"), **out)
    print(f"scores of {len(out)} traces x {len(Dirs)} layers")


def cmd_directions(a):
    """The real-loop direction per layer in the toolkit's format (scratch directions.npz, keys U and writer), so that
    `python -m wsp.case edit <case> --layers ... --topk ... --alpha ...` ablates it from its strongest MLP writers:
    U_l = normalized mean residual after layer l at in-loop positions minus at normal positions; writer score of unit j
    = (mean activation in loop - mean normal) x (U_l . W_down[:, j]), as wsp.directions.extract computes from pairs."""
    from wsp.probe import _child, decoder, load_model  # noqa: PLC0415
    st = dict(np.load(sdir(a.case) / "stats.npz"))
    rp, rn = st["rp"] / st["np"], st["rn"] / st["nn"]
    U = (rp - rn) / np.linalg.norm(rp - rn, axis=1, keepdims=True)
    model, _ = load_model(C.model_path(a.case, C.BASE), "cpu")
    _, layers = decoder(model)
    da = st["ap"] / st["np"] - st["an"] / st["nn"]
    writer = np.stack([da[li] * (U[li] @ _child(layers[li], "down_proj").weight.detach().float().numpy())
                       for li in range(len(U))])
    out = C.scratch(a.case) / "directions.npz"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (--force replaces it)")
    np.savez_compressed(out, U=U.astype(np.float32), writer=writer.astype(np.float32))
    rows = []
    for li in range(len(U)):
        w = writer[li]
        top = np.sort(w)[::-1]
        rows.append({"layer": li, "relative_norm": float(np.linalg.norm(rp[li] - rn[li]) / np.linalg.norm(rn[li])),
                     "writer_total": float(w.sum()), "top64_share": float(top[:64].sum() / max(w[w > 0].sum(), 1e-9))})
        print(f"layer {li:2d}  |loop - normal| / |normal| {rows[-1]['relative_norm']:.3f}  writer total "
              f"{rows[-1]['writer_total']:9.3f}  top64 share {rows[-1]['top64_share']:.2f}")
    write_json(C.cdir(a.case) / "results" / "realloops_directions.json", rows)
    print(f"-> {out} (wsp.case edit uses it)")


def cmd_escape(a):
    """On-policy matched pairs at real loop states, in the toolkit's pairs format (scratch <out>.json and
    <out>_heldout.json, for `python -m wsp.case extract <case> --pairs <out>` and `selectivity --pairs <out>_heldout`):
    at a position inside a real loop (its in-loop positions), "bad" = the model's own next stretch of the loop, "good" =
    the model's own continuation of exactly the same prefix when the loop's next token is blocked (first token sampled
    with that token banned, the rest with the model card's sampling), kept only if it leaves the loop (few n-grams
    reused from the recent text). Same prefix, same model: the contrast is staying in the loop vs leaving it."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415
    prof = profiles.get(C.conf(a.case)["profile"])
    model = C.model_path(a.case, C.BASE)
    meta, get = load_labels(a.case)
    tok = AutoTokenizer.from_pretrained(model)
    rng = np.random.default_rng(a.seed)
    sites = []  # (meta index, position t, heldout)
    for m in meta:
        if m["kind"] != "loop":
            continue
        ids, pos, _ = get(m)
        cand = np.flatnonzero(pos[: len(ids) - a.length - 1])
        if not len(cand):
            continue
        picks = sorted(rng.choice(cand, min(a.per_trace * 8, len(cand)), replace=False))
        chosen = []
        for t in picks:  # spread over the loop
            if all(abs(t - c) >= a.spacing for c in chosen):
                chosen.append(int(t))
            if len(chosen) >= a.per_trace:
                break
        sites += [(m["i"], t, m["heldout"]) for t in chosen]
    print(f"{len(sites)} loop positions from {len({s_[0] for s_ in sites})} traces", flush=True)
    samp = dict(prof["sampling"])
    samp.pop("max_tokens", None)
    longest = max(get(meta[i])[0][:t].shape[0] for i, t, _ in sites) + a.length + 64
    llm = LLM(model=str(model), dtype="bfloat16", max_model_len=longest, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    prefixes = [get(meta[i])[0][:t].tolist() for i, t, _ in sites]
    first = llm.generate([TokensPrompt(prompt_token_ids=p_) for p_ in prefixes],
                         [SamplingParams(max_tokens=1, logit_bias={int(get(meta[i])[0][t]): -100.0},
                                         seed=stable_seed(i, t, "esc1"), **samp) for i, t, _ in sites])
    e1 = [list(o.outputs[0].token_ids) for o in first]
    rest = llm.generate([TokensPrompt(prompt_token_ids=p_ + e) for p_, e in zip(prefixes, e1)],
                        [SamplingParams(max_tokens=a.length - 1, seed=stable_seed(i, t, "esc2"), **samp)
                         for i, t, _ in sites])
    pairs = {False: [], True: []}
    dropped = 0
    for (i, t, ho), p_, e, r in zip(sites, prefixes, e1, rest):
        ids = get(meta[i])[0]
        good_ids = e + list(r.outputs[0].token_ids)
        bad_ids = ids[t:t + a.length].tolist()
        recent = p_[-a.window:]
        grams = {tuple(recent[j:j + a.ngram]) for j in range(len(recent) - a.ngram + 1)}
        g = [tuple(good_ids[j:j + a.ngram]) for j in range(len(good_ids) - a.ngram + 1)]
        reuse = sum(x in grams for x in g) / max(len(g), 1)
        if reuse > a.max_reuse:  # the blocked continuation went straight back into the loop: not an escape
            dropped += 1
            continue
        base = tok.decode(p_)
        good, bad = tok.decode(good_ids), tok.decode(bad_ids)
        mlen = min(len(good), len(bad))
        if mlen < 40:
            dropped += 1
            continue
        seg = [len(base), len(base) + mlen]
        pairs[ho].append({"id": meta[i]["id"], "pos": t, "reuse": round(reuse, 3),
                          "good": {"text": base + good[:mlen], "seg": seg}, "bad": {"text": base + bad[:mlen], "seg": seg}})
    write_json(C.scratch(a.case) / f"{a.out}.json", pairs[False])
    write_json(C.scratch(a.case) / f"{a.out}_heldout.json", pairs[True])
    print(f"pairs: fit {len(pairs[False])}, held-out {len(pairs[True])}; dropped {dropped} (no escape or too short)")
    for q in (pairs[False] + pairs[True])[:3]:
        print("  BAD :", q["bad"]["text"][q["bad"]["seg"][0]:q["bad"]["seg"][1]][:160].replace("\n", " | "))
        print("  GOOD:", q["good"]["text"][q["good"]["seg"][0]:q["good"]["seg"][1]][:160].replace("\n", " | "))


def cmd_restack(a):
    """Make a second round's directions safe to stack on the parent (rewrites scratch directions.npz in place; the
    original is kept as directions_unstacked.npz).

    Measured on the parent model, a loop direction contains the parent's own correction: the units the parent edited
    now write against its loop direction, and they are active in loops, so the new direction comes out partly
    anti-aligned with the parent's. Ablating it from the same units removes that correction, and with alpha > 1 (a
    reflection: the unit's component along u becomes (1 - alpha) times what it was) overshoots it: the loops the parent
    removed come back. So, per layer the parent edited: the direction is orthogonalized against each parent direction,
    the writer scores are recomputed for it (the same formula as wsp.directions.extract: activation difference x
    write along the direction), and the parent's units are excluded from the writers."""
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415
    from wsp.probe import _child, decoder, load_model  # noqa: PLC0415
    c = C.conf(a.case)
    path = C.scratch(a.case) / "directions.npz"
    d = dict(np.load(path))
    U, writer = d["U"].astype(np.float64), d["writer"].astype(np.float64)
    parent = {}  # layer -> [(u, cols)]
    for spec, sd in load_chain(C.cdir(a.case) / c["parent"]) if c.get("parent") else []:
        for op in spec["ops"]:
            if op["op"] == "dir_ablate" and op.get("side", "out") == "out" and "down_proj" in op["tensor"]:
                li = int(op["tensor"].split("layers.")[1].split(".")[0])
                cols = _resolve(op.get("cols"), sd)
                parent.setdefault(li, []).append((np.asarray(_resolve(op["u"], sd), np.float64).ravel(),
                                                  None if cols is None else np.asarray(cols).ravel()))
    if not parent:
        raise SystemExit("the case has no parent dir_ablate ops on down_proj: nothing to restack against")
    model, _ = load_model(C.model_path(a.case, C.BASE), "cpu")
    _, layers = decoder(model)
    np.savez_compressed(C.scratch(a.case) / "directions_unstacked.npz", **d)
    for li, ops in sorted(parent.items()):
        W = _child(layers[li], "down_proj").weight.detach().float().numpy().astype(np.float64)
        old = U[li].copy()
        den = old @ W
        da = np.where(np.abs(den) > 1e-8, writer[li] / np.where(np.abs(den) > 1e-8, den, 1.0), 0.0)
        new = old.copy()
        for u, _ in ops:
            uh = u / np.linalg.norm(u)
            new -= (new @ uh) * uh
        new /= np.linalg.norm(new)
        w_new = da * (new @ W)
        excluded = 0
        for _, cols in ops:
            if cols is not None:
                w_new[cols] = -np.inf
                excluded += len(cols)
        U[li], writer[li] = new, w_new
        cosines = [float(old @ (u / np.linalg.norm(u))) for u, _ in ops]
        print(f"layer {li}: cos(new direction, parent) {cosines} -> 0; parent units excluded {excluded}; "
              f"cos(restacked, original) {float(new @ old):.3f}")
    np.savez_compressed(path, U=U.astype(np.float32), writer=writer.astype(np.float32))
    print(f"-> {path} (original in directions_unstacked.npz); now `python -m wsp.case edit` as usual")


# ----------------------------------------------------------------------------------------------------- refine
def cmd_refine(a):
    """Hard-negative refit of the read direction of some layers: the threshold must clear the most loop-like normal
    positions, which an LDA (fit on averages) does not optimise. Each round mines, per fit trace, the highest-scoring
    normal positions and random in-loop positions, then fits a class-balanced logistic direction (L2) on them plus the
    collect samples; scores are recomputed for the next round."""
    import torch  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415
    meta, get = load_labels(a.case)
    fit = [m for m in meta if not m["heldout"]]
    st = dict(np.load(sdir(a.case) / "stats.npz"))
    Dirs = np.load(tagged(a.case, "dirs", a.tag_in, ".npy")).copy()
    z = np.load(tagged(a.case, "scores", a.tag_in, ".npz"))
    S = {k: z[k] for k in z.files}
    model, _ = load_model(C.model_path(a.case, C.BASE), a.device)
    rng = np.random.default_rng(0)
    P = {li: [st["rows_pos"][li].astype(np.float32)] for li in a.layers}
    N = {li: [st["rows_neg"][li].astype(np.float32)] for li in a.layers}
    for rnd in range(a.rounds):
        want, cur = {}, {}
        for m in fit:
            _, pos, neg = get(m)
            s = S[f"s{m['i']}"]
            for li in a.layers:
                ni = np.flatnonzero(neg)
                hard = ni[np.argsort(-s[ni, li])[: a.hard]]
                pi = np.flatnonzero(pos)
                pp = rng.choice(pi, min(a.hard, len(pi)), replace=False) if len(pi) else pi
                want[(m["i"], li)] = (hard, pp)

        def fn(li, kind, t, mi=None):
            if li in a.layers:
                hard, pp = want[(cur["i"], li)]
                x = t.float()
                N[li].append(x[torch.as_tensor(hard, device=a.device)].cpu().numpy())
                if len(pp):
                    P[li].append(x[torch.as_tensor(pp, device=a.device)].cpu().numpy())

        taps = Taps(model, fn, ("x",))
        try:
            for m in fit:
                cur["i"] = m["i"]
                forward(model, get(m)[0], a.device)
        finally:
            taps.remove()
        for li in a.layers:
            Xp = torch.as_tensor(np.concatenate(P[li]), device=a.device)
            Xn = torch.as_tensor(np.concatenate(N[li]), device=a.device)
            mu = torch.cat([Xp, Xn]).mean(0)
            sd = torch.cat([Xp, Xn]).std(0) + 1e-6
            w = torch.as_tensor(Dirs[li], device=a.device).float() * sd  # start from the current direction
            w.requires_grad_(True)
            b = torch.zeros((), device=a.device, requires_grad=True)
            opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                zp = ((Xp - mu) / sd) @ w + b
                zn = ((Xn - mu) / sd) @ w + b
                loss = torch.nn.functional.softplus(-zp).mean() + torch.nn.functional.softplus(zn).mean() \
                    + a.l2 * (w ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            d = (w.detach() / sd).cpu().numpy().astype(np.float64)
            Dirs[li] = (d / np.linalg.norm(d)).astype(np.float32)
        new = scan_scores(a.case, Dirs, a.device)
        for k in S:
            S[k] = S[k].copy()
            S[k][:, a.layers] = new[k][:, a.layers]
        print(f"[refine] round {rnd + 1}: {sum(len(x) for x in N[a.layers[0]])} negatives, "
              f"{sum(len(x) for x in P[a.layers[0]])} positives per layer", flush=True)
    np.save(tagged(a.case, "dirs", a.tag, ".npy"), Dirs)
    np.savez_compressed(tagged(a.case, "scores", a.tag, ".npz"), **S)
    print(f"refined layers {a.layers} -> dirs_{a.tag}.npy, scores_{a.tag}.npz (choose/build with --tag {a.tag})")


# ----------------------------------------------------------------------------------------------------- choose
def layer_report(meta, get, S, li, margin):
    """Threshold = max normal score over the fit traces + margin * (std of normal scores) -> held-out statistics."""
    fit_neg, all_neg = [], []
    for m in meta:
        _, _, neg = get(m)
        s = S[f"s{m['i']}"][:, li]
        if not neg.any():
            continue
        if not m["heldout"]:
            fit_neg.append(s[neg].max())
        all_neg.append(s[neg][:: 50])
    sd = float(np.concatenate(all_neg).std())
    theta = float(max(fit_neg) + margin * sd)
    res = {"layer": li, "theta": theta, "neg_sd": sd}
    for split in ("fit", "heldout"):
        ms = [m for m in meta if m["heldout"] == (split == "heldout")]
        fires, fired_traces, n_neg, caught, delay, recall = 0, 0, 0, 0, [], []
        loops, other, other_fired = 0, 0, 0
        for m in ms:
            _, pos, neg = get(m)
            s = S[f"s{m['i']}"][:, li]
            f = int((s[neg] > theta).sum())
            fires += f
            fired_traces += f > 0
            n_neg += int(neg.sum())
            if m["kind"] != "loop" and not neg.any():  # traces where a fire costs nothing (wrong or cut anyway)
                other += 1
                other_fired += bool((s > theta).any())
            if m["kind"] == "loop":
                loops += 1
                above = np.flatnonzero((s > theta) & pos)
                recall.append(float((s[pos] > theta).mean()) if pos.any() else 0.0)
                if len(above):
                    caught += 1
                    delay.append(int(above[0] - m["onset"]))
        res[split] = {"normal_tokens": n_neg, "false_fires": fires, "traces_with_false_fire": fired_traces,
                      "loop_traces": loops, "loops_caught": caught, "free_traces": other, "free_fired": other_fired,
                      "token_recall": float(np.mean(recall)) if recall else None,
                      "median_delay": int(np.median(delay)) if delay else None}
    return res


def cmd_choose(a):
    meta, get = load_labels(a.case)
    z = np.load(tagged(a.case, "scores", a.tag, ".npz"))
    S = {k: z[k] for k in z.files}  # decompress once
    L = S[f"s{meta[0]['i']}"].shape[1]
    table = [layer_report(meta, get, S, li, a.margin) for li in (a.layers or range(L))]
    write_json(tagged(a.case, "choose", a.tag, f"_m{a.margin:g}.json"), table)
    print("layer  theta   held-out: false fires (traces) / normal tokens   loops caught   token recall   median delay"
          "   free traces fired")
    for r in table:
        h = r["heldout"]
        print(f"{r['layer']:5d} {r['theta']:7.2f}   {h['false_fires']:6d} ({h['traces_with_false_fire']:3d}) / "
              f"{h['normal_tokens']:9d}   {h['loops_caught']:3d}/{h['loop_traces']:<3d}        "
              f"{(h['token_recall'] or 0):.3f}        {str(h['median_delay']):>6s}   {h['free_fired']}/{h['free_traces']}")


# ----------------------------------------------------------------------------------------------------- build
def parent_units(case, layer):
    """Down_proj columns the parent chain edits at this layer (left alone)."""
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415
    c = C.conf(case)
    if not c.get("parent"):
        return set()
    out = set()
    for spec, d in load_chain(C.cdir(case) / c["parent"]):
        for op in spec["ops"]:
            if f"layers.{layer}.mlp." in op["tensor"]:
                for k in ("cols", "rows"):
                    if op.get(k) is not None:
                        out |= {int(x) for x in np.asarray(_resolve(op[k], d)).ravel()}
    return out


def relay_sensitivity(case, la, lb, u, d_lb, sel_by_trace, c0, device):
    """Raise of layer lb's read score (x . d_lb at its MLP input) when the vector c0 * u is added to the residual after
    layer la at the given positions (what the detector unit at la writes when it fires), teacher-forced on fit loop
    traces. -> (median raise per unit of c at this c0, scores at those positions without the write)."""
    import torch  # noqa: PLC0415
    from wsp.probe import _child, decoder, load_model  # noqa: PLC0415
    model, _ = load_model(C.model_path(case, C.BASE), device)
    _, layers = decoder(model)
    meta, get = load_labels(case)
    ut = torch.as_tensor(u, dtype=torch.float32, device=device)
    dt = torch.as_tensor(d_lb, dtype=torch.float32, device=device)
    cur = {}

    def add(mod, i, o):
        if cur.get("inject") is None:
            return None
        h = o[0] if isinstance(o, tuple) else o
        h = h.clone()
        h[0, cur["inject"]] += (c0 * ut).to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    def read(mod, i):
        cur["x"] = i[0][0].float()

    h1 = layers[la].register_forward_hook(add)
    h2 = _child(layers[lb], "mlp").register_forward_pre_hook(read)
    raise_, plain = [], []
    try:
        for mi, sel in sel_by_trace:
            ids = get(meta[mi])[0]
            selt = torch.as_tensor(sel, device=device)
            out = {}
            for inj in (None, selt):
                cur["inject"] = inj
                forward(model, ids, device)
                out[inj is None] = cur["x"][selt]
            s0, s1 = out[True] @ dt, out[False] @ dt
            raise_.append(((s1 - s0) / c0).cpu().numpy())
            plain.append(s0.cpu().numpy())
    finally:
        h1.remove()
        h2.remove()
    del model
    torch.cuda.empty_cache()
    return float(np.median(np.concatenate(raise_))), np.concatenate(plain)


def cmd_build(a):
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: PLC0415
    from wsp.gate import constant_direction, silu  # noqa: PLC0415

    c = C.conf(a.case)
    st = dict(np.load(sdir(a.case) / "stats.npz"))
    Dirs = np.load(tagged(a.case, "dirs", a.tag, ".npy"))
    base = profiles.base_dir(c["profile"])
    model_dir = Path(C.model_path(a.case, C.BASE))
    prefix = decoder_prefix(tensor_index(base))
    index = tensor_index(model_dir)

    def tensor(name):
        with safe_open(str(model_dir / index[name]), "pt") as f:
            return f.get_tensor(name).float().numpy()

    layers = list(a.layers) + ([a.relay_to] if a.relay_to is not None else [])
    if a.thetas and len(a.thetas) != len(layers):
        raise SystemExit("--thetas: one per layer (detector layers, then the --relay-to layer)")
    choose = {r["layer"]: r for r in read_json(tagged(a.case, "choose", a.tag, f"_m{a.margin:g}.json"))}
    thetas = dict(zip(layers, a.thetas)) if a.thetas else {li: choose[li]["theta"] for li in layers}
    meta, get = load_labels(a.case)
    z = np.load(tagged(a.case, "scores", a.tag, ".npz"))
    fit_loops = [m for m in meta if not m["heldout"] and m["kind"] == "loop"]
    S = {m["i"]: z[f"s{m['i']}"] for m in fit_loops}

    def target(li, kind):
        mu_loop = st["rp"][li] / max(int(st["np"]), 1)
        mu = {"close": st["rc"][li] / max(int(st["nc"]), 1), "break": st["rb"][li] / max(int(st["nb"]), 1)}
        w = a.scale * (mu.get(kind, mu["close"]) - mu_loop)
        if kind == "random":  # control: a norm-matched random direction orthogonal to the close write
            r = np.random.default_rng(li).standard_normal(len(w))
            r -= (r @ w) / (w @ w) * w
            w = r / np.linalg.norm(r) * np.linalg.norm(w)
        return w

    def unit(li, w):
        """One gate unit at layer li: reads Dirs[li] above thetas[li], writes ~w at a typical fire. -> (ops, report)"""
        theta, d = thetas[li], Dirs[li].astype(np.float64)
        P, N = st["rows_pos"][li].astype(np.float64), st["rows_neg"][li].astype(np.float64)
        m = constant_direction(np.concatenate([P, N]))
        t = f"{prefix}layers.{li}.mlp."
        # the unit to overwrite: the least used (mean |activation| x write norm on normal text), not edited by the parent
        Wd = tensor(t + "down_proj.weight")
        use = st["usage"][li] / st["nn"] * np.linalg.norm(Wd, axis=0)
        for uu in parent_units(a.case, li):
            use[uu] = np.inf
        uidx = int(np.argmin(use))
        # gate gain g: SiLU is not zero below its threshold (its negative lobe reaches -0.28), so a shallow gate leaks a
        # fraction of the write onto every normal position just under theta. Take the steepest gate the relative-change
        # budget of gate_proj allows (at most --g-max), or --g.
        Wg = tensor(t + "gate_proj.weight")
        g = a.g or a.g_max
        while not a.g and g > 4.0 and np.linalg.norm(g * (d - theta * m) - Wg[uidx]) / np.linalg.norm(Wg) > a.max_rel_gate:
            g *= 0.95
        gate_row, up_row = g * (d - theta * m), a.k * m
        # the write's size: w at the --norm-quantile of the unit's activation over the in-loop positions of the fit
        # traces that cross the threshold (scores of the scan; m . x ~ 1), larger deeper inside a loop
        sp = np.concatenate([S[mm["i"]][get(mm)[1], li] for mm in fit_loops])
        exc = sp[sp > theta] - theta
        if not len(exc):
            raise SystemExit(f"layer {li}: no in-loop position of the fit traces is above theta {theta:.3f}")
        a_ref = float(np.quantile(silu(g * exc) * a.k, a.norm_quantile))
        down_col = w / a_ref
        leak = float(np.abs(silu(N @ gate_row) * (N @ up_row)).max() / a_ref)
        # keep each tensor's relative change small: move magnitude from down_proj into up_proj (exact: linear in k)
        col_change = np.linalg.norm(down_col - Wd[:, uidx]) / np.linalg.norm(Wd)
        if col_change > a.max_rel:
            f = col_change / a.max_rel * 1.05
            up_row, down_col = up_row * f, down_col / f
        # values inline (float32, exact in JSON): wsp.edits.export / wsp.upload carry inline lists as they are, but
        # move only "u" / "cols" / "rows" npz references into the uploaded edit.npz
        f32 = lambda v: np.asarray(v, np.float32).tolist()  # noqa: E731
        ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [uidx], "values": f32(gate_row[None])},
               {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [uidx], "values": f32(up_row[None])},
               {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [uidx], "values": f32(down_col[:, None])}]
        rel = {}
        for nm, new, ax in (("gate_proj", gate_row, 0), ("up_proj", up_row, 0), ("down_proj", down_col, 1)):
            W = Wd if nm == "down_proj" else (Wg if nm == "gate_proj" else tensor(t + nm + ".weight"))
            old = W[uidx] if ax == 0 else W[:, uidx]
            rel[nm] = float(np.linalg.norm(new - old) / np.linalg.norm(W))
        return ops, {"layer": li, "unit": uidx, "theta": theta, "g": g, "max_leak_normal_rows": leak,
                     "fit_inloop_above_theta": f"{len(exc)}/{len(sp)}", "write_norm": float(np.linalg.norm(w)),
                     "relative_change": rel}

    ops, report = [], []
    if a.relay_to is None:
        for li in a.layers:
            o, r = unit(li, target(li, a.write))
            ops += o
            report.append(dict(r, write=a.write, scale=a.scale))
    else:
        # relay: detector units at --layers write a flag that lifts the --relay-to layer's own read score over its
        # threshold; the unit there (an ordinary exit unit) then fires at the same position and writes the target
        lb = a.relay_to
        o, r = unit(lb, target(lb, a.write))
        ops += o
        report.append(dict(r, write=a.write, scale=a.scale, role="actuator"))
        gamma = 1.0 + tensor(f"{prefix}layers.{lb}.post_attention_layernorm.weight")
        u = gamma * Dirs[lb]
        u /= np.linalg.norm(u)
        sd_b = choose[lb]["neg_sd"]
        for la in a.layers:
            sel = [(mm["i"], np.flatnonzero(S[mm["i"]][:, la] > thetas[la])[:256]) for mm in fit_loops]
            sel = [(i, s_) for i, s_ in sel if len(s_)][: a.relay_traces]
            # the flag's size, by measurement (the path through the layers between is far from linear): geometric
            # bisection for the smallest size whose median raise at the detector's fire positions lifts the median
            # score there over theta + margin
            c_try, lo, hi, hist = a.relay_c0, None, None, []
            for _ in range(a.relay_iters):
                sens, plain = relay_sensitivity(a.case, la, lb, u, Dirs[lb], sel, c_try, a.device)
                need = thetas[lb] + a.relay_margin * sd_b - float(np.median(plain))
                hist.append((round(c_try, 3), round(sens * c_try, 3), round(need, 3)))
                if sens * c_try >= need:
                    hi = c_try
                    c_try = c_try / 4 if lo is None else (lo * c_try) ** 0.5
                else:
                    lo = c_try
                    c_try = c_try * 4 if hi is None else (hi * c_try) ** 0.5
            if hi is None:
                raise SystemExit(f"relay {la}->{lb}: no flag size tried lifts layer {lb}'s score enough ({hist})")
            cvec = hi
            print(f"relay {la}->{lb} calibration (c, raise, needed): {hist} -> c = {cvec:.3f}", flush=True)
            o, r = unit(la, cvec * u)
            ops += o
            report.append(dict(r, role="detector", relay_to=lb, relay_calibration=hist, relay_norm=cvec,
                               lb_score_median_at_fires=float(np.median(plain)), lb_theta=thetas[lb]))
    edits = C.cdir(a.case) / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], edits)
    write_json(edits / f"{a.name}.json", spec)
    write_json(sdir(a.case) / f"build_{a.name}.json", report)
    n_t, payload = spec_size(edits / f"{a.name}.json")
    print(json.dumps(report, indent=1))
    print(f"{edits / (a.name + '.json')}: {n_t} tensors, payload {payload} bytes")


# ----------------------------------------------------------------------------------------------------- check
def cmd_check(a):
    import torch  # noqa: PLC0415
    from wsp.probe import _child, load_model  # noqa: PLC0415
    meta, get = load_labels(a.case)
    spec = read_json(C.cdir(a.case) / "edits" / f"{a.edit}.json")
    sites = [(int(op["tensor"].split("layers.")[1].split(".")[0]), int(op["rows"][0])) for op in spec["ops"]
             if op["op"] == "set_rows" and op["tensor"].endswith("gate_proj.weight")]
    held = [m for m in meta if m["heldout"]]
    rng = np.random.default_rng(1)
    normal = [m for m in held if m["kind"] in ("finished", "cut_other")]
    lp_sample = {m["i"] for m in rng.choice(normal, min(a.logp_traces, len(normal)), replace=False)}
    out = {}
    for tag, path in (("edit", C.model_path(a.case, a.edit)), ("start", C.model_path(a.case, C.BASE))):
        model, tok = load_model(path, a.device)
        end_id = tok.convert_tokens_to_ids(THINK_END)
        from wsp.probe import decoder  # noqa: PLC0415
        owner, layers = decoder(model)
        head = model.get_output_embeddings()
        cur = {}
        hooks = [_child(layers[li], "gate_proj").register_forward_hook(
            lambda mod, i, o, li=li, u=u: cur.__setitem__(li, o[0, :, u].float().cpu().numpy())) for li, u in sites]
        hooks.append(owner.norm.register_forward_hook(lambda mod, i, o: cur.__setitem__("h", o[0])))
        res = {"normal_traces": 0, "normal_tokens": 0, "false_fires": 0, "traces_with_false_fire": 0,
               "loop_traces": 0, "loops_fired": 0, "first_fire_delay": [], "p_end_first_fire": [],
               "p_end_fires_mean": [], "logp": {}, "loop_logp": {}}

        def mean_logp(h, ids, positions):
            """mean log-prob of the token after each given position"""
            tgt = torch.as_tensor(ids, dtype=torch.long, device=a.device)
            positions = positions[positions < len(ids) - 1]
            lps = []
            with torch.no_grad():
                for c in range(0, len(positions), 1024):
                    pp = torch.as_tensor(positions[c:c + 1024], device=a.device)
                    lps.append(torch.log_softmax(head(h[pp]).float(), -1).gather(1, tgt[pp + 1, None])[:, 0])
            return float(torch.cat(lps).mean())
        try:
            for m in held:
                if m["kind"] not in ("finished", "cut_other", "loop"):
                    continue
                ids, pos, neg = get(m)
                with torch.no_grad():
                    model(input_ids=torch.as_tensor(ids, dtype=torch.long, device=a.device)[None], logits_to_keep=1,
                          use_cache=False)
                fire = np.zeros(len(ids), bool)
                for li, _ in sites:
                    fire |= cur[li] > 0
                h = cur.pop("h")
                if m["kind"] == "loop":
                    res["loop_traces"] += 1
                    if pos.any():  # how likely the loop's own continuation is (an anti-loop edit lowers it)
                        res["loop_logp"][m["i"]] = mean_logp(h, ids, np.flatnonzero(pos))
                    f = np.flatnonzero(fire & pos)
                    if tag == "edit" and len(f):
                        res["loops_fired"] += 1
                        res["first_fire_delay"].append(int(f[0] - m["onset"]))
                        idx = torch.as_tensor(f[:256], device=a.device)
                        with torch.no_grad():
                            p = torch.softmax(head(h[idx]).float(), -1)[:, end_id].cpu().numpy()
                        res["p_end_first_fire"].append(float(p[0]))
                        res["p_end_fires_mean"].append(float(p.mean()))
                else:
                    res["normal_traces"] += 1
                    res["normal_tokens"] += int(neg.sum())
                    nf = int((fire & neg).sum())
                    res["false_fires"] += nf
                    res["traces_with_false_fire"] += nf > 0
                    if m["i"] in lp_sample:  # mean log-prob of the generated tokens (position t predicts t + 1)
                        res["logp"][m["i"]] = mean_logp(h, ids, np.arange(m["n_prompt"] - 1, len(ids) - 1))
        finally:
            for hk in hooks:
                hk.remove()
        del model
        torch.cuda.empty_cache()
        out[tag] = res
    e, s = out["edit"], out["start"]
    common = sorted(set(e["logp"]) & set(s["logp"]))
    lcommon = sorted(set(e["loop_logp"]) & set(s["loop_logp"]))
    summary = {"edit": a.edit, "sites": sites,
               "normal": {"traces": e["normal_traces"], "tokens": e["normal_tokens"], "false_fires": e["false_fires"],
                          "traces_with_false_fire": e["traces_with_false_fire"],
                          "logp_change": float(np.mean([e["logp"][i] - s["logp"][i] for i in common])) if common else None},
               "loops": {"traces": e["loop_traces"], "fired": e["loops_fired"],
                         "loop_logp_change": float(np.mean([e["loop_logp"][i] - s["loop_logp"][i] for i in lcommon]))
                         if lcommon else None,
                         "median_first_fire_delay": int(np.median(e["first_fire_delay"])) if e["first_fire_delay"] else None,
                         "p_end_first_fire": float(np.mean(e["p_end_first_fire"])) if e["p_end_first_fire"] else None,
                         "p_end_fires_mean": float(np.mean(e["p_end_fires_mean"])) if e["p_end_fires_mean"] else None}}
    write_json(C.cdir(a.case) / "results" / f"{a.edit}__realloops_check.json", summary)
    print(json.dumps(summary, indent=1))


def cmd_replay(a):
    """An edit's own generations, replayed teacher-forced through the edited model: where its units fired, whether the
    thinking closed right after a fire, and what became of each trace against the starting model's run of the same
    problems and seeds. Changing any weight changes every sampled trace, so a paired comparison mixes the gate's
    effect with sampling divergence; only traces where a unit fired show the gate's own effect."""
    import collections  # noqa: PLC0415
    from wsp.probe import _child, decoder, load_model  # noqa: PLC0415
    spec = read_json(C.cdir(a.case) / "edits" / f"{a.edit}.json")
    sites = [(int(op["tensor"].split("layers.")[1].split(".")[0]), int(op["rows"][0])) for op in spec["ops"]
             if op["op"] == "set_rows" and op["tensor"].endswith("gate_proj.weight")]
    tag, _ = C.set_items(a.case, a.set)
    edit_rows = read_jsonl(C.run_path(a.case, a.edit, tag))
    base_rows = C._base_run_for(a.case, {(r["id"], r["k"]) for r in edit_rows}, tag) or []
    base = {(r["id"], r["k"]): r for r in base_rows}
    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, a.edit), a.device)
    _, layers = decoder(model)
    cur = {}
    hooks = [_child(layers[li], "gate_proj").register_forward_hook(
        lambda m, i, o, li=li, u=u: cur.__setitem__(li, o[0, :, u].float().cpu().numpy())) for li, u in sites]
    end_id = tok.convert_tokens_to_ids(THINK_END)
    items = {it["id"]: it for it in C.set_items(a.case, a.set)[1]}
    out = []
    try:
        for r in edit_rows:
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
            p = tok(prompt, add_special_tokens=False).input_ids
            ids = (p + tok(r["text"], add_special_tokens=False).input_ids)[: a.max_tokens]
            forward(model, ids, a.device)
            fire = np.zeros(len(ids), bool)
            for li, _ in sites:
                fire |= cur[li] > 0
            f = np.flatnonzero(fire[len(p):])
            ends = [i - len(p) for i, t in enumerate(ids) if t == end_id and i >= len(p)]
            te = ends[0] if ends else None
            b = base.get((r["id"], r["k"]))
            out.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "base": b["kind"] if b else None,
                        "edit": r["kind"], "fires": int(len(f)), "first_fire": int(f[0]) if len(f) else None,
                        "think_end": te, "exit_at_fire": bool(te is not None and len(f) and (te - 1) in set(f.tolist())),
                        "fires_in_answer": int((f >= te).sum()) if te is not None else 0})
    finally:
        for h in hooks:
            h.remove()
    write_json(C.cdir(a.case) / "results" / f"{a.edit}__{tag}_replay.json", out)
    fired = [o for o in out if o["fires"]]
    exits = [o for o in fired if o["exit_at_fire"]]
    tr = collections.Counter((o["base"], o["edit"]) for o in exits)
    print(f"{len(out)} traces; a unit fired in {len(fired)}; the thinking closed right after a fire in {len(exits)}")
    print("  base kind of fired traces:", dict(collections.Counter(o["base"] for o in fired)))
    print("  exits (base kind -> edit kind):", {f"{k[0]}->{k[1]}": v for k, v in sorted(tr.items())})
    print("  exits correct:", sum(o["edit"] == "correct" for o in exits), "/", len(exits),
          "| exits whose answer then hit the limit:", sum(o["edit"].startswith("cut") for o in exits))
    nf = [o for o in out if not o["fires"] and o["base"]]
    print(f"  traces without fire: {len(nf)}; changed by sampling divergence: rescued "
          f"{sum(o['edit'] == 'correct' and o['base'] != 'correct' for o in nf)}, lost "
          f"{sum(o['base'] == 'correct' and o['edit'] != 'correct' for o in nf)}")
    print("  normal traces (base correct) with a fire:", sum(o["base"] == "correct" for o in fired))


def cmd_merge(a):
    """One spec from edits applied in turn (e.g. a second round derived on the first round's model): the ops of each, in
    order, on the parent of the first, written into <case>'s edits (the case whose starting model is that parent, so
    `wsp.case run` pairs it with it). An upload can only name the published parent, so a second round is delivered
    this way. The ops apply in sequence, so the merged spec rebuilds the stacked model (up to the rounding between
    links); the limits then count all of its ops against the parent: check with axonry-check."""
    from wsp.edits import _resolve, spec_size  # noqa: PLC0415
    out_dir = C.cdir(a.case) / "edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [Path(x).resolve() for x in a.specs]
    def shape(spec):  # what an op does, without where its arrays are stored (exports rename the npz references)
        return [(o["op"], o["tensor"], o.get("alpha"), o.get("side"), o.get("factor")) for o in spec["ops"]]

    for prev, nxt in zip(specs, specs[1:]):  # each later spec must stack on the one before it
        sp = read_json(nxt)
        if not sp.get("parent") or shape(read_json((nxt.parent / sp["parent"]).resolve())) != shape(read_json(prev)):
            raise SystemExit(f"{nxt.name} does not stack on {prev.name}")
    ops, arrays = [], {}
    for k, sp_path in enumerate(specs):
        for op in read_json(sp_path)["ops"]:
            op = dict(op)
            for key in ("u", "cols", "rows", "values"):
                ref = op.get(key)
                if isinstance(ref, str):
                    arr = np.asarray(_resolve(ref, sp_path.parent))
                    if key == "values":  # wsp.upload carries only u / cols / rows references: values go inline
                        op[key] = arr.astype(np.float32).tolist()
                    else:
                        name = f"s{k}_{key}_{len(arrays)}"
                        arrays[name] = arr
                        op[key] = f"{a.name}.npz#{name}"
            ops.append(op)
    first = read_json(specs[0])
    spec = {"name": a.name, "base": first["base"], "ops": ops}
    if first.get("parent"):
        spec["parent"] = os.path.relpath((specs[0].parent / first["parent"]).resolve(), out_dir)
    if arrays:
        np.savez(out_dir / f"{a.name}.npz", **arrays)
    write_json(out_dir / f"{a.name}.json", spec)
    n_t, payload = spec_size(out_dir / f"{a.name}.json")
    print(f"{out_dir / (a.name + '.json')}: {len(ops)} ops, {n_t} tensors, payload {payload} bytes, parent "
          f"{spec.get('parent')}")


def cmd_subset(a):
    """A set file of one case split restricted to some practice sources (e.g. the MATH problems of the holdout split),
    written as cases/<case>/sets/<name>.json, usable as a set name by wsp.case run and gen."""
    ids = [i for i in C.set_ids(a.case, a.split) if i.split("_")[0] in {s_.split("_")[0] for s_ in a.sources}]
    ids = ids[: a.n] if a.n else ids
    write_json(C.cdir(a.case) / "sets" / f"{a.name}.json", ids)
    print(f"{len(ids)} problems of split {a.split} from {a.sources} -> sets/{a.name}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="wsp.case run, one process per GPU")
    g.add_argument("case"), g.add_argument("model"), g.add_argument("set")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--shards", type=int, default=4)
    g.add_argument("--gpus", nargs="+", help="CUDA device ids (default 0..shards-1)")
    g.add_argument("--force", action="store_true")
    s = sub.add_parser("_shard")
    for k in ("case", "model", "set"):
        s.add_argument(k)
    for k in ("i", "of", "n"):
        s.add_argument(k, type=int)
    lb = sub.add_parser("label", help="in-loop and normal tokens of the starting model's runs")
    lb.add_argument("case")
    lb.add_argument("--sets", nargs="+", default=["dev"], help="base runs to label (wsp.case run <case> base <set>)")
    lb.add_argument("--ngram", type=int, default=8)
    lb.add_argument("--window", type=int, default=256)
    lb.add_argument("--frac", type=float, default=0.7, help="share of repeated n-grams in every window of the loop")
    lb.add_argument("--min-loop", type=int, default=600, help="tokens of repeated suffix to call a cut trace a loop")
    lb.add_argument("--min-prev", type=int, default=2, help="positives: the n-gram occurred at least this often before")
    lb.add_argument("--guard", type=int, default=512, help="--neg-from all: tokens before a loop's onset left out")
    lb.add_argument("--neg-from", choices=("correct", "finished", "all"), default="correct",
                    help="traces whose positions are normal (a fire there may cost an answer): correct ones (default), "
                         "all finished ones, or every non-loop position")
    lb.add_argument("--heldout-frac", type=float, default=0.3)
    lb.add_argument("--max-tokens", type=int, default=40000)
    for name in ("collect", "scan", "refine"):
        p = sub.add_parser(name)
        p.add_argument("case")
        p.add_argument("--device", default="cuda")
        if name == "collect":
            p.add_argument("--rows", type=int, default=48, help="raw MLP-input rows kept per trace and class")
        elif name == "scan":
            p.add_argument("--shrink", type=float, default=0.1)
            p.add_argument("--dirs-tag", help="score with dirs_<tag>.npy (e.g. refined) instead of fitting the LDA")
            p.add_argument("--tag", help="output names dirs_<tag> / scores_<tag>")
        else:
            p.add_argument("--layers", type=int, nargs="+", required=True)
            p.add_argument("--tag-in", help="start from dirs_<tag> / scores_<tag> (default: the LDA ones)")
            p.add_argument("--tag", default="refined")
            p.add_argument("--hard", type=int, default=32, help="hardest normal positions mined per fit trace")
            p.add_argument("--rounds", type=int, default=2)
            p.add_argument("--l2", type=float, default=1e-3)
    ch = sub.add_parser("choose")
    ch.add_argument("case")
    ch.add_argument("--margin", type=float, default=1.0, help="threshold = max fit normal score + margin x normal sd")
    ch.add_argument("--tag", help="use scores_<tag>.npz (e.g. refined)")
    ch.add_argument("--layers", type=int, nargs="+")
    b = sub.add_parser("build")
    b.add_argument("case")
    b.add_argument("--layers", type=int, nargs="+", required=True)
    b.add_argument("--thetas", type=float, nargs="+", help="thresholds (default: from choose --margin)")
    b.add_argument("--margin", type=float, default=1.0)
    b.add_argument("--tag", help="directions / thresholds from dirs_<tag>, choose_<tag>_m<margin>")
    b.add_argument("--scale", type=float, default=1.0, help="write = scale x (target state - loop state)")
    b.add_argument("--write", choices=("close", "break", "random"), default="close",
                   help="target state: where the model closes its thinking (close), where a normal reasoning step "
                        "ends and a new one starts (break); random: norm-matched random control")
    b.add_argument("--k", type=float, default=4.0)
    b.add_argument("--relay-to", type=int, help="relay: --layers are detectors whose units lift this layer's read "
                                                "score over its threshold; the unit here writes the target")
    b.add_argument("--relay-margin", type=float, default=2.0, help="lift to theta + this x normal sd at that layer")
    b.add_argument("--relay-c0", type=float, default=1.0, help="probe size of the flag for the calibration")
    b.add_argument("--relay-traces", type=int, default=12)
    b.add_argument("--relay-iters", type=int, default=6)
    b.add_argument("--device", default="cuda")
    b.add_argument("--g", type=float, help="gate gain (default: the steepest within --max-rel-gate, at most --g-max)")
    b.add_argument("--g-max", type=float, default=32.0)
    b.add_argument("--max-rel-gate", type=float, default=0.4, help="budget for gate_proj's relative change")
    b.add_argument("--norm-quantile", type=float, default=0.25,
                   help="the write equals scale x target at this quantile of the firing in-loop activations")
    b.add_argument("--max-rel", type=float, default=0.3, help="max relative change of down_proj from the unit")
    b.add_argument("--name", required=True)
    su = sub.add_parser("subset", help="a set file: one split restricted to some practice sources")
    su.add_argument("case")
    su.add_argument("--split", default="holdout")
    su.add_argument("--sources", nargs="+", required=True, help="practice sources, e.g. math_train")
    su.add_argument("--n", type=int, help="at most this many problems")
    su.add_argument("--name", required=True)
    rs = sub.add_parser("restack", help="second-round directions that do not undo the parent's edit")
    rs.add_argument("case")
    es = sub.add_parser("escape", help="on-policy matched pairs at real loop states (stay in the loop vs leave it)")
    es.add_argument("case")
    es.add_argument("--per-trace", type=int, default=4)
    es.add_argument("--spacing", type=int, default=400, help="min tokens between two positions of one trace")
    es.add_argument("--length", type=int, default=96, help="tokens of each continuation")
    es.add_argument("--ngram", type=int, default=8)
    es.add_argument("--window", type=int, default=2048, help="recent tokens an escape must not copy from")
    es.add_argument("--max-reuse", type=float, default=0.3, help="max share of the escape's n-grams seen recently")
    es.add_argument("--gpu-mem", type=float, default=0.85)
    es.add_argument("--seed", type=int, default=0)
    es.add_argument("--out", default="escpairs")
    mg = sub.add_parser("merge", help="one spec from stacked edits (deliver a second round on the published parent)")
    mg.add_argument("case", help="the case whose starting model is the first spec's parent")
    mg.add_argument("specs", nargs="+", help="edit specs in stacking order (each later one stacks on the one before)")
    mg.add_argument("--name", required=True)
    dr = sub.add_parser("directions", help="real-loop direction + writers as the toolkit's directions.npz")
    dr.add_argument("case")
    dr.add_argument("--force", action="store_true")
    rp = sub.add_parser("replay", help="where an edit's units fired in its own generations, and what came of it")
    rp.add_argument("case"), rp.add_argument("edit"), rp.add_argument("set")
    rp.add_argument("--device", default="cuda")
    rp.add_argument("--max-tokens", type=int, default=40000)
    k = sub.add_parser("check")
    k.add_argument("case"), k.add_argument("edit")
    k.add_argument("--device", default="cuda")
    k.add_argument("--logp-traces", type=int, default=24)
    a = ap.parse_args()
    {"gen": cmd_gen, "_shard": cmd_shard, "label": cmd_label, "collect": cmd_collect, "scan": cmd_scan,
     "refine": cmd_refine, "choose": cmd_choose, "build": cmd_build, "check": cmd_check, "replay": cmd_replay,
     "directions": cmd_directions, "merge": cmd_merge, "escape": cmd_escape,
     "restack": cmd_restack, "subset": cmd_subset}[a.cmd](a)


if __name__ == "__main__":
    main()
