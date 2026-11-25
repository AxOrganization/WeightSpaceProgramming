import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny_lm(tmp_path):
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   initializer_range=0.5)  # large weights: gains visibly change the outputs
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    d = tmp_path / "base"
    m.save_pretrained(d, safe_serialization=True)
    return m, d


def test_fit_lowers_the_loss_and_its_spec_builds_the_fitted_model(tmp_path, tiny_lm, monkeypatch):
    from wsp import case as C
    from wsp import fit, profiles
    from wsp.edits import build

    model, base = tiny_lm
    tgts = fit.targets(model, [0, 1], fit.DEFAULT_MODULES)
    assert {(t, w) for t, _, w in tgts} == {(f"layers.{l}.{n}.weight", w) for l in (0, 1)
                                          for n, w in (("mlp.down_proj", 1), ("self_attn.o_proj", 4))}
    rng = np.random.default_rng(0)
    ex = [{"kind": "nll", "w": 1.0, "ids": (rng.integers(0, 8, 6).tolist() * 4), "start": 6, "stop": 24}
          for _ in range(4)]  # repeats: learnable

    def mean_loss(vals=None):
        g = fit.Gains(tgts, "cpu")
        for t, (p, _) in g.params.items():
            if vals:
                p.data = torch.tensor(vals[t], dtype=torch.float32)
        with torch.no_grad():
            out = float(np.mean([float(-fit._Head(model).logp(e)[0].mean()) for e in ex]))
        g.remove()
        return out
    before = mean_loss()
    params = fit.Gains(tgts, "cpu")
    fit.train(model, ex, params, lr=5e-2, epochs=8, accum=2, log_every=100)
    gains = params.values()
    params.remove()
    assert mean_loss(gains) < before - 0.05 and mean_loss() == before  # fitted lower; hooks removed afterwards

    # the spec, built like any edit, is the model with the gains (few distinct values: quantisation is exact)
    vals = {t: rng.choice([0.9, 1.0, 1.15], size=g.shape) for t, g in gains.items()}
    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    meta = {"params": "gains", "widths": {t: w for t, _, w in tgts}}
    spec = fit.write_spec("c", "F", vals, meta, levels=8)
    assert all(op["op"] == "scale_cols" for op in json.loads(spec.read_text())["ops"])
    out, _ = build(spec, tmp_path / "built", base_dir=base)
    edited = transformers.Qwen2ForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([ex[0]["ids"]])
    g = fit.Gains(tgts, "cpu")
    for t, (p, _) in g.params.items():
        p.data = torch.tensor(vals[t], dtype=torch.float32)
    with torch.no_grad():
        want, got = model(ids).logits, edited(ids).logits
    g.remove()
    assert torch.allclose(want, got, atol=1e-5)

    shuffled = fit.write_spec("c", "F_shuf", vals, meta, levels=8, shuffle=True)
    a, b = (json.loads(p.read_text())["ops"] for p in (spec, shuffled))
    assert sorted(o["factor"] for o in a) == sorted(o["factor"] for o in b)  # same values, other units


def test_quantisation_keeps_strongly_moved_units():
    from wsp.fit import _levels
    g = 1 + 0.005 * np.random.default_rng(0).standard_t(2, 9728)
    moved = np.where(np.abs(g - 1) >= 0.002)[0]
    q = g.copy()
    for grp in _levels(g, moved, 256):
        q[grp] = g[grp].mean()
    assert np.abs(q[moved] - g[moved]).max() < 0.01 and np.linalg.norm(q - g) / np.linalg.norm(g[moved] - 1) < 0.03


def test_svd_spec_builds_the_fitted_model_and_pairs_move_toward_the_winner(tmp_path, tiny_lm, monkeypatch):
    """Singular-value scales: the spec of dir_ablate ops builds the hooked model; a DPO pair raises the passing
    continuation against the failing one, and an anchor's KL stays near 0 while the pair moves."""
    from wsp import case as C
    from wsp import fit, profiles
    from wsp.edits import build

    model, base = tiny_lm
    tgts = fit.targets(model, [0, 1], fit.DEFAULT_MODULES)
    rng = np.random.default_rng(1)
    p = rng.integers(0, 64, 5).tolist()
    win, lose = {"ids": p + [3, 4, 5, 6], "start": 5, "stop": 9}, {"ids": p + [7, 8, 9, 10], "start": 5, "stop": 9}
    anchor = {"kind": "anchor", "ids": rng.integers(0, 64, 12).tolist(), "start": 4, "stop": 12}
    ex = [{"kind": "pair", "win": win, "lose": lose}, anchor]
    params = fit.SVF(tgts, "cpu", k=3, zmax=0.5)
    H = fit._Head(model)

    def margin():
        with torch.no_grad():
            return float(H.logp(win)[0].sum() - H.logp(lose)[0].sum())
    before = margin()
    fit.train(model, ex, params, lr=5e-2, epochs=20, accum=2, beta=1.0, nll=0.0, log_every=100)
    assert margin() > before + 0.1
    vals = params.values()
    assert all(np.abs(v).max() <= 0.5 for k, v in vals.items() if not k.startswith("U|"))

    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    spec = fit.write_spec("c", "S", vals, {"params": "svd"}, min_change=0.0)
    assert {op["op"] for op in json.loads(spec.read_text())["ops"]} == {"dir_ablate"}
    out, _ = build(spec, tmp_path / "built", base_dir=base)
    edited = transformers.Qwen2ForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([win["ids"]])
    with torch.no_grad():
        want, got = model(ids).logits, edited(ids).logits
    assert torch.allclose(want, got, atol=1e-3)
    params.remove()
    with torch.no_grad():
        assert not torch.allclose(model(ids).logits, got, atol=1e-3)  # hooks removed: the starting model again
    rev = fit.write_spec("c", "S_rev", vals, {"params": "svd"}, dose=-1.0, min_change=0.0)
    a, b = (json.loads(q.read_text())["ops"] for q in (spec, rev))
    assert [o["alpha"] for o in a] == [-o["alpha"] for o in b]  # a negative dose reverses every scale


def test_examples_by_objective(tmp_path, monkeypatch):
    """sft keeps passing samples, rloo weights them by advantage, pairs match passing with failing samples of one
    start (a branch run: the same thinking), anchors keep their sources out of the objective."""
    from wsp import case as C
    from wsp import data, fit

    items = {f"m{i}": {"id": f"m{i}", "source": "math", "task": "math", "problem": f"q{i}", "answer": "1"}
             for i in range(3)}
    items["i0"] = {"id": "i0", "source": "ifsynth", "task": "math", "problem": "q", "answer": "1"}
    monkeypatch.setattr(data, "by_id", lambda: items)
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    (tmp_path / "cases" / "c" / "sets").mkdir(parents=True)
    (tmp_path / "cases" / "c" / "sets" / "dev.json").write_text(json.dumps(["m2"]))
    (tmp_path / "cases" / "c" / "sets" / "fit.json").write_text(json.dumps(["m0", "m1", "i0"]))
    row = lambda i, k, ok, **kw: {"id": i, "k": k, "source": items[i]["source"], "correct": ok, "cut": False,  # noqa: E731
                                  "text": f"<think>t{k}</think> a{k}", **kw}
    rows = [row("m0", 0, True), row("m0", 1, False), row("m0", 2, False), row("m1", 0, True), row("m1", 1, True),
            row("m2", 0, True), row("m2", 1, False), row("i0", 0, True)]
    (tmp_path / "runs" / "c").mkdir(parents=True)
    (tmp_path / "runs" / "c" / "base__fit.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    br = [row("m0", b, b == 0, from_k=0, prefix_chars=len("<think>t0</think>")) for b in range(3)]
    for r in br:
        r["text"] = "<think>t0</think>" + f" a{r['k']}"
    (tmp_path / "runs" / "c" / "base__fit.branches.jsonl").write_text("\n".join(json.dumps(r) for r in br))

    class Tok:
        all_special_ids, eos_token = [], "E"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "P:" + msgs[0]["content"]

        def __call__(self, text, add_special_tokens=True):
            return type("T", (), {"input_ids": [ord(ch) for ch in text]})()
    prof = {"thinking": True}
    ex, n = fit.examples("c", ["base__fit"], Tok(), prof, "sft", seed=0)
    assert {e["id"] for e in ex} == {"m0", "m1", "i0"} and n["held_out"] == 2  # dev problem m2 never enters
    (tmp_path / "runs" / "c" / "base__dev.jsonl").write_text(json.dumps(row("m2", 0, True)))
    assert fit.examples("c", ["base__dev"], Tok(), prof, "sft")[1]["held_out"] == 1  # not even from a run on dev
    ex, _ = fit.examples("c", ["base__fit"], Tok(), prof, "rloo", anchor=["ifsynth"])
    assert sorted(round(e["w"], 2) for e in ex if e["kind"] == "nll") == [-0.5, -0.5, 1.0]  # m1: all pass, no signal
    assert [e["id"] for e in ex if e["kind"] == "anchor"] == ["i0"]
    ex, _ = fit.examples("c", ["base__fit"], Tok(), prof, "rloo", credit="answer")
    e = ex[0]
    assert "".join(map(chr, e["ids"][e["start"]:e["stop"]])).startswith(" a")  # after the thinking only
    ex, _ = fit.examples("c", ["base__fit.branches"], Tok(), prof, "pairs", per_problem=5)
    assert len(ex) == 1 and ex[0]["kind"] == "pair"  # one passing branch: one pair
    w = ex[0]["win"]
    assert "".join(map(chr, w["ids"][w["start"]:w["stop"]])) == " a0E"  # credit after the shared thinking


def test_units_fit_and_spec_build_the_fitted_model(tmp_path, tiny_lm, monkeypatch):
    """Rewritten MLP units: chosen by gradient outside the units the start's chain rewrote; the spec (start values plus
    the deltas, float16) builds the hooked model; averaging takes the union, dose scales, shuffle moves the deltas."""
    from wsp import case as C
    from wsp import fit, profiles
    from wsp.edits import build

    model, base = tiny_lm
    d = tmp_path / "cases" / "c"
    (d / "edits").mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"profile": "tiny", "parent": None}))
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(profiles, "base_dir", lambda name: base)
    chain = d / "edits" / "P.json"
    chain.write_text(json.dumps({"name": "P", "base": "tiny", "ops": [
        {"op": "set_cols", "tensor": "model.layers.1.mlp.down_proj.weight", "cols": [0, 1, 2],
         "values": [[0.0] * 3] * 16},
        {"op": "set_rows", "tensor": "model.layers.0.mlp.up_proj.weight", "rows": [5], "values": [[0.0] * 16]}]}))
    exclude = fit.chain_units(chain)
    assert exclude == {1: {0, 1, 2}, 0: {5}}

    rng = np.random.default_rng(0)
    ex = [{"kind": "nll", "w": 1.0, "ids": (rng.integers(0, 8, 6).tolist() * 4), "start": 6, "stop": 24}
          for _ in range(4)]
    mlps, chosen = fit.choose_units(model, ex, [0, 1], 10, exclude, (1.0, 0.1, 1.0, 0.0))
    assert sum(map(len, chosen.values())) == 10
    assert not set(chosen.get(1, [])) & {0, 1, 2} and 5 not in chosen.get(0, [])
    H = fit._Head(model)
    with torch.no_grad():
        before = float(np.mean([float(-H.logp(e)[0].mean()) for e in ex]))
    params = fit.Units(mlps, chosen, "cpu")
    fit.train(model, ex, params, lr=2e-2, epochs=8, accum=2, log_every=100)
    with torch.no_grad():
        after = float(np.mean([float(-H.logp(e)[0].mean()) for e in ex]))
        ids = torch.tensor([ex[0]["ids"]])
        want = model(ids).logits
    vals = params.values()
    params.remove()
    assert after < before - 0.05

    meta = {"params": "units", "excluded": {"0": [5], "1": [0, 1, 2]}}
    spec = fit.write_spec("c", "U", vals, meta)
    assert {o["op"] for o in json.loads(spec.read_text())["ops"]} == {"set_rows", "set_cols"}
    out, _ = build(spec, tmp_path / "built", base_dir=base)
    with torch.no_grad():
        got = transformers.Qwen2ForCausalLM.from_pretrained(out).eval()(ids).logits
    assert torch.allclose(want, got, atol=2e-2)

    half = {k: v for k, v in vals.items()}  # a second fit with other units: the mean over the union
    m0 = next(k.removesuffix("|units") for k in vals if k.endswith("|units"))
    mean = fit._mean_units([vals, {k: v for k, v in half.items() if not k.startswith(m0)}])
    assert np.allclose(mean[f"{m0}|gate"], vals[f"{m0}|gate"] / 2)
    z, zr, z0 = (np.load(fit.write_spec("c", n, vals, meta, dose=x).with_suffix(".npz"))
                 for n, x in (("U1", 1.0), ("U_rev", -1.0), ("U0", 0.0)))
    assert np.allclose(z["gate0"] + zr["gate0"], 2 * z0["gate0"], atol=1e-2)  # a negative dose reverses the deltas
    for _ in range(5):
        zs = np.load(fit.write_spec("c", "U_shuf", vals, meta, shuffle=True).with_suffix(".npz"))
        for j, m in enumerate(sorted(k.removesuffix("|units") for k in vals if k.endswith("|units"))):
            assert not set(zs[f"units{j}"].tolist()) & set(meta["excluded"][m.split(".")[1]])


def test_window_and_starts_inside_the_thinking(tmp_path, monkeypatch):
    """--window caps the credited tokens; continuations of different starts of one sample are separate groups."""
    from wsp import case as C
    from wsp import data, fit

    items = {"m0": {"id": "m0", "source": "math", "task": "math", "problem": "q", "answer": "1"}}
    monkeypatch.setattr(data, "by_id", lambda: items)
    monkeypatch.setattr(C, "CASES", tmp_path / "cases")
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    (tmp_path / "cases" / "c" / "sets").mkdir(parents=True)
    think = "a\nbb\nccc\n</think> x"
    rows = [{"id": "m0", "k": b, "from_k": 0, "source": "math", "correct": b % 2 == 0, "cut": False, "at": at,
             "prefix_chars": c, "text": think[:c] + f"y{b}" * 5} for at, c in ((0.25, 2), (0.5, 5)) for b in range(2)]
    (tmp_path / "runs" / "c").mkdir(parents=True)
    (tmp_path / "runs" / "c" / "base__fit.branches.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    class Tok:
        all_special_ids, eos_token = [], "E"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "P:" + msgs[0]["content"]

        def __call__(self, text, add_special_tokens=True):
            return type("T", (), {"input_ids": [ord(ch) for ch in text]})()
    ex, _ = fit.examples("c", ["base__fit.branches"], Tok(), {"thinking": True}, "pairs", window=4)
    assert len(ex) == 2  # one pair per start
    for e in ex:
        w = e["win"]
        assert "".join(map(chr, w["ids"][w["start"]:w["stop"]])) == "y0y0"  # after the start, 4 tokens


def test_cut_points_end_a_line():
    from wsp.case import _cut_points
    text = "one\ntwo two\nthree three three\n</think> 7"
    cuts = _cut_points(text, [0.1, 0.5])
    assert all(text[c - 1] == "\n" for c in cuts.values()) and cuts[0.1] < cuts[0.5] <= text.index("</think>")
