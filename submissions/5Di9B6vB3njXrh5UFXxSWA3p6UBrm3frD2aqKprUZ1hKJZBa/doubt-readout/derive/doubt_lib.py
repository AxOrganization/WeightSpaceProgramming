"""Shared helpers for the doubt-readout method (uses the wsp toolkit; run from the WeightSpaceProgramming checkout
with WSP_CASES / WSP_WORK set as for `python -m wsp.case`)."""
import json
import re
from pathlib import Path

import numpy as np

SENT_START = re.compile(r"(?:\n\n|\n|(?<=[.!?])[ \t]+)([A-Z][a-z]+)\b")


def case_dir(case):
    from wsp.case import cdir  # noqa: PLC0415
    return Path(cdir(case))


def scratch(case):
    from wsp.common import WORK  # noqa: PLC0415
    d = Path(WORK) / "doubt" / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs(case, model, tags):
    """Generation rows of `model` (base = the case's starting model) on the given set tags."""
    from wsp.case import run_path  # noqa: PLC0415
    rows = []
    for t in tags:
        p = Path(run_path(case, model, t))
        if not p.exists():
            raise SystemExit(f"no run {p}: python -m wsp.case run {case} {model} {t}")
        rows += [json.loads(x) for x in p.read_text().split("\n") if x.strip()]
    return rows


def think_of(row):
    return row["text"].split("</think>")[0]


def sentence_starts(think):
    """[(char position of the word, word)] for capitalized words that start a sentence or a line."""
    return [(m.start(1), m.group(1)) for m in SENT_START.finditer(think)]


def prompt_of(tok, item):
    from wsp.data import user_message  # noqa: PLC0415
    return tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def starting_model(case):
    from wsp.case import BASE, model_path  # noqa: PLC0415
    return model_path(case, BASE)


def readout_scale(model_dir):
    """Output embedding matrix E (vocab x d, the tied input embedding when there is no lm_head) and the final norm's
    effective gain (1 + weight for Qwen3.5's zero-centred RMSNorm), from the safetensors of a model dir."""
    from safetensors import safe_open  # noqa: PLC0415
    md = Path(model_dir)
    idx = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]

    def get(n):
        with safe_open(str(md / idx[n]), "pt") as f:
            return f.get_tensor(n).float().numpy()
    head = [n for n in idx if n.endswith("lm_head.weight")]
    emb = [n for n in idx if n.endswith("embed_tokens.weight") and "visual" not in n]
    E = get(head[0] if head else emb[0])
    norm = [n for n in idx if re.search(r"(language_model\.)?norm\.weight$", n) and "layers." not in n
            and "visual" not in n and "mtp" not in n]
    g = get(sorted(norm, key=len)[0])
    cfg = json.loads((md / "config.json").read_text())
    mt = " ".join(str(c.get("model_type", "")) for c in (cfg, cfg.get("text_config", {})))
    # Qwen3.5 / Qwen3-Next RMSNorm computes x * (1 + weight) (zero-centred weight); check the modeling code if unsure
    gain = 1.0 + g if ("qwen3_5" in mt or "qwen3_next" in mt) else g
    return E, gain


def tensor(model_dir, name):
    from safetensors import safe_open  # noqa: PLC0415
    md = Path(model_dir)
    idx = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(md / idx[name]), "pt") as f:
        return f.get_tensor(name).float().numpy()


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)
