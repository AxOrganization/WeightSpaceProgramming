"""Built-model check of a unit edit, before any generation: build the edit with the toolkit (the same build the subnet
runs), teacher-force the case's own base traces through the parent and through the edited model, and compare the eval
sampler's next-token distributions.

  python unit_check.py --case C --edit HX_B --words W1 [W2 ...] --runs dev_math dev_mmlupro dev_ifsynth auxtest \
      [--twin H_B] [--items data/aux_if.jsonl] [--traces 2] [--tokens 6000] [--tau 0.01] [--device cuda]

Per run (the first --traces base traces in dplib.pick_traces order, first --tokens tokens of each; 0 = whole trace):
  dp n / mk before -> after   decision positions (the PARENT's p(marker) >= tau at thinking positions) and the summed
                              p(marker) there under the parent and under the edit = the expected number of openers
                              drawn at those positions;
  top1 changed (thinking / answer)  share of the other thinking positions / of the answer positions where the edit's
                              most likely next token differs from the parent's;
  KL                          mean KL(parent || edit) of the temperature-scaled next-token distributions at those
                              other positions (nats).
  vs twin (--twin T)          the edit's marker mass at the decision positions divided by that of T (an earlier
                              check of the always-on unit of the same dose on the same runs): the gated edit should
                              lower the markers about as much as its always-on twin where it is meant to act.
A gated penalty must lower the marker mass at the decision positions of the calculation runs by about as much as the
always-on unit, keep it (and everything else) unchanged on the instruction runs, and change almost nothing away from
the decision positions; a close unit must raise p(close tag) at its decision positions and nothing else.
Writes results/<EDIT>__check.json. Practice data and self-written prompts only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dplib as D  # noqa: E402
from wsp.case import cdir, model_path, run_path  # noqa: E402
from wsp.common import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--words", nargs="+", required=True, help="marker words (as for the gate); <...> = a special token")
    ap.add_argument("--runs", nargs="+", required=True, help="base runs to read traces from")
    ap.add_argument("--items", nargs="*", default=[], help="self-written item files (runs of run_items.py)")
    ap.add_argument("--traces", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--twin", default=None, help="edit whose earlier check (results/<TWIN>__check.json) is compared")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    missing = [str(run_path(a.case, "base", r)) for r in a.runs if not run_path(a.case, "base", r).exists()]
    if missing:
        raise SystemExit(f"missing base runs: {missing}")
    cm0 = D.CaseModel(model_path(a.case, "base"), a.device)
    cm1 = D.CaseModel(model_path(a.case, a.edit), a.device)
    sampler = D.EvalSampler.for_case(a.case)
    tokens = D.Tokens(a.case, D.read_items(a.items))
    mk = torch.as_tensor(D.word_ids(tokens.tok, a.words), device=a.device)
    out = {}
    print(f"{'run':16s} {'dp n':>6s} {'mk before -> after':>20s} {'top1 changed think/answer':>27s} {'KL think/answer':>17s}")
    for run in a.runs:
        rows, _ = D.pick_traces(a.case, "base", run, a.traces, a.seed, tokens.items)
        acc = {"dp": 0, "mk0": 0.0, "mk1": 0.0, "think": [0, 0, 0.0], "answer": [0, 0, 0.0]}
        for r in rows:
            enc = tokens.encode(r)
            ids, n0 = (enc["ids"] if a.tokens <= 0 else enc["ids"][: a.tokens]), enc["n_prompt"]
            cl = np.flatnonzero(ids[n0:] == tokens.close_id)
            think_end = n0 + int(cl[0]) if len(cl) else len(ids)
            tq = np.arange(n0 - 1, min(think_end, len(ids) - 1))
            aq = np.arange(think_end, len(ids) - 1) if len(cl) else np.zeros(0, np.int64)
            h0 = cm0.run(ids, resid=True)["h"]
            h1 = cm1.run(ids, resid=True)["h"]
            fo = D.first_occurrence(ids, n0, cm0.vocab)
            for kind, qq in (("think", tq), ("answer", aq)):
                for c0 in range(0, len(qq), a.chunk):
                    qs = qq[c0:c0 + a.chunk]
                    sel = torch.as_tensor(qs)
                    lg0, lg1 = cm0.logits(h0[sel]), cm1.logits(h1[sel])
                    pres = D.presence_mask(fo, qs, a.device)
                    i0, p0 = D.kept_safe(sampler, lg0, pres)
                    i1, p1 = D.kept_safe(sampler, lg1, pres)
                    pm0 = (p0 * torch.isin(i0, mk)).sum(1).cpu().numpy()
                    pm1 = (p1 * torch.isin(i1, mk)).sum(1).cpu().numpy()
                    z0, z1 = sampler.processed(lg0, pres), sampler.processed(lg1, pres)
                    lq0, lq1 = torch.log_softmax(z0, -1), torch.log_softmax(z1, -1)
                    kl = (lq0.exp() * (lq0 - lq1)).sum(-1).cpu().numpy()
                    top = (z0.argmax(-1) != z1.argmax(-1)).cpu().numpy()
                    dp = (pm0 >= a.tau) if kind == "think" else np.zeros(len(qs), bool)
                    acc["dp"] += int(dp.sum())
                    acc["mk0"] += float(pm0[dp].sum())
                    acc["mk1"] += float(pm1[dp].sum())
                    acc[kind][0] += int((~dp).sum())
                    acc[kind][1] += int(top[~dp].sum())
                    acc[kind][2] += float(kl[~dp].sum())
        e = {"traces": len(rows), "dp_n": acc["dp"], "markers_before": round(acc["mk0"], 4),
             "markers_after": round(acc["mk1"], 4)}
        for kind in ("think", "answer"):
            n, ch, kl = acc[kind]
            e[f"{kind}_positions"] = n
            e[f"{kind}_top1_changed"] = round(ch / n, 5) if n else None
            e[f"{kind}_kl_mean"] = round(kl / n, 6) if n else None
        out[run] = e
        f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"{run:16s} {e['dp_n']:6d} {e['markers_before']:9.3f} -> {e['markers_after']:7.3f} "
              f"{f(e['think_top1_changed']):>13s} / {f(e['answer_top1_changed']):<11s} "
              f"{f(e['think_kl_mean']):>8s} / {f(e['answer_kl_mean'])}", flush=True)
    for e in out.values():
        e["after_vs_before"] = round(e["markers_after"] / e["markers_before"], 4) if e["markers_before"] > 0 else None
    if a.twin:
        tw = cdir(a.case) / "results" / f"{a.twin}__check.json"
        if not tw.exists():
            raise SystemExit(f"--twin {a.twin}: run unit_check.py on it first ({tw} missing)")
        t = json.loads(tw.read_text())
        same = {k: t.get(k) == v for k, v in (("words", a.words), ("tau", a.tau), ("tokens", a.tokens))}
        if not all(same.values()) or t.get("traces", a.traces) != a.traces:
            print(f"NOTE: the twin's check used other settings: {same}, traces {t.get('traces')}", flush=True)
        print(f"{'run':16s} {'markers after: edit / twin ' + a.twin:>34s} {'vs twin':>8s} {'after/before':>13s}")
        for run, e in out.items():
            te = t["runs"].get(run)
            if te is None:
                continue
            e["twin"], e["markers_after_twin"] = a.twin, te["markers_after"]
            e["vs_twin"] = round(e["markers_after"] / te["markers_after"], 4) if te["markers_after"] > 0 else None
            f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"{run:16s} {e['markers_after']:16.3f} / {te['markers_after']:<16.3f} {f(e['vs_twin']):>8s} "
                  f"{f(e['after_vs_before']):>13s}", flush=True)
    write_json(cdir(a.case) / "results" / f"{a.edit}__check.json",
               {"edit": a.edit, "words": a.words, "tau": a.tau, "tokens": a.tokens, "traces": a.traces,
                "twin": a.twin, "runs": out})
    print("wrote", cdir(a.case) / "results" / f"{a.edit}__check.json")


if __name__ == "__main__":
    main()
