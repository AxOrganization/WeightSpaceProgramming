"""Step 4 of the method: selectivity check of an edit against its starting model, teacher-forced on the model's own
practice traces. Inside the thinking block and in the final answer it reports the shift of the end-of-thinking
log-odds, the KL of the remaining next-token distribution (the end token left out) and the top-1 agreement.
A good edit moves the end token in the thinking block, moves it little or downwards in the answer, and leaves the
other tokens alone (KL of a few 1e-3 nats, top-1 agreement near 99%).

usage: python derive/verify.py <case> <edit> [n_traces] [max_tokens]"""
import json, sys
import numpy as np, torch
from wsp import case as C, data
from wsp.common import read_jsonl
from wsp.data import user_message
from wsp.probe import Probe


def main():
    case, edit = sys.argv[1], sys.argv[2]
    n_tr = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    rows = read_jsonl(C.run_path(case, "base", "dev128")); items = data.by_id()
    rng = np.random.default_rng(1)
    pick = [rows[i] for i in rng.permutation(len(rows))[:n_tr]]
    PB, PE = Probe(C.model_path(case, "base")), Probe(C.model_path(case, edit))
    tok = PB.tok; END = tok.convert_tokens_to_ids("</think>")
    cb, ce = {}, {}
    hb = PB.norm.register_forward_hook(lambda m, i, o: cb.__setitem__("f", o[0]))
    he = PE.norm.register_forward_hook(lambda m, i, o: ce.__setitem__("f", o[0]))
    stats = {"think": [], "answer": []}
    for r in pick:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]; gi = tok(r["text"], add_special_tokens=False)["input_ids"]
        g = np.array(gi); e = np.where(g == END)[0]
        if len(e): a = max(0, int(e[0]) - max_tok); gi = gi[a: int(e[0]) + 400]; t_end = int(e[0]) - a
        else: gi = gi[:max_tok]; t_end = len(gi)
        ids = torch.tensor([pi + gi], device=PB.device)
        with torch.no_grad():
            PB.model(input_ids=ids, logits_to_keep=1, use_cache=False); PE.model(input_ids=ids, logits_to_keep=1, use_cache=False)
            acc = {"think": [], "answer": []}
            for c in range(len(pi) - 1, len(pi) + len(gi), 512):
                b = torch.log_softmax((cb["f"][c:c + 512] @ PB.head.weight.T).double(), -1); e2 = torch.log_softmax((ce["f"][c:c + 512] @ PE.head.weight.T).double(), -1)
                lo = lambda x: x[:, END] - torch.log1p(-x[:, END].exp().clamp(max=1 - 1e-12))
                d = lo(e2) - lo(b)
                b2, e3 = b.clone(), e2.clone(); b2[:, END] = -1e9; e3[:, END] = -1e9
                pb, pe = torch.log_softmax(b2, -1), torch.log_softmax(e3, -1)
                kl = (pb.exp() * (pb - pe)).sum(-1); agree = (pb.argmax(-1) == pe.argmax(-1)).double()
                j = torch.arange(c, c + len(b), device=b.device) - (len(pi) - 1)       # index of the generated token predicted
                for key, m in (("think", j < t_end), ("answer", j > t_end)):
                    if m.any(): acc[key].append(torch.stack([d[m], kl[m], agree[m], b[m, END].exp(), e2[m, END].exp()], 1))
            for key in acc:
                if acc[key]:
                    t = torch.cat(acc[key]); stats[key].append((t[:, 0].mean().item(), t[:, 0].std().item(), t[:, 1].mean().item(), t[:, 1].max().item(), t[:, 2].mean().item(), t[:, 3].max().item(), t[:, 4].max().item(), len(t)))
    for key, v in stats.items():
        a = np.array(v)
        if len(a): print(f"[{key}] traces {len(a)} positions {int(a[:, 7].sum())}: end-token log-odds shift mean {a[:, 0].mean():+.2f} (sd within trace {a[:, 1].mean():.2f}); KL of the other tokens mean {a[:, 2].mean():.5f} nats, worst position {a[:, 3].max():.3f}; top-1 agreement {a[:, 4].mean():.4f}; max P(end) per trace {a[:, 5].mean():.3f} -> {a[:, 6].mean():.3f}")


if __name__ == "__main__":
    main()
