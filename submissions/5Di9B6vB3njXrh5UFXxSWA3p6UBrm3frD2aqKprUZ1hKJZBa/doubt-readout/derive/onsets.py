"""Step 1 - the doubt readout direction, and how the model's own traces use the opener words.

Doubt openers are the discourse markers that open a re-check or a change of course; progress openers are the ones
that continue, conclude or state a result. The two role lists below define the behavior (like a contrast kind of the
toolkit); the traces only confirm that the model uses them. The readout direction is the residual-stream direction
that raises the doubt openers' logits against the progress openers' at the last layer:
  u = normalize( mean_{w in doubt} E[w] * gain  -  mean_{w in progress} E[w] * gain )
(E = output embedding, gain = the final norm's effective weight; words that are not a single token are skipped).

Do NOT pick the words by how enriched they are in unfinished traces: unfinished traces are mostly hard problems, so
content words of hard topics look 'enriched' (a topic confound), and the second half of an unfinished trace is mostly
verbatim repetition, where every opener is rarer. The report below (per practice source, capped counts per 10k
characters) is for reading, not for selection.

  python <submission>/derive/onsets.py <case> --tags dev128 [--doubt ...] [--progress ...]
Writes <scratch>/onsets.json and <scratch>/readout.npz.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doubt_lib as L  # noqa: E402

DOUBT = ["Wait", "However", "But", "Maybe", "Perhaps", "Hmm", "Alternatively", "Actually"]
PROGRESS = ["Therefore", "Thus", "So", "Now", "The", "First", "Next", "Let"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--tags", nargs="+", default=["dev128"])
    ap.add_argument("--doubt", nargs="+", default=DOUBT)
    ap.add_argument("--progress", nargs="+", default=PROGRESS)
    ap.add_argument("--cap", type=int, default=10, help="max occurrences of a word counted per trace (report)")
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    rows = L.runs(a.case, "base", a.tags)
    model = L.starting_model(a.case)
    tok = AutoTokenizer.from_pretrained(model)
    one = lambda ws: [w for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1]  # noqa: E731
    doubt, prog = one(a.doubt), one(a.progress)
    skipped = sorted(set(a.doubt + a.progress) - set(doubt + prog))
    words = doubt + prog
    print(f"{len(rows)} traces, {sum(r['cut'] for r in rows)} cut" + (f"; skipped (not one token): {skipped}"
                                                                     if skipped else ""))
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["id"].split("_")[0]].append(r)
    for src, rs in sorted(by_src.items()):
        late, fin, lc, fc = collections.Counter(), collections.Counter(), 1e-9, 1e-9
        for r in rs:
            th = L.think_of(r)
            if r["cut"]:
                c = collections.Counter(w for p, w in L.sentence_starts(th) if p > len(th) / 2 and w in words)
                late.update({w: min(k, a.cap) for w, k in c.items()})
                lc += len(th) / 2
            else:
                c = collections.Counter(w for p, w in L.sentence_starts(th) if w in words)
                fin.update({w: min(k, a.cap) for w, k in c.items()})
                fc += len(th)
        print(f"  {src}: {sum(r['cut'] for r in rs)}/{len(rs)} cut; per 10k chars finished|late: " +
              ", ".join(f"{w} {1e4 * fin[w] / fc:.1f}|{1e4 * late[w] / lc:.1f}" for w in words))
    E, gain = L.readout_scale(model)
    ids = lambda ws: [tok.encode(w, add_special_tokens=False)[0] for w in ws]  # noqa: E731
    u = L.unit((E[ids(doubt)] * gain).mean(0) - (E[ids(prog)] * gain).mean(0))
    out = L.scratch(a.case)
    np.savez(out / "readout.npz", u=u)
    (out / "onsets.json").write_text(json.dumps({"doubt": doubt, "progress": prog}, indent=1))
    print(f"doubt openers {doubt}\nprogress openers {prog}\nreadout direction -> {out / 'readout.npz'}")


if __name__ == "__main__":
    main()
