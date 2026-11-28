#!/usr/bin/env python3
"""ws_draw.py --out DIR [--fit 3000] [--max-len 24] [--per-band 80] [--pow 1] [--tag la] [--bands 3-6,7-10,...]
: word-sorting puzzles for a test on LIST LENGTH
(practice only).

The toolkit's generator draws a list as `rng.sample(vocab, randint(3, 18))`: every length equally often, none over 18.
A working fitted on such lists reads worst on the longest ones (14-17 words: 82% against 95-99% below). Here the same
vocabulary (split in two halves by a hash of the word) gives
  DIR/ws_fit.jsonl   `fit` lists from half A, lengths 3..max-len, a length n drawn with weight n (long lists more
                     often, and somewhat longer than the generator's longest),
  DIR/ws_test.jsonl  `per-band` lists of half B (no word of them in a fit list) for each band of lengths
                     3-6, 7-10, 11-14, 15-18, 19-21, 22-24,
  DIR/ids_ws_test.json, DIR/items_ws.jsonl (all, for REQ_ITEMS).
The puzzle text and the answer are formed exactly as the toolkit forms them. No benchmark question
(wsp.data.bench_fingerprints) and none of the shipped puzzles is kept.
--pow P: a length n is drawn with weight n**P (0 = every length equally often). --tag: the id tag of the fit lists
(another tag and seed for another draw). --bands: the test bands (default the six above); --fit 0 writes test lists
only."""
import argparse, hashlib, json, random
from pathlib import Path

from wsp import data
from wsp import puzzles as TK

BANDS = ((3, 6), (7, 10), (11, 14), (15, 18), (19, 21), (22, 24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--fit", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=24); ap.add_argument("--per-band", type=int, default=80)
    ap.add_argument("--pow", type=float, default=1.0); ap.add_argument("--tag", default="la"); ap.add_argument("--bands", default="")
    ap.add_argument("--vocab", choices=("a", "all"), default="a", help="fit lists from half A (a test on unseen words) or from the whole vocabulary (a fit for a candidate)")
    a = ap.parse_args()
    vocab = TK._vocabulary()
    half = {w: int(hashlib.md5(w.encode()).hexdigest(), 16) % 2 for w in vocab}
    va, vb = [w for w in vocab if half[w] == 0], [w for w in vocab if half[w] == 1]
    fps = data.bench_fingerprints()
    seen = {data.fingerprint(r["problem"]) for r in data.load("practice:bbhsynth") if r["subject"] == "word_sorting"}

    def one(rng, words, n):
        ws = rng.sample(words, n)
        return f"Sort the following words alphabetically: List: {' '.join(ws)}", " ".join(sorted(ws))

    def draw(tag, words, n, seed, pick_len):
        rng, out, tries = random.Random(seed), [], 0
        while len(out) < n:
            tries += 1
            if tries > 200 * n:
                raise SystemExit("too few new puzzles")
            problem, answer = one(rng, words, pick_len(rng))
            f = data.fingerprint(problem)
            if f in fps or f in seen:
                continue
            seen.add(f)
            out.append({"id": f"bbhsynth_word_sorting_{tag}{len(out)}", "source": "bbhsynth", "task": "exact", "subject": "word_sorting",
                        "problem": problem, "answer": answer})
        return out
    lens = list(range(3, a.max_len + 1))
    wts = [n ** a.pow for n in lens]
    fit = draw(a.tag, va if a.vocab == "a" else list(vocab), a.fit, "len:fit" if a.tag == "la" else f"len:fit:{a.tag}", lambda rng: rng.choices(lens, weights=wts)[0]) if a.fit else []
    bands = tuple(tuple(int(x) for x in b.split("-")) for b in a.bands.split(",")) if a.bands else BANDS
    test = []
    for lo, hi in bands:
        test += draw(f"lt{lo}x{hi}x", vb, a.per_band, f"len:test:{lo}", lambda rng, lo=lo, hi=hi: rng.randint(lo, hi))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ws_fit.jsonl", fit), ("ws_test.jsonl", test), ("items_ws.jsonl", fit + test)):
        with open(out / name, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    json.dump([r["id"] for r in test], open(out / "ids_ws_test.json", "w"))
    mean = sum(len(r["answer"].split()) for r in fit) / max(1, len(fit))
    print(f"fit {len(fit)} lists of {'the whole vocabulary' if a.vocab == 'all' else 'half A'}, lengths 3-{a.max_len} (weight n^{a.pow:g}; mean {mean:.1f} words); test {len(test)} lists of half B in {len(bands)} bands")


if __name__ == "__main__":
    main()
