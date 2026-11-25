"""ws_lists.py (ours): word-sorting puzzles for the spelling drill, and lists of held-out words to check it.

  python derive/ws_lists.py --out-dir DIR --model MODEL_DIR [--n-gen 2500] [--piece-reps 2] [--piece-cont 0.4]
         [--n-check 40] [--exclude PUZZLES.jsonl ...]

  DIR/ws_fit.jsonl     the drill's puzzles, all of the toolkit's form ("Sort the following words alphabetically:
                       List: ...", 3 to 18 words):
      --n-gen       lists from the toolkit's own program (wsp.puzzles word_sorting: its vocabulary is the lower-case
                    words of gsm8k_train), seeded;
      piece lists   lists whose words are made from the tokenizer's own lower-case pieces (the tokenizer of
                    MODEL_DIR): every word-start piece of 2 to 16 letters is a word --piece-reps times; with
                    probability --piece-cont a continuation piece of 2 to 10 letters is appended (each continuation
                    piece about equally often). The model spells a word piece by piece, and spells right the pieces
                    it has spelled before: these lists show it every piece it can read. No text is written and no
                    word list is brought: a puzzle is the toolkit's question sentence and pieces of the model's own
                    vocabulary.
  DIR/ws_check.jsonl   --n-check lists of lower-case words of the other practice sets' texts (math_train, mmlupro,
                       ifsynth, tuluif) that the toolkit's vocabulary does not hold; these words are in NO drill
                       list as words (their pieces are, in other words): what the fit does on words it never spelled.
No puzzle is a practice item, has a benchmark fingerprint (wsp.data.bench_fingerprints) or is in a file of --exclude.
Rows: {"id", "source": "pz", "subject": "word_sorting", "task": "exact", "problem", "answer"}.
"""
import argparse
import json
import random
import re
from pathlib import Path

QUESTION = "Sort the following words alphabetically: List: "


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True, help="the start's model directory (its tokenizer is read)")
    ap.add_argument("--n-gen", type=int, default=2500)
    ap.add_argument("--piece-reps", type=int, default=2)
    ap.add_argument("--piece-cont", type=float, default=0.4)
    ap.add_argument("--piece-share", type=float, default=1.0, help="share of the word-start pieces used (a rehearsal)")
    ap.add_argument("--n-check", type=int, default=40)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp import data, puzzles  # noqa: PLC0415
    fp = data.fingerprint
    seen = set(data.bench_fingerprints()) | {fp(r["problem"]) for r in data.load("practice:bbhsynth")}
    for path in a.exclude:
        seen |= {fp(json.loads(line)["problem"]) for line in open(path) if line.strip()}

    def take(prefix, problem, answer, into):
        f = fp(problem)
        if f in seen:
            return False
        seen.add(f)
        into.append({"id": f"{prefix}_word_sorting_{a.seed}_{len(into)}", "source": "pz", "subject": "word_sorting",
                     "task": "exact", "problem": problem, "answer": answer})
        return True

    # the check lists first: their words are kept out of every drill list
    vocab = set(puzzles._vocabulary())
    other = set()
    for name in ("math_train", "mmlupro", "ifsynth", "tuluif"):
        for r in data.load(f"practice:{name}"):
            text = r["problem"] + " " + " ".join(map(str, r.get("options") or []))
            other |= {w for w in re.findall(r"\b[a-z]+\b", text) if 3 <= len(w) <= 14}
    other = sorted(other - vocab)
    rc, check, held = random.Random(a.seed + 3), [], set()
    while len(check) < a.n_check:
        ws = rc.sample(other, rc.randint(3, 18))
        if take("wsc", QUESTION + " ".join(ws), " ".join(sorted(ws)), check):
            held |= set(ws)

    gen = puzzles.generators()["word_sorting"]
    rg, part = random.Random(f"{a.seed}:word_sorting"), []
    while len(part) < a.n_gen:
        take("wsg", *gen(rg), part)
    rows = part

    tok = AutoTokenizer.from_pretrained(a.model)
    vocab_t = tok.get_vocab()
    start = sorted(t[1:] for t in vocab_t if re.fullmatch(r"Ġ[a-z]{2,16}", t))
    cont = sorted(t for t in vocab_t if re.fullmatch(r"[a-z]{2,10}", t))
    rp = random.Random(a.seed + 1)
    if a.piece_share < 1:
        start = rp.sample(start, max(40, int(len(start) * a.piece_share)))
    words, cpool = [], []
    for _ in range(a.piece_reps):
        order = start[:]
        rp.shuffle(order)
        for s in order:
            w = s
            if cont and rp.random() < a.piece_cont:
                if not cpool:
                    cpool = cont[:]
                    rp.shuffle(cpool)
                c = cpool.pop()
                if len(s) + len(c) <= 18:
                    w = s + c
            if w not in held:
                words.append(w)
    rp.shuffle(words)
    part, i = [], 0
    while i < len(words):
        n = rp.randint(3, 18)
        ws = list(dict.fromkeys(words[i:i + n]))
        i += n
        if len(ws) >= 3:
            take("wsp", QUESTION + " ".join(ws), " ".join(sorted(ws)), part)
    covered = set()
    for w in set(words):
        covered |= set(tok.convert_ids_to_tokens(tok(" " + w, add_special_tokens=False)["input_ids"]))
    n_piece = len(part)
    rows += part
    random.Random(a.seed + 2).shuffle(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ws_fit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "ws_check.jsonl").write_text("".join(json.dumps(r) + "\n" for r in check))
    print(f"word-sorting lists: {a.n_gen} from the toolkit's program (vocabulary {len(vocab)}); {n_piece} piece lists "
          f"({len(words)} words from {len(start)} word-start and {len(cont)} continuation pieces; covered: "
          f"{len(covered & {'Ġ' + s for s in start})} word-start, {len(covered & set(cont))} continuation); "
          f"{len(check)} check lists of {len(held)} held-out practice words -> {out}")


if __name__ == "__main__":
    main()
