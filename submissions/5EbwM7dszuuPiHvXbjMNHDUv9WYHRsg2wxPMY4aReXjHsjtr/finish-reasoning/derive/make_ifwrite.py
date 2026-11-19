"""Write a second instruction-following practice set, `ifwrite`: short writing tasks with one to three
IFEval-style constraints, checkable by the official checkers (the same construction as the toolkit's `ifsynth`,
wsp.data.make_ifsynth, which attaches the constraints to GSM8K-train questions).

Why: `ifsynth` is the only instruction-following practice set, and all its questions are arithmetic word
problems. An edit fitted on it learns to respect constraints while solving a word problem; free writing under
constraints (essays, letters, stories) is a different situation, and a method that makes reasoning shorter must
be checked, and fitted, there too. The tasks below are written for this method (a genre crossed with an everyday
topic); none is taken from a benchmark.

usage: python make_ifwrite.py [--train 280] [--dev 64] [--holdout 128] [--seed 5]
writes $WSP_WORK/gen/ifwrite_train.jsonl, ifwrite_dev.jsonl and ifwrite_ho0.jsonl .. ifwrite_ho3.jsonl (the holdout
in four parts): rows {id, source "ifwrite", task "ifeval", problem, instruction_id_list, kwargs, answer None}.
"""
import argparse
import json
import os
import random
from pathlib import Path

GENRES = (
    "Write a short essay about {t}.",
    "Write a blog post about {t}.",
    "Write a letter to a friend about {t}.",
    "Write a short speech about {t}.",
    "Write a short story that involves {t}.",
    "Write a poem about {t}.",
    "Describe {t} to someone who has never heard of it.",
    "Give advice to a beginner about {t}.",
    "Write a newspaper-style article about {t}.",
    "Write a dialogue between two people who discuss {t}.",
    "Write an introduction for a podcast episode about {t}.",
    "List and explain three interesting things about {t}.",
    "Write a diary entry about {t}.",
    "Write a persuasive paragraph about {t}.",
    "Write a set of practical tips related to {t}.",
    "Write a review of an imaginary book about {t}.",
    "Write an encouraging message to someone, inspired by {t}.",
    "Explain why {t} matters.",
    "Write a short lesson plan about {t}.",
    "Write a toast for a celebration themed around {t}.",
    "Write a riddle about {t} and then explain it.",
    "Write a short guide for visitors about {t}.",
    "Write the opening scene of a play about {t}.",
    "Write a thank-you note that mentions {t}.",
)
TOPICS = (
    "the first day of spring", "learning to ride a bicycle", "a village market at dawn", "saving water at home",
    "the history of the bicycle", "a rainy afternoon in a library", "volunteering at an animal shelter",
    "the importance of sleep", "a lighthouse keeper", "cooking dinner for friends", "moving to a new city",
    "the sound of a train at night", "a community garden", "learning a musical instrument",
    "a long hike in the mountains", "the invention of the printing press", "recycling in a small town",
    "a robot that learns to paint", "the migration of birds", "a family board game night",
    "starting a small bakery", "the night sky far from city lights", "public libraries", "a forgotten umbrella",
    "teamwork on a school project", "the smell of fresh bread", "a ferry crossing a wide river", "planting a tree",
    "working from home", "a street musician", "the changing of the seasons", "a broken clock tower",
    "healthy breakfast habits", "a visit to a science museum", "kindness to strangers", "a winter festival",
    "learning to swim as an adult", "the life of a honeybee", "old photographs", "a quiet morning by the sea",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=280)
    ap.add_argument("--dev", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    from lm_eval.tasks.ifeval import instructions_registry as R  # noqa: PLC0415
    from wsp.common import WORK  # noqa: PLC0415
    from wsp.data import IFSYNTH_TYPES  # noqa: PLC0415

    rng = random.Random(a.seed)
    random.seed(a.seed)            # the registry draws its random arguments from the global generator
    pairs = [(g, t) for g in GENRES for t in TOPICS]
    rng.shuffle(pairs)
    n = a.train + a.dev + a.holdout
    rows = []
    for i, (g, t) in enumerate(pairs[:n]):
        ids = []
        for iid in rng.sample(IFSYNTH_TYPES, len(IFSYNTH_TYPES)):
            if len(ids) == 1 + (i % 3):
                break
            if not any(iid in R.INSTRUCTION_CONFLICTS.get(j, ()) or j in R.INSTRUCTION_CONFLICTS.get(iid, ())
                       for j in ids):
                ids.append(iid)
        descs, kwargs = [], []
        for iid in ids:
            inst = R.INSTRUCTION_DICT[iid](iid)
            descs.append(inst.build_description())
            kwargs.append({k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None})
        rows.append({"id": f"ifwrite_{i}", "source": "ifwrite", "task": "ifeval", "answer": None,
                     "problem": g.format(t=t) + " " + " ".join(descs), "instruction_id_list": ids, "kwargs": kwargs})
    out = Path(WORK) / "gen"
    os.makedirs(out, exist_ok=True)
    parts = {"train": rows[:a.train], "dev": rows[a.train:a.train + a.dev]}
    hold = rows[a.train + a.dev:]
    for k in range(4):
        parts[f"ho{k}"] = hold[k::4]
    for name, part in parts.items():
        with open(out / f"ifwrite_{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"{out / ('ifwrite_' + name + '.jsonl')}: {len(part)} tasks")


if __name__ == "__main__":
    main()
