"""Build data/ifguard.jsonl: instruction-following guard items for the five instruction types of lm-eval's IFEval
registry that the toolkit's ifsynth practice set does not cover (repeat the request, two responses, response language, first word of
the n-th paragraph, constrained response). Like ifsynth, each item is a GSM8K *train* question (from the toolkit's
gsm8k_train practice file, never a benchmark item) plus one constraint written by lm-eval's IFEval instruction
registry (the generator ifsynth uses), so the official checkers grade it; no IFEval prompt is used. The n-th
paragraph items get a paragraph count of 2-5 and an n within it (the registry's own draw can ask for a paragraph
beyond the count, which no answer can satisfy). Two of these types (repeat the request; two responses) need the
model to copy or duplicate text: exactly what an edit against repetition could damage, so they guard that side
effect. Already built (data/ifguard.jsonl); there is no need to rerun this.

  python derive/make_ifguard.py <WeightSpaceProgramming>/data/practice/gsm8k_train.jsonl.gz data/ifguard.jsonl [n_per_type]
"""
import gzip
import json
import random
import sys

from lm_eval.tasks.ifeval import instructions_registry as R

TYPES = ["combination:repeat_prompt", "combination:two_responses", "language:response_language",
         "length_constraints:nth_paragraph_first_word", "detectable_format:constrained_response"]


def main():
    src, out = sys.argv[1], sys.argv[2]
    n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    rng = random.Random(11)
    random.seed(11)  # the registry draws its random arguments from the global generator
    qs = [json.loads(line) for line in gzip.open(src, "rt")]
    rows = []
    for i, q in enumerate(rng.sample(qs, n_per * len(TYPES))):
        iid = TYPES[i % len(TYPES)]
        inst = R.INSTRUCTION_DICT[iid](iid)
        question = q["problem"]
        if iid == "detectable_format:constrained_response":
            question = "Is the answer to the following problem an even number? " + question
        if iid == "combination:repeat_prompt":
            desc = inst.build_description(prompt_to_repeat=question)
        elif iid == "length_constraints:nth_paragraph_first_word":
            n_par = rng.randint(2, 5)
            desc = inst.build_description(num_paragraphs=n_par, nth_paragraph=rng.randint(1, n_par))
        else:
            desc = inst.build_description()
        kwargs = {k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None}
        rows.append({"id": f"ifguard_{i}", "source": "ifguard", "task": "ifeval", "answer": None,
                     "problem": question + " " + desc, "instruction_id_list": [iid], "kwargs": [kwargs],
                     "from": q["id"]})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} items -> {out}")


if __name__ == "__main__":
    main()
