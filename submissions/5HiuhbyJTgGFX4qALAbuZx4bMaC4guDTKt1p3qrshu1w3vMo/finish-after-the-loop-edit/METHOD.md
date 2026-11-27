# Finish after the loop edit: remove the 2B's thinking loop, then teach it, from its own answers, to finish

## Credits
This entry is a transfer and a combination of finished, published work, with a few additions of ours.

- **Stage A, the loop edit**, is the toolkit's loop recipe (`playbook.md`, "The loop edit"; the toolkit's own transfer
  to this base in `cases/qwen35-2b-think_loop/NOTES.md`), first entered on the 0.8B line as **#2 "loop-removal"
  (0009531781-0011)**.
- **Stage B, the fit on the model's own answers**, is the method of **#8 "finish-from-own-answers" (0009538688-0021)**,
  which was built on #2's edit. Its scripts `derive/pgen.py`, `derive/traces.py`, `derive/fit.py`, `derive/compare.py`
  and `derive/sets.py` are reused here. Each file's first lines say what was changed:
  - `pgen.py` and `compare.py` are unchanged;
  - `fit.py` has 2B defaults (profile, payload);
  - `sets.py` has a wider practice mix;
  - `traces.py` has the `pick --by-cut` option (ours).
- **Ours:**
  - running both stages on this base inside one run's budget, with stage A's measurements reused as stage B's baseline;
  - the wider practice mix: puzzle and second instruction set next to maths, multiple choice and instructions;
  - weighting the fit by how often each task type fails to finish (`--by-cut`);
  - folding both stages into one spec on the base;
  - the floor-first upload rule below.

## The problem
In thinking mode, Qwen3.5-2B often does not reach an answer. Its reasoning falls into a loop: it repeats a block of one
to four sentences until the token limit, and the answer is missing. The answers that do finish are mostly right, so
the ability is there; what is lost is the ending.

Measure this first, on practice problems only:
- the share of answers cut at the limit;
- of those, the share whose tail repeats (`kind` = `cut_loop` in `wsp.grade`);
- the accuracy of finished answers against all answers, per task type.

If few answers are cut, or finished answers are no better than all answers, this method does not apply: say so and stop.

## Why we think it happens, and why two stages
1. **A loop is a state the model enters and then reinforces.** Once text repeats, repeating again becomes the likeliest
   continuation. On the Qwen3.5 models this state is carried by one residual direction, written mostly by a few MLP
   units in the middle layers (layers 15 and 19 of 24 on both sizes, per the toolkit's case). Removing what those
   units write along it removes most loops. That is stage A.
2. **What remains after stage A is not one direction.** At many layers, components tilt the next token a little toward
   "continue as before" (restart, re-check a settled result, drift) rather than "take the next step and conclude".
   The model's own generations hold the contrast without anyone naming it:
   - its finished, correct answers show what it should do;
   - the repeating tails of its own cut answers show what it should not.

   A fit that makes the first likelier and the second unlikelier, token by token, and is otherwise as small as
   possible, is stage B (#8's reasoning).
3. **We expect the loss to differ by task type.** We expect long puzzles and instruction-following requests to loop
   more often than short arithmetic, and their answers to be lost more often. Hence our two additions:
   - puzzle and instruction practice in every split;
   - `--by-cut`, which weights each task type's positives by its measured share of cut answers (clipped to 0.5 to
     2 times the mean), so the fit learns most where finishing fails most.

   Check this on the measurements of step A1. If the shares do not differ, drop `--by-cut`.

## Approach

### Stage A: the loop edit, derived in the run
Use the toolkit's case workflow (`python -m wsp.case -h`; `.claude/skills/weight-edit/SKILL.md`) on profile
`qwen35-2b-think`, starting from the base (`parent` null).

1. **Measure the start.** Create a case on the maths practice sources with a 128-problem dev subset. Generate the
   base's own answers on it at the profile's full token limit, and record the cut and loop shares and the finished
   accuracy (the measurement above).
2. **Find the direction.** Build contrast pairs from the base's own correct traces (`pairs`): same prefix, a looping
   continuation against the real one. Keep some pairs held out. Extract the directions and writer scores
   (`extract`). The direction must be stable: split halves must agree, with a cosine above about 0.9. The largest
   writers should sit in the middle layers. Leave the last layer alone, even if it writes most, because it is
   entangled with next-token prediction.
3. **Edit the writers.** Write `dir_ablate` specs on the top-64 MLP writers (`edit`):
   - the transferred recipe: layers 15 and 19, alpha 6;
   - a lower dose: alpha 3;
   - the norm-matched random control: `--random-control`.

   Pre-screen with `selectivity` on the held-out pairs. The loop continuation's log-probability must drop clearly
   while the normal one barely moves. An effect the random control also gives is a generic perturbation, not the loop.
4. **Choose on real generation.** Run each candidate on the dev subset at the full limit, paired with the base
   (`compare`). Take the smallest dose that removes most loops and loses on no task type.
5. **Upload the floor.** Freeze the chosen edit, check it with `axonry-check`, and upload it with the report as it
   stands. Every later step can only replace it with something measured to be better.

### Stage B: the fit on stage A's own answers (#8's method, starting model `S` = stage A's built model)

**B1. Splits.** Run `python derive/sets.py --out sets`. It writes three disjoint splits, each with every task type:
- `fit`: answers to learn from;
- `dev`: every choice;
- `hold`: one confirmation at the end.

If the run is short on time, use smaller splits (`--fit 300 --dev 64`).

**B2. Answers.** Start both jobs together, right after stage A is chosen; both are slow.
- (a) The baseline: `S` on `dev` at the full limit, 2 samples per problem, on one GPU. Every later comparison is paired
  with it.
- (b) `S` on `fit`, 4 samples per problem, with `--max-tokens 8192`, on the other three GPUs.

**B3. Select.** Run `traces.py pick --by-cut --loops 900`.
- Positives: finished correct answers, chosen at random rather than by length. Problems the model only sometimes
  solves count more.
- Negatives: cut answers whose tail repeats.

Read a few of each. After stage A the true loops may be few: a few dozen negatives are enough, and the positives
alone already teach the ending. Never fill the negatives with other cut answers. An answer cut while it was still
making progress is not a failure to unlearn.

**B4. Fit**, one fit per GPU, detached. The 2B needs about half of #8's `--budget` per batch on a 24 GB GPU (3000); use
`--max-steps` as the time cap.
- (a) MLP matrices and attention output projections only (`--targets down_proj o_proj out_proj gate_proj up_proj`);
- (b) all matrices;
- (c), when a GPU is free: (a) without the tether (`--tether 0`), to see what the tether protects.

Before generating, look at where the strength went and whether the fits agree on it. Build the control: the same
directions with the alphas shuffled (`--export ... --shuffle-alphas`).

**B5. Decide on `dev`**, at the full limit, paired with B2(a) (`derive/compare.py`).
- Fix the rule before reading the numbers: a task type down by more than one standard error is a loss to repair, not
  noise. Watch instruction following most: it has the least to gain from finishing.
- Among edits within noise of each other, take the one best on its weakest task type, then the one with fewer
  matrices.

**B6. Fold and upload.** The uploaded spec must start from the base, so fold the two stages into one spec:
- stage A's ops first, then stage B's ops, with their arrays in one npz under distinct names, and `parent` the pinned
  base;
- check that the folded spec builds the same weights as B stacked on A (largest absolute difference near 0);
- check that it is within the limits (8 MiB payload on this base; `fit.py` keeps B under 7 MB);
- run `axonry-check`, then upload it in place of stage A alone, with the report.

**B7. Confirm on `hold`** for the chosen edit if time is left, one sample per problem, paired with `S`.

## Time plan (4 GPUs, 6 h)
| When | Step |
|---|---|
| ~0:00–1:30 | Stage A: base dev run and pairs, then candidates, selectivity and dev runs in parallel |
| ~1:30 | Upload stage A |
| ~1:30–2:45 | B2 (a) and (b) in parallel |
| ~2:45–3:30 | B3–B4 |
| ~3:30–4:45 | B5 |
| by ~5:00 | B6 |
| after B6 | B7, if time is left |

If stage B is not clearly better than stage A on `dev` with no task-type loss, keep stage A. Say why in the report.

Rewrite the report after every measurement, so the upload never lags behind what is known. To stop a generation job,
kill its processes by PID and check `nvidia-smi`.

## What to expect
- **After stage A:** most loops gone; a large gain on every task type where loops were common; correct answers may get
  longer, because harder problems now finish. Accuracy over all answers is what matters.
- **After stage B:** fewer answers still cut at the limit; a further gain where stage A left the most cut answers; no
  loss where the model already finished nearly always. Some answers will now stop without a final answer where they
  used to loop. That is still a miss: report it by task type.
- **If something goes wrong:** if a task type loses, or the control gives the same gain, reduce the scope (fewer
  matrices, lower rank, stronger pull toward 0) or keep stage A alone. Say so in the report rather than upload a
  larger edit.
