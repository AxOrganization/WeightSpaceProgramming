# conclude-then-dead-ends: the parent's answers keep their instructions; this entry makes its thinking end

## The idea
This entry stacks on its parent (`entry.json`, sid `0009550031-0005`), the published answer-branching edit: on the
champion's model it changed how the **answer** is written once the thinking is over, so that the answer keeps the
constraints of the request. It did not change **when the thinking ends**. On the parent's model the answers to hard
math problems still run into the 32k-token limit and score nothing (18.9% of its practice MATH answers in our runs,
against 17.3% for the champion it stacks on).

Two published remedies exist for when the thinking ends. Both were measured on older models, and neither is in the
parent's lineage (`0009550031-0005` on `0009547524-0008` on `0009538688-0021` on `0009531781-0011`):

1. **Conclude** (`0009538689-0017`): after the thinking has stated its result, the model writes one more check, and
   another. The toolkit's `conclude` contrast measures that state on the model's own text; the edit removes it from
   the attention output of four layers.
2. **Dead ends** (`0009549068-0003`, built on conclude): a run that will never finish is recognisable in the last
   layer's MLP input at the same depth as a productive one. A gated unit of the last layer reads that state and
   raises the end-of-thinking token there.

This entry measures both **again, in the run, on the parent's own model**, and stacks them: first the conclude edit,
then, on the model that results, the dead-end unit. Our exploration on this lineage (practice sets only, below) adds
what the run needs to know:

- The two remedies add up, and where. The conclude edit shortens every answer and rescues a part of the cut ones.
  The dead-end unit then closes runs that are still cut, most of them between 20,000 and 28,000 thinking tokens, and
  their answers are right as often as those of runs that end there by themselves (about four in ten).
- **Depth is the wrong trigger.** Units that close by the length of the thinking (a ramp from 6,000 tokens) close as
  many cut runs and gain nothing: they also end runs that would have ended right. The model cannot read its own
  depth well enough either: positions beyond 26,000 tokens could not be told from positions before 20,000.
- **The threshold decides.** With the gate's threshold at the 99.9% quantile of the negative positions the unit is
  almost never on and does nothing; at the 97% quantile it works best on both models we tried. The run reads this
  from a replay before any generation (step 5).
- **How to measure it.** A run that is cut in one sampling finishes in about half of the re-samplings of the same
  problem. "Rows that were cut and now finish" is therefore no evidence for an edit; only the paired total over
  many problems is, and a replay of the start's own runs through the edit shows directly what the unit does.
- **How much the run's own checks can tell.** We built the whole stack twice on the parent, independently (other
  splits, other problems, other seeds). The two models gained the same on 376 unseen MATH problems (+3.9 +- 1.1 and
  +3.8 +- 1.2), while the replay on the 50 held-out dead runs printed 96% for one and 69% for the other, and a
  32-problem screen cannot see two points at all. The rules of steps 5 and 7 are set by that: they stop a unit that
  is broken, and they do not ask small sets for more than they can show.

If the run's own measurements do not show these two failures on the start (step 1 and step 5 say how to tell), the
method does not apply: say so in the report.

## Why the model shows it
*Conclude.* After the thinking first states the result, the attention layers write a state that makes the next
sentence another check instead of the close. The `conclude` contrast (`wsp/directions.py`) takes a correct, finished
answer and cuts it right after the thinking first states the right result. From there the good branch closes the
thinking (`</think>` and the model's own final answer) and the bad branch is the model's own next text. Per layer,
`u = mean(residual | keep verifying) - mean(residual | close)`. The edit removes `u` (`dir_ablate`, side `out`,
alpha 1) from what the attention writes (`self_attn.o_proj`) at layers 11, 15, 19 and 23.

*Dead ends.* What is left after that are runs that never settle: they re-derive, doubt a result they had, try the
next reading of the problem. Ending the thinking is one event, the end-of-thinking token, and in such a run its
logit lies far below the top token at every line (15 to 18 nats in our measurements), so the run cannot leave by
chance. The state itself differs from that of a productive run: a linear read of the last layer's MLP input, fitted
on **how runs end** (line ends of runs that never finish, past 12,000 thinking tokens, against every position of
runs that end right, long ones included, every answer position and all early thinking), is on again and again in
dead runs and rarely in runs that end right at the same depth, on problems held out from the fit. A unit of the last
layer acts on its own position's next token only, so it can raise the closing token there and nothing else.

## What we measured (practice sets only)
All numbers are from the toolkit's practice sets and the models' own generations: 376 MATH-train, 373 GSM8K-train
and 256 MMLU-Pro practice problems and 412 `tuluif` requests that no fit saw, 4 answers per problem, model-card
sampling, paired by problem and seed.

**The parent against the champion it stacks on** (`0009547524-0008`): instruction following 78.5% -> 88.8%
(+10.3 +- 1.3), GSM8K +2.2 +- 0.9, MMLU-Pro +1.0 +- 1.4, MATH 60.3% -> 60.1% (-0.2 +- 1.1). Its math answers are cut
as often as the champion's (18.9% against 17.3%): the parent changed the answers, not the thinking.

**Both remedies, on the champion's model** (where we explored first; MATH, 376 problems, 4 answers each):

| model | MATH correct | MATH answers cut |
|---|---|---|
| champion (two samplings) | 60.3, 61.0 | 17.3%, 18.0% |
| + conclude (six samplings) | 62.0 to 63.2, mean 62.6 | 12% |
| + conclude + dead-end unit, threshold 99% (two samplings) | 64.5, 63.9 | 7.3%, 6.2% |
| + conclude + dead-end unit, threshold 97% (two samplings) | 65.3, 64.2 | 2.5%, 2.7% |

Paired, same problems and seeds: the unit (99%) against conclude +1.6 +- 1.1 and +1.7 +- 1.0; the unit (97%) against
conclude +2.4 +- 1.0 and +2.1 +- 1.0; conclude with the unit (97%) against the champion +5.0 +- 1.2 and
+3.2 +- 1.0. With the unit, the answers that finish between 20,000 and 28,000 tokens grow from 12% to 16-17% of all
answers and are right as often as before (four in ten). The unit does not move the other sets: instruction
following -0.7 +- 1.0, GSM8K -1.1 +- 0.9 (inside the spread of five samplings of the model without it), MMLU-Pro
+0.2 +- 1.3.

**Both remedies, on the parent's model, by the steps below** (a rehearsal of steps 1 to 5 as written):
- conclude: 128 pairs from 92 answers, stability 0.97 to 0.98 at layers 11, 15, 19 and 23; the direction is the
  champion's (cosine 0.99 at layers 11, 15 and 19, 0.95 at layer 23). Instruction following +0.2 +- 0.8 against the
  parent: the parent's gain stays.
- 640 math problems x 2 answers with the conclude edit: 185 runs never finish (14%).
- `fit` (threshold 97%), share of held-out line ends with the gate on, by thinking tokens so far:

  | thinking tokens | below 6k | 6-12k | 12-16k | 16-20k | 20-24k | 24-28k | 28k and more |
  |---|---|---|---|---|---|---|---|
  | dead runs | 0.6% | 6.6% | 13.7% | 24.1% | 31.5% | 45.9% | 50.1% |
  | runs that end right | 0.0% | 0.5% | 4.1% | 3.2% | 10.5% | 3.0% | |

- `spec`: the closing token lies 15 to 18 nats below the top token where the gate is on; push 19.7 nats; the other
  tokens' distribution moves by 0.03 nats (KL).
- `replay` on the held-out runs: the edit ends the thinking 2,000 tokens or more before the run's own end in 96% of
  the dead runs (median: after 17,600 thinking tokens) and in 3% of the runs that end right.
- The model that results (`CD`), full practice sets, against the parent: MATH 60.1% -> 64.0% (+3.9 +- 1.1; answers
  cut 18.9% -> 2.7%), instruction following +1.1 +- 0.8, GSM8K +1.3 +- 1.0, MMLU-Pro -1.8 +- 1.4. Conclude alone:
  MATH 61.7% (+1.6 +- 1.2, cut 14.0%).
- **A second build by the same steps**, independent of the first (another dev split and so other pairs, 640 other
  math problems, other seeds): 128 pairs, stability 0.98 at the four layers; 196 of 1,280 runs never finish (15%);
  `replay` on the held-out runs: the edit ends the thinking 2,000 tokens or more before the run's own end in 69% of
  the dead runs (median: after 18,900 thinking tokens) and in 6% of the runs that end right. The model that
  results: MATH 60.1% -> 63.9% (+3.8 +- 1.2 against the parent; answers cut 18.9% -> 3.3%; +2.2 +- 1.2 against the
  model with the conclude edit alone, the first build +2.3 +- 1.0). The replay numbers of the two builds differ by
  more than their models do: 50 held-out dead runs and 150 right ones are a small sample, and each replay number
  carries several points of chance.

**The same ops carried from one model to the other do not work**: the unit fitted on the champion's model with
conclude, written onto the parent with conclude, cuts as few answers (3.9%) but gains only +2.6 +- 1.1 on MATH,
because it also ends runs early that would have ended right (answers below 12,000 tokens: 92% -> 88% right). The
unit reads the model's own state: it must be fitted on the model it sits on, which is what the steps do.

**The threshold** (replay on held-out runs: share of runs the edit ends 2,000 tokens or more before their own end;
MATH: the stack against the same model with conclude only, one number per sampling):

| threshold (quantile of the negatives) | champion's model: dead / right runs ended | MATH | parent's model: dead / right runs ended | MATH |
|---|---|---|---|---|
| 99.9% | 32% / 1% | +0.3, -0.3 | | |
| 99% | 58% / 2% | +1.6, +1.7 | 67% / 1% | +1.3, +0.5 |
| 97% | 74% / 4% | +2.4, +2.1 | 96% / 3% (second build: 69% / 6%) | +2.3 (second build: +2.2) |
| 95% | | +1.4 | 98% / 5% | +2.1 |

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`, `python -m wsp.gen`), `axonry-check`, and this
entry's `derive/` scripts (copy `derive/` to the sandbox and run them as `python derive/<script>` from the work
directory): `pairs_src.py` is the script of the published entry `0009538689-0017`, used as it is; `deadend.py` and
`stack.py` are ours (each prints its usage with `-h`). Every GPU job uses one GPU: put `CUDA_VISIBLE_DEVICES=<n>`
in front, start it detached (`nohup ... &`), one job per GPU, and read its log when it is done. Never stop jobs by a
name pattern; keep the PID. Times are for one RTX 4090.

**0. Start.** `task.json` lists the starts. Build on the one with role `parent` (the entry's parent); if the task
offers only one start, take it. Below `<P>` is its id. Copy `<P>/edit/` from the bundle to the sandbox, next to
`task.json`, and read its `REPORT.md`.

**1. Cases and the start's own answers.**
```
python -m wsp.case init cc --profile qwen35-0.8b-think --contrast conclude --parent <P>/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256
python -m wsp.case init cif --profile qwen35-0.8b-think --parent <P>/edit/edit.json --practice tuluif ifsynth --dev 256
python -c "from wsp import case as C; print(C.model_path('cc', 'base')); print(C.model_path('cif', 'base'))"
```
The last line builds both starting models once, before any run uses them. Then, in parallel:
```
CUDA_VISIBLE_DEVICES=0 python -m wsp.case run cc base dev --n 2         # about 45 min; the conclude pairs come from it
CUDA_VISIBLE_DEVICES=1 python -m wsp.case run cc base dev128 --n 4      # about 25 min
CUDA_VISIBLE_DEVICES=2 python -m wsp.case run cif base dev --n 4        # about 20 min
CUDA_VISIBLE_DEVICES=3 python -m wsp.case run cc base holdout256 --n 4  # about 45 min; used only in step 7
```
Run `python -m wsp.case report` and read `$WSP_CASES/cc/RESULTS.md`: accuracy and the share of answers cut. The cut
answers should be almost all MATH.

**2. The conclude contrast on the start** (after `cc base dev`):
```
python derive/pairs_src.py cc --sources gsm8k math --sets dev --exclude dev128 holdout256 --n 128
CUDA_VISIBLE_DEVICES=0 python -m wsp.case extract cc
python -m wsp.case edit cc --layers 11 15 19 23 --alpha 1 --targets o_proj --name C1
python -m wsp.case edit cc --layers 11 15 19 23 --alpha 1 --targets o_proj --random-control --name RAND_C1
S=$(python -c "from wsp import case as C; print(C.model_path('cc', 'C1'))")
```
`pairs_src.py` builds the pairs of the case's contrast from correct, finished GSM8K and MATH answers only (the
other sources have no result that can be found in the thinking, and the toolkit's own `pairs` stops on them). Go
on with 60 pairs or more. `extract` prints stability and separation per layer (layers are counted from 0): go on if
the stability at layers 11, 15, 19 and 23 is 0.9 or more. `C1` removes the direction from the attention output of
those four layers: 4 tensors, 16 KB of payload. `$S` is the model with `C1`: the dead-end unit is fitted on it.

**3. Upload early.** Write a short `REPORT.md` (what is targeted, how the direction was measured, the pair count
and stabilities), then
```
python -m wsp.upload cc C1 --report REPORT.md --task task.json --dry-run
axonry-check $WSP_WORK/deliver/cc__C1/edit --task task.json
python -m wsp.upload cc C1 --report REPORT.md --task task.json
```

**4. Runs of the model with `C1`** (all four GPUs, about 30 min). The dead-end read needs runs that never finish,
and they are rare: sample math problems the case does not use.
```
python derive/deadend.py pick --case cc --n 640 --parts 4 --out ids
CUDA_VISIBLE_DEVICES=0 python -m wsp.gen --profile qwen35-0.8b-think --model $S --ids ids_0.json --n 2 --salt dd --out dd_0.jsonl
```
and the same for `ids_1.json`, `ids_2.json`, `ids_3.json` on GPUs 1, 2 and 3. When they are done:
```
CUDA_VISIBLE_DEVICES=0 python -m wsp.case run cc C1 dev128 --n 4
mkdir -p $WSP_CASES/cif/edits && cp $WSP_CASES/cc/edits/C1.* $WSP_CASES/cif/edits/
CUDA_VISIBLE_DEVICES=1 python -m wsp.case run cif C1 dev --n 4
```
(the specs point to the same start, so they are copied into `cif` as they are). Count the runs that never finish:
`python -c "import json,glob; r=[json.loads(l) for f in glob.glob('dd_*.jsonl') for l in open(f)]; print(len(r), sum(x['cut'] for x in r))"`.
With fewer than 60 of them, sample another 640 problems the same way (`pick --seed 1 --out ids2`, `--salt dd2`,
`--out dd2_<i>.jsonl`) and use both sets below.

**5. The dead-end unit** (GPU 2; the read takes about 15 min, the rest minutes):
```
D128=$(python -c "from wsp import case as C; print(C.run_path('cc', 'C1', 'dev128'))")
CUDA_VISIBLE_DEVICES=2 python derive/deadend.py read --model $S --rows dd_0.jsonl dd_1.jsonl dd_2.jsonl dd_3.jsonl $D128 --out $WSP_WORK/de_reads.npz
python derive/deadend.py fit --reads $WSP_WORK/de_reads.npz
CUDA_VISIBLE_DEVICES=2 python derive/deadend.py spec --reads $WSP_WORK/de_reads.npz --model $S --case cc --name DE
python derive/stack.py cc CD --edits C1 DE
E=$(python -c "from wsp import case as C; print(C.model_path('cc', 'CD'))")
CUDA_VISIBLE_DEVICES=2 python derive/deadend.py replay --model $E --rows dd_0.jsonl dd_1.jsonl dd_2.jsonl dd_3.jsonl --held-only
```
(`$D128` adds runs of the other practice sources, so that their thinking and answers are negatives too.) `read`
stores the last MLP's input at thinking positions, half of them at line ends, and at answer positions. `fit` and
`spec` hold out 30% of the problems (a fixed hash of the problem id). `fit` prints, on those, the share of line
ends with the gate on, by depth, in dead runs and in runs that end right: it should be near zero below 6,000
thinking tokens, grow with depth in dead runs, and stay several times lower in right runs at the same depth. `spec` writes
the unit as the edit `DE` of the case, on the two least-used units of the last layer that no edit of the start's
chain has written; its dose is the gap between the closing token and the top token at the gate's own positions in
dead runs. `stack.py` puts the ops of `C1` and `DE` into one edit, `CD` (10 ops on 7 tensors). `replay` passes the
start's held-out runs through the model with `CD` and prints where the edit would end the thinking.

**The replay decides whether the unit is used.** Keep the unit as built (threshold 97%) when the replay ends the
thinking 2,000 tokens or more before the run's own end in at least 50% of the held-out dead runs and in at most 10%
of the held-out runs that end right. Both of our builds were inside these bounds (96% / 3% and 69% / 6%) and gained
the same on MATH; the bounds are wide because the held-out sample is small. Only outside them change the threshold,
once: fewer than 50% of the dead runs -> build `DE` again with `--q 0.95`; more than 10% of the right runs -> with
`--q 0.99` (then `CD` and the replay again). If the replay is still outside the bounds, leave the unit out and
deliver `C1` by step 7. Do not tune the threshold on anything else: the screens of step 6 cannot tell 97% from 95%.

**6. Screen `CD`.** One job per GPU:
```
CUDA_VISIBLE_DEVICES=0 python -m wsp.case run cc CD dev128 --n 4
cp $WSP_CASES/cc/edits/{DE,CD}.* $WSP_CASES/cif/edits/
CUDA_VISIBLE_DEVICES=1 python -m wsp.case run cif CD dev --n 4
CUDA_VISIBLE_DEVICES=2 python -m wsp.case run cc RAND_C1 dev128 --n 4
```
Run `python -m wsp.case report`; `python -m wsp.case compare cc base CD dev128` and `... compare cif base CD dev`
print the gain by source. `dev128` has 32 problems per source: it can show a loss, it cannot show a gain of two
points on one source. Read a few answers of runs the unit closed (long thinking, then a short answer): the answer
should state a result and keep the task's format.

**7. Choose, confirm, deliver** (on these practice numbers only).
- `C1` qualifies if its gain on `dev128` is not below -2 SE and its gain on the `cif` dev set is not below -2 SE
  (in our rehearsal it gained 1.5 points on `dev128` and 1.1 +- 0.9 on `cif`; a loss of one SE on 128 problems
  happens by chance in one run of six).
  If it does not qualify, the conclude state is not what holds this start back: say so in the report and upload
  nothing new.
- With `C1` qualified, the replay rule of step 5 decides for `CD`. The screens can only veto it: deliver `C1`
  instead if `CD` is below `C1` by more than 4 points on `dev128` or by more than 3 points on the `cif` dev set
  (twice the SE of those differences; in our rehearsal `CD` was 0.5 and 0.6 points below `C1` on them, inside
  the noise, while it gained two points of MATH on 376 problems).
- The random control is a check of the measurement, never a candidate.

Then `python -m wsp.case freeze cc <choice>`, `CUDA_VISIBLE_DEVICES=0 python -m wsp.case run cc <choice> holdout256
--n 4`, and upload the choice (dry run, `axonry-check`, upload, as in step 3) with every measured number in
`REPORT.md`: both dev sets by source, the control, the holdout, the pair count and stabilities, the `fit` table, the
`spec` report and the replay. Keep the last 40 minutes for this upload. With time left after it, nothing else in this
method needs it: finish.

## What to expect
`C1`: answers get shorter (a fifth to a third fewer tokens on GSM8K and MMLU-Pro), the share of answers cut falls
by about a third, `dev128` moves by +1 to +3 points with an SE of 2. `CD`: the share cut falls to a few percent; on `dev128`
the pooled gain over `C1` is small and inside its SE, of either sign (32 math problems cannot show two points); the
replay is the evidence that the unit does its job: it ends the thinking early in most held-out dead runs (70% to
95% from build to build), at 12,000 to 28,000 thinking tokens, and in a few percent of the runs that end right.
Instruction following (`cif`) stays where the start has it, within noise: the unit is silent below 6,000 thinking
tokens, and instruction answers think for 3,000. The holdout should repeat the `dev128` picture within its SE.

## What does not work on this lineage (measured; do not spend the run on it)
Paired with the model named, on the practice sets above (4 answers per problem):
- **Closing by depth.** End-of-thinking units gated by the length of the thinking (the published `0009538706-0022`
  as it is, ramp from 2,500 tokens: instruction following -3.2 +- 1.1 on the champion; the same units re-fitted
  with a ramp from 6,300 tokens, on top of the conclude edit: MATH cut 12% -> 5% and MATH +0.0). They end runs that
  would have ended right. A gate for "near the limit" could not be built at all: thinking positions beyond 26,000
  tokens do not separate from positions before 20,000.
- **A penalty on the "Wait" opener** on top of the conclude edit (decoding test, the toolkit's `penalize`): MATH
  -0.8 +- 1.0 at 2 nats and -1.9 +- 1.1 at 4: fewer cut answers, more wrong ones.
- **A larger conclude dose** (alpha 1.5 on MATH; alpha 1.5 and 2 and all six attention layers on instruction
  following and GSM8K): no better than alpha 1 at the four layers.
- **The dead-end unit at a cautious threshold** (99.9% quantile of the negatives, as the published entry starts):
  on this lineage it is on too rarely to end a run (MATH cut 12% -> 11%, +0.3 +- 1.0 and -0.3 +- 1.0 in two builds);
  at 99% it ends only two in five of the dead runs in real sampling (MATH cut 14% -> 8.5% on the parent).
- **More token budget is not what is missing.** With a 46,000-token limit the model with the conclude edit gains
  2.5 points of MATH, and the runs that end between 32,000 and 46,000 tokens are right three times in ten: late
  runs are hard problems, and thinking faster would not rescue most of them.
- **Units for single instructions** (no commas, all capitals) fitted on small sets: +18 to +25 points on their own
  requests and -2.9 +- 1.1 over all 412 requests. Not in this entry.
- **Lower sampling temperature or no presence penalty** are no levers for math or multiple choice (within +-1 at
  temperature 0.3 to 0.45; MMLU-Pro -1.7 to -2.1), and the final norm leaves no way to write them into the weights.

## Rules kept
Only the toolkit's practice sets and the models' own generations are used, in our exploration and in the run. The
submission ships no arrays: the direction and the unit are measured in the run on the start's own model. Nothing is
downloaded.
