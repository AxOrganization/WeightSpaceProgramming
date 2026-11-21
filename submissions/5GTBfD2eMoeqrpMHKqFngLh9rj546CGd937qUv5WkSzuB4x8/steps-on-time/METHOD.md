# Steps that decide: close when done, stop repeating, and credit the thinking steps that make a problem solvable

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## Time rules for this method (read first)
- **No command may run longer than 40 minutes.** One background job is cut at 60 minutes; every step below is sized
  to stay well under that, and the long one (step 5) is resumable: run it again and it continues where it stopped.
- **Write PROGRESS.md after every numbered step** (what finished, the numbers, the next command). If you are asked to
  continue, read PROGRESS.md first and go on from there.
- **Upload early:** a safe interim edit is uploaded at step 4; the full edit replaces it at step 8.
- Start each long command detached (`nohup ... > log 2>&1 &`) and check its log every few minutes; never wait on a
  command for more than 15 minutes without looking at its progress lines.

## The failure that is left
Two things still cost the parent's line on practice problems:
- **Long math thinking runs to the token limit.** On `math_train` practice problems many answers are cut at the 32k
  limit; many of these traces re-check a result they have already stated, or repeat their own earlier text.
- **On word problems, the answer is often decided by a few steps of the thinking.** On mixed `gsm8k_train` problems
  (some samples right, some wrong) the traces differ in a handful of steps: one sample sets up the quantities
  correctly and the rest follows, another makes the wrong choice early and never recovers.

The method has three parts. They touch different tensors (or the same ones in ways that commute) and stack.

## Part A: close when done (conclude direction, attention 15 and 19)
The toolkit's `conclude` contrast (`wsp/directions.py`) on the parent's own text: take a right, finished answer and cut
it right after the thinking first states the right result. The good branch closes the thinking (`</think>` and the
model's own final answer); the bad branch keeps verifying (the model's own next text). The two branches share the
prefix, so the per-layer mean difference of the residual stream isolates the keep-verifying state. Project it out of
the attention outputs (`self_attn.o_proj`) of layers 15 and 19 (`dir_ablate`), strength 0.75 or 1. If the parent's
chain already removes a conclude direction at these layers (look for `dir_ablate` ops on layers 15 and 19 in the
`edit.json` files under `parent/`), keep the candidates without Part A in the search as well (they are listed below).

## Part B: stop repeating (loop direction, attention 11 and 23)
The toolkit's `loop` contrast on the same answers: cut a right answer at a sentence boundary; the good branch is the
model's own continuation, the bad branch repeats the model's earlier text, as a loop does. Project that direction out
of the attention outputs of layers 11 and 23 at strength 1.

## Part C: credit the steps whose finishing value jumps
For each of the parent's own traces on a mixed practice problem, measure the **value of the thinking so far** at
checkpoints (every 512 tokens of the first 2048): let the model continue the thinking from that point 4 times, with
the normal sampling settings and a budget of 6144 tokens, and grade the answers. v(T) = share right. The budget makes
the value mean "solvable from here, in time".

Between two checkpoints the value either stays, rises or falls. The segments where it changes by at least 1/2 are the
decisive steps: a rise means that stretch of thinking made the problem solvable, a fall means it lost it. Each such
segment becomes a training row: the prompt is the question plus the thinking up to the segment (no loss there), the
target is the segment's own text, and the weight is the value change (positive or negative). Everything else gets no
credit. This is a policy-gradient step with a measured advantage on each step of the thinking, instead of one reward
for the whole answer.

The parameters are per-unit gains of the MLPs (`scale_rows` on `up_proj`) and per-head gains of the attention
outputs (`scale_cols` on `o_proj`), layers 4 to 21. The fit learns log-gains; the edit applies a dose of them (0.5:
half the learned log-gains). The values come from full continuations, not from forcing an answer: forced answers at a
cut of the thinking are a poor estimate of where the thinking will end up (see "What does not work").

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`. `prun.py` is `wsp.case run` split over all GPUs (the
same seeds and files, N times faster); `kinds.py` shows where the answers go (right, finished wrong, cut).

| step | what | about (4x RTX 4090) |
|---|---|---|
| 1 | case, parent on dev128 | 20 min |
| 2 | the parent's own answers | 15 min |
| 3 | directions, grid | 10 min |
| 4 | screen Parts A+B, interim upload | 40 min |
| 5 | step values (resumable) | 30-40 min |
| 6 | rows, fit, join | 10 min |
| 7 | screen and confirm | 80 min |
| 8 | final upload | 10 min |

1. **Case and the parent's starting run** (practice only):
   `python -m wsp.case init co --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`.
   Check that the parent chain is complete (`parent/edit/` and its own parents). Then
   `python derive/prun.py co base dev128 --n 4` and `python derive/kinds.py co dev128`.
2. **The parent's own answers** on fresh practice problems, one run per GPU i of 4 (all four in parallel):
   `python derive/sample.py co base ANS --source gsm8k_train:512 --source math_train:128 --n 4 --max-tokens 8192 --shard i/4`.
   These problems are outside dev and holdout.
3. **Directions.** With `W=$WSP_WORK/cases/co`:
   `python derive/pairs_more.py co --kind conclude --sources gsm8k math --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out concl`, then
   `python derive/pairs_more.py co --kind loop --sources gsm8k math --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out loop --compare concl_dirs.npz`.
   Go on only if both are stable at their layers (split-half stability well above 0.9) and far from parallel
   (|cosine| well below 0.5). Then
   `python derive/grid.py build co --dirs $W/concl_dirs.npz --sets 15,19 --alphas 0.75 1 --prefix C` and
   `python derive/grid.py build co --dirs $W/loop_dirs.npz --sets 11,23 --alphas 1 --prefix L`, and the Part A+B
   stacks: `python derive/concat.py co CL_075 C15-19_a0.75 L11-23_a1` and `python derive/concat.py co CL_1 C15-19_a1 L11-23_a1`.
4. **Screen Parts A+B and upload an interim edit.** `python derive/prun.py co <edit> dev128 --n 4` for `CL_075` and
   `CL_1` (and `L11-23_a1` alone if the parent already removes a conclude direction); rank with
   `python derive/grid.py table co dev128`. Upload the best one that beats the parent (step 8), as the interim result.
5. **Part C, values** (resumable). `python derive/pick_traces.py co TR.jsonl ANS.*.jsonl --sources gsm8k math --problems 220 --per 2`
   (one right and one wrong trace per mixed problem). Then on each GPU i of 4, in parallel:
   `python derive/step_values.py co base TR.jsonl --out VAL --step 512 --maxt 2048 --k 4 --budget 6144 --shard i/4`.
   Each prints a progress line per batch and appends its rows as it goes; if a run is cut, run the same command again
   and it continues. Done when every shard has printed `STEP_VALUES_DONE`.
6. **Part C, rows, fit and join.** `python derive/step_rows.py co ROWS.jsonl VAL.*.jsonl --traces TR.jsonl --jump 0.5`
   (expect 250-500 rows, both signs). Fit on one GPU (a few minutes), with `MP` = the parent model directory
   (`python -c "from wsp import case; print(case.model_path('co', 'base'))"`) and S = 2 x rows / 8:
   `python derive/gains_fit.py train $MP ROWS.jsonl FIT --heldout ROWS.held.jsonl --rank 0 --unit-gains --gains --gain-layers 4 21 --lr 2e-3 --accum 8 --steps S --warmup 5 --max-tokens 512 --eval-every 25`,
   then `python derive/gains_fit.py export FIT co STEPS_05 --model $MP --scale 0.5`. Join:
   `python derive/concat.py co CLS_075 C15-19_a0.75 L11-23_a1 STEPS_05`,
   `python derive/concat.py co CLS_1 C15-19_a1 L11-23_a1 STEPS_05` and
   `python derive/concat.py co LS L11-23_a1 STEPS_05` (without Part A).
7. **Screen and confirm.** `prun.py ... dev128 --n 4` for `CLS_075`, `CLS_1` and `LS`; rank them together with the
   step-4 results. Run `base` and the two best on `holdout256 --n 4` (one command at a time, about 25 minutes each). Keep
   the candidate with the best holdout gain among those that beat the parent on both dev128 and holdout256, and never a
   candidate with a practice source clearly worse than the parent. One split is noisy: a candidate that is clearly
   positive on holdout256 but slightly negative on dev128 can still be kept if no candidate beats it on both.
8. **Upload.** Write REPORT.md: the cut shares and kinds per source, both directions' stability and their cosine, the
   number of mixed problems, traces, rows (up and down) and the fit's held-out loss, the screen table and the holdout.
   Package without uploading: `python -m wsp.upload co <edit> --report REPORT.md --dry-run`; check it:
   `axonry-check $WSP_WORK/deliver/co__<edit>/edit --task task.json`; then the same upload command with
   `--task task.json` instead of `--dry-run`. Upload at step 4 the same way, then replace it here.

## What our exploration found (practice splits only)
We ran these parts on the line this entry stacks on (0009547524-0008 and its descendant 0009550031-0005), with the
toolkit's sampling and grading, paired with the starting model on the same problems and seeds (4 answers each).
Support for what to expect, not a fixed answer: the search decides.
- **All three parts** (conclude 0.75, loop 1, Part C at dose 0.5): on 0009547524-0008, dev128 +2.7, holdout256 +1.5,
  three further mixed splits +3.2 / +2.1 / +1.2, two 512-problem `gsm8k_train` splits +4.2 and +2.7, a 256-problem
  `ifsynth` split +2.5, three 384-problem `math_train` splits +1.6 / +1.4 / +2.0. On 0009550031-0005, the second
  `gsm8k_train` split +2.5 and the `ifsynth` split unchanged (that model already follows instructions far better).
- **Parts A and B without Part C:** about half the word-problem gain (+2.2 and +1.3 on the two `gsm8k_train` splits).
- **Control:** the same Part C weights on random segments of the same traces, joined with Part A: dev128 +2.5 against
  +2.9, the second `gsm8k_train` split +1.5 against +2.8, holdout256 +0.2 against +1.4 for the real rows.
- **A full run of these steps on 4 GPUs** (our own dry run on 0009547524-0008): the best joined candidate gave dev128
  +2.1 and holdout256 +1.8; Part A + Part C alone gave holdout256 +2.8. One 128-problem dev split can be misleading by
  several points in either direction, which is why step 7 confirms on holdout256.

## What does not work (measured on practice; do not spend the run on it)
- **A higher dose of Part C:** dose 1 or 2 costs instruction following and math on 0009547524-0008, and adds no
  word-problem gain over 0.5 or 0.75 on 0009550031-0005.
- **Values from forced answers** (close the thinking at the checkpoint and sample short answers) instead of full
  continuations: the rows cost instruction following at dose 1 and above, and add less at 0.5.
- **Training the model to close where its forced answer is already right:** it learns to close after any line, and
  accuracy collapses (-20 points and more).
- **Part A at four attention layers (11, 15, 19, 23)** or at strength 1.25 and above: no better on average.
- **Long single commands:** a step-value run over all traces at once takes more than an hour; split it (step 5 is).

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: the directions, the values, the rows and the gains are all measured and fitted in the run.
