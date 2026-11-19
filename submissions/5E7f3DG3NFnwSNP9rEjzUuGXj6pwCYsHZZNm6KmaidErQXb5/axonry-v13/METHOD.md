# Make the model finish the way it already can: self-distillation into a few hundred MLP neurons

## The problem
After the parent edit (loop removal), the model still loses most of its points in two ways, and both show that the
ability is there:
- **Inconsistency.** The same problem is solved in one sample and lost in another: across the four practice areas,
  the share of problems solved at least once in 4 samples is far above the accuracy of a single sample.
- **Answers that run out of tokens.** Many lost samples are long, wandering reasoning that never reaches an answer
  within the budget, on problems that other samples of the same model finish correctly and much sooner.

Measure both on practice problems only (never the benchmarks), with the starting model (the parent edit built on
the base), from its dev run in step 8: per area, accuracy, accuracy when finished, the share cut at the token limit,
and the share of problems answered right in some samples and wrong in others ("mixed"). The method applies when
answers cut at the limit and mixed problems are common and the finished answers are far more accurate than all
answers; if not, say so in the report.

## Why we think it happens
The model's own sampling distribution puts too much weight on continuations that lead away from a solution it can
reach: when it hesitates, it keeps re-deriving instead of concluding. A short, successful sample of a problem is
direct evidence of a better way the model itself already produces: reach the answer, state it, stop. Raising the
likelihood of the model's own **short** right answers should move every similar problem toward finishing; what is
taught is a style of answering, so it should carry over across areas.
Testable: the gain must hold on problems never used for training (held-out practice problems), and the same training
on all right answers regardless of length must gain clearly less (if it gains as much, conciseness is not what
helps: report it). In development both held: +1 to +2 points without the length limit, +4 to +7 with it.
Which neurons carry the change matters less than the data: gradient-ranked and randomly spread neurons both worked.

## Approach
Everything below runs in the sandbox (4 GPUs), with the code in `submission/derive/` (copy `submission/` to
`/workspace/work/`). Work in `/workspace/work`; `$D = /workspace/work/submission/derive`.

1. **Starting model and an honest check set.** Copy the bundle's `parent/edit/` to the sandbox as
   `/workspace/work/parent/edit/`, then
   `python -m wsp.case init main --profile qwen35-0.8b-think --parent /workspace/work/parent/edit/edit.json
   --practice gsm8k_train math_train ifsynth mmlupro --dev 128 --holdout 256 --subset dev128=dev:128`.
   The starting model is `python -c "from wsp import case; print(case.model_path('main', 'base'))"` (`$START`).
2. **Training problems,** never in dev or holdout:
   `python3 $D/rollouts.py pick --spec gsm8k=2000,math=2000,mmlupro=all,ifsynth=all --exclude
   $WSP_CASES/main/sets/dev.json $WSP_CASES/main/sets/holdout.json --out ids.json`.
3. **Short rollouts of the starting model** on all 4 GPUs: `python3 $D/rollouts.py run --model $START --ids
   ids.json --out roll --caps gsm8k=4608,math=4608,mmlupro=4608,ifsynth=4608 --more 0` (2 samples per problem,
   each stopped at 4,608 tokens: only short answers are kept for training, so longer generations would be wasted;
   ~5,600 problems cost ~50M tokens, about 30-40 minutes). Read `roll/summary.json`: per area accuracy and cut
   share at this budget (most math answers do not finish in 4,608 tokens: that is expected).
4. **Training traces: the model's own short, clean right answers only** (`python3 $D/traces.py select --rows
   roll/rows.jsonl --out train.jsonl --max-chosen-tokens 4000`):
   - per problem, its **shortest clean right answer** (correct, finished, thinking closed, no loop-like tail), and
     only if it is **at most 4,000 tokens** long: the behavior to teach is reaching the answer and stopping, so
     the long right answers are left out on purpose (in development, the same training without this limit gained
     about a quarter as much, and a 6,000-token limit about two thirds as much);
   - problems right in every sample count as anchors (at most 25% of the examples): they keep what already works;
   - problems never solved are dropped; multiple-choice problems right only once are dropped (a single right
     answer out of 4 with 10 options can be a lucky guess; keeping them lowered the gain in development).
   Check `train.stats.json`: a few hundred examples are enough (development: ~230 and ~680 examples gave the same
   effect); many come from the instruction-following area, where short right answers are common, and the effect
   still carries over to math and multiple choice. Keep one limit for all areas: allowing longer instruction-
   following answers (8,000 tokens) gained less in development.
5. **Which neurons.** A neuron of an MLP is one row of `gate_proj` and of `up_proj` (what it reads) and one column of
   `down_proj` (what it writes); an edit can rewrite whole rows and columns (`set_rows`, `set_cols`). Rank all
   neurons of the decoder layers by the size of the accumulated gradient of the training loss over a calibration
   subset, never the last layer (playbook: its MLP is entangled with next-token prediction), and keep as many as the
   payload limit allows (about 650 in fp16 for 4 MB).
6. **Train only those** (`torchrun --standalone --nproc_per_node 4 $D/train_sparse.py --model $START --data
   train.jsonl --out-spec $WSP_CASES/main/edits/E1.json --parent-ref ../parent/edit.json --epochs 2 --accum 8`):
   the mean token log-likelihood of the chosen traces, the EOS included (finishing is part of the behavior); every
   row or column may change by at most half its norm; the result is rounded to bf16 exactly as the build will do
   and the held-out loss is measured again. The log says which layers the chosen neurons sit in and how the
   held-out loss moved: it must go down.
7. **Control** (if time allows): the same training on traces selected without the length limit
   (`--max-chosen-tokens 0`) as `R1`. It should gain clearly less than E1.
8. **Measure** on dev128 with 2 samples, paired with the starting model: `python -m wsp.case run main base dev128
   --n 2`, then the same for `E1` (and `R1`), one GPU each. Per area gain and SE are in
   `$WSP_CASES/main/results/E1__dev128.json`.
9. **Choose.** Upload E1 when its dev gain is positive overall, no area is below -3 points, and it beats R1 (when
   run). Otherwise lower the learning rate (`--lr 5e-5`) when an area dropped, or raise it (`--lr 3e-4`) when the
   held-out loss barely moved, and train once more.
10. **Package and upload.** `python3 $D/edit_tools.py package --spec $WSP_CASES/main/edits/E1.json --out up/E1
    --report REPORT.md`, then `axonry-check up/E1/edit --task task.json`, then `python3 $D/edit_tools.py upload
    --dir up/E1 --task task.json`. Use this packaging, not `wsp.upload`: the toolkit's packaging leaves out the
    arrays of set_rows / set_cols ops and such an upload fails admission.

## Time plan (6 hours, 4 GPUs)
- 0:00-0:20 set up: copy the bundle, init the case, build the starting model, pick the training problems.
- 0:20-1:00 short rollouts (step 3); meanwhile write the measurements of step 1 into the report.
- 1:00-1:30 traces and training (steps 4-6).
- 1:30-2:20 dev check of E1 (step 8); **upload the first E1 that passes step 9 at once.**
- 2:20-5:00 the control (step 7) and at most one retrained variant; re-upload only a confirmed better edit.
- From 30 minutes before the deadline: no new experiments; final `axonry-check` and upload.

## What to expect
The held-out loss of the chosen traces goes down during training, the share of answers cut at the token limit
drops sharply (development: about 21% to 11% on the practice dev set), and accuracy rises in math, gsm8k and
multiple choice (development: +7 points overall on 256 practice problems x 4 samples, +9 to +12 in those three
areas, instruction following within noise). A higher learning rate (3e-4) on unfiltered data cost instruction
following: keep 1e-4. The control without the length limit gains clearly less.

## What to report (REPORT.md)
The step-1 measurements per area; the rollout summary; the trace counts per area and type; the chosen neurons per
layer; the held-out loss before, after training, and after bf16 rounding; the dev results per area with paired
standard errors for E1 (and R1); what was changed between attempts and why; everything that did not work.
