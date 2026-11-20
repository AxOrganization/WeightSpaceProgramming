# Exact answers: the box in the reference form

**Resuming?** Run `python derive/gpu.py status` in the sandbox (in `/workspace/work`, with the two exports of
section 3), then read `/workspace/work/PROGRESS.md`. Never start a second copy of a job that is running, never start
GPU work outside `derive/gpu.py launch`, and upload only with the commands of step E.

This method adds one layer to the strongest published start (the entry's parent or the current champion, chosen on
practice data in step A): learned `dir_ablate` directions with one strength per matrix and direction, the
parameterization of the published edit 0009538688-0021 (a finished entry this method builds on; `derive/fit_dirs.py`
is based on its fit code). The layer is fitted to change one thing: how the model writes its final answer when it
has already found the right value. No arrays are shipped: everything comes from the toolkit's practice sets and the
start model's own generations.

All figures below are the author's own measurements in a rehearsal of this method on 4090s, on practice problems
only (start: the parent named in `entry.json`; paired with the start by problem and seed; "exact" as in section 1).

## 1. The behaviour to change, and why the model shows it

**Right value, wrong form.** The prompt asks for the final answer "within \boxed{}". `gsm8k_train` writes every
reference as a bare integer. The start often boxes the right value in another form: `\boxed{26.00}`,
`\boxed{60\%}`, `\boxed{\$720}`, `\boxed{8 \text{ cm}}` for 26, 60, 720, 8. Compared with the reference after the
toolkit's own light normalisation (`wsp.grade._norm`), these are wrong; only the toolkit's symbolic fallback
(math-verify) accepts them. An exact reader, such as an answer key, rejects them: the right behaviour is the bare
form of the reference. `derive/exact.py` grades that way ("exact") and classifies every miss.

On 1424 fit problems x 4 samples (8192-token cap), the start's `gsm8k_train` answers were 60.8% right by the
toolkit's grader and 58.6% exact: 2.1% of all answers (34) had the right value in a decorated form (trailing zeros
15, units 9, percent 5, currency 5). `math_train` had no such plain-number miss; its 4.0% of other accepted forms are
mostly the references' own idiosyncrasies (`45` for `45^\circ`, `40` for `40\text{ min}`, `\frac{2}{11}` for
`\dfrac{2}{11}`, a different factor order): no convention the model could learn, and they contradict the
`gsm8k_train` one. They are not repaired (see "What did not work").

**It is a habit, not an accident of the sampler.** The profile samples with presence penalty 1.5, so the penalty
was the first suspect. At the first token where a repaired answer departs from the original (the `.` of `26.00`, the
`\%`), the start preferred the decoration over the reference's next token in 22 of 30 answers without the penalty,
and in the same 22 with it (the penalty flipped 1 of 30: `fit_dirs.py --diagnose`). Self-distillation fits keep the
habit: they grade with the lenient checker, so decorated boxes come back as positives.

**Two ways to fit it, both candidates.** Fitting every token of the start's correct and repaired answers (thinking
and answer, "trace" scope) moved the habit: on 448 practice problems (dev128 + devB + hold192, 2 samples each) the
two such fits gained +0.9 +- 1.5 and +0.9 +- 1.4 exact points, exact `gsm8k_train` went up on every split (by up to
+12.5) with the lenient score flat, while `tuluif` was down 5-6 points on devB and the holdout for one of them.
Fitting the answer after `</think>` only, with the thinking held to the start ("answer" scope), is the other way:
its fit with the thinking tether at weight 5 gained +0.6 +- 1.4 on the same 448 problems, with exact `gsm8k_train`
+1.6 / +9.4 / +6.2 on dev128 / devB / hold192, `mmlupro` -4.7 / +1.6 / 0.0 and `tuluif` +3.1 / -4.7 / -2.1 (for
comparison, the better whole-trace fit: `gsm8k_train` +4.7 / +7.8 / +9.4, `tuluif` 0.0 / -6.2 / -6.2).
More repaired answers help: with 1200 `gsm8k_train` fit problems instead of 400 (105 repaired answers instead of
30), the pooled exact gain on the same 448 problems rose to +1.8 +- 1.4 for a trace-scope fit (from +0.9) and to
+1.1 +- 1.4 for the answer-scope fit (from +0.6), with exact `gsm8k_train` +6 to +11 on devB and the holdout; `tuluif`
still lost 4-8 points on some splits. This method samples that many. The run fits both kinds and its own dev splits
decide (step F): per-source differences of this size are within the noise of a few hundred problems, and the two
kinds trade the size of the fix against how much else moves.

## 2. The edit

Per decoder layer, two small orthonormal sets of r = 16 learned directions in the residual stream: one shared by the
matrices that read from it (`gate_proj`, `up_proj`), one by those that write into it (`down_proj`, `o_proj`,
linear-attention `out_proj`), and one strength alpha per (matrix, direction), starting at 0. With orthonormal
directions the ops on a matrix are exactly `W (I - Q diag(alpha) Q^T)` (reading) or `(I - Q diag(alpha) Q^T) W`
(writing), which the training forward computes; the spec is that list of `dir_ablate` ops. Weight decay pulls every
alpha to 0, so the fit uses the smallest change that explains the data.

The fit (`derive/fit_dirs.py`) uses the start model's own practice answers (`derive/build_data.py`):

- **pos**: finished answers right *in the exact form*, up to two per problem, at random: next-token loss;
- **fix**: answers whose box held the right value in a decorated form, with only the box content rewritten to the
  reference (`derive/exact.repair`; the thinking and the prose stay the model's own): next-token loss with
  1 + `--fix-weight` its weight on the answer, and unlikelihood on the decoration's first token (the `.` of `26.00`,
  the `\%`);
- **scope**: `trace` labels every token and holds the answer tokens of positives to the start (KL, `--tether`);
  `answer` labels only the answer after `</think>` and holds 256 random thinking positions of every answer to the
  start instead: the edit may change how the answer is written, not how the problem is solved.

Every log-probability is taken under the sampler's penalised distribution, `log softmax(z - P*seen)[y]` with seen =
generated earlier in the same answer (prompt tokens do not count, as in vLLM's presence penalty): the answers were
sampled that way, and fitting them with the plain loss teaches the penalised choices as raw preferences, which
generation then penalises again. In the rehearsal the same whole-trace fit with the plain loss (control `Z1`) lost
3.9 +- 3.0 exact points on dev128 where the two penalised fits scored +1.6 and -0.4.

Candidates (step D): T1 and T2 (trace scope, seeds 1 and 2) and A1 (answer scope, tether weight 5). Control
`SHUF` (the chosen fit with its alphas permuted within each matrix) separates the learned assignment from a
perturbation of the same size; in the rehearsal it lost 2.3 +- 2.7 points (-6.2 on `gsm8k_train`).

## 3. Setup (first 10 min)

On the agent machine, copy the task file, this submission's code and both published starts to the sandbox (tar over
ssh if scp is slow; there is no rsync):

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r parent sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r champion sandbox:/workspace/work/      # if the bundle has champion/
```

In the sandbox, in every shell:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
SRC="gsm8k_train math_train mmlupro tuluif"
SETS="--dev 256 --holdout 384 --subset dev128=dev:128 --subset hold192=holdout:192 --seed 7"
python -m wsp.case init P --profile qwen35-0.8b-think --parent parent/edit/edit.json --practice $SRC $SETS
python -m wsp.case init K --profile qwen35-0.8b-think --parent champion/edit/edit.json --practice $SRC $SETS
python -c "from wsp import case; case.model_path('P','base'); case.model_path('K','base')"   # build both ONCE
```

(Skip K everywhere if the bundle has no `champion/`; then S = P.) Both cases have the same dev and holdout
problems. Write the first line of PROGRESS.md; keep notes in `/workspace/work/notes.md`.

Every GPU job goes through `python derive/gpu.py launch --gpu N --name NAME -- <command>`: it waits until that GPU
has no process left (a vLLM engine still shutting down makes the next one fail), runs the job detached in its own
process group with `CUDA_VISIBLE_DEVICES=N`, and keeps its log and exit code in `/workspace/work/jobs/`.
`gpu.py status` shows everything; `gpu.py wait NAME...` blocks at most 8 min; `gpu.py stop NAME` kills a job with
all its processes. Check every 8-10 min and update PROGRESS.md; never `pkill -f` (it matches your own shell). Use
only `practice:` data.

## 4. Steps

**A. Choose the start, and start sampling (0:10-0:35).**

```
G=derive/gpu.py
PS="gsm8k_train=1200 math_train=480 mmlupro=240 tuluif=240 ifsynth=64"
python $G launch --gpu 0 --name devP -- python -m wsp.case run P base dev128 --n 2
python $G launch --gpu 1 --name devK -- python -m wsp.case run K base dev128 --n 2
python derive/sets.py --case P --out /workspace/work/fitP --shards 4 --per-source $PS
for s in 2 3; do python $G launch --gpu $s --name genP$s -- python -m wsp.gen --profile qwen35-0.8b-think \
  --model $WSP_WORK/models/P__base --ids /workspace/work/fitP/fit_ids_$s.json --n 4 --salt fit --max-tokens 8192 \
  --out /workspace/work/fitP/gen_$s.jsonl; done
```

When devP and devK end (about 15 min): `python derive/exact.py summary $WSP_WORK/runs/P/base__dev128.jsonl
$WSP_WORK/runs/K/base__dev128.jsonl`. **S** is the case with the higher exact accuracy over all sources; within one
point, the one with fewer answers cut (rehearsal: P 63.7 exact, 5.9% cut; the then champion 60.9, 13.3%). If S = P,
launch shards 0 and 1 the same way on GPUs 0 and 1. If S = K, stop genP2/genP3, run `python derive/sets.py --case K
--out /workspace/work/fitK --shards 4 --per-source $PS` and sample all four shards from `$WSP_WORK/models/K__base`
into `/workspace/work/fitK`. Below, FIT is that directory. Record the choice in notes. A shard (556 problems x 4) takes
about 12-15 min.

**B. Base runs for the paired comparisons (as GPUs free up):** `python $G launch --gpu N --name devB -- python -m
wsp.case run S base devB --n 2`, then `hold -- ... run S base hold192 --n 2` (about 10 min each).

**C. Diagnose and build the training set (2 min), then the presence probe (1 min, any free GPU).**

```
python derive/build_data.py --case S --runs $FIT/gen_*.jsonl --out $FIT/train.jsonl
python $G launch --gpu 0 --name probe -- python derive/fit_dirs.py --case S --data $FIT/train.jsonl --diagnose
```

`build_data.py` prints and saves (`train.jsonl.diag.json`) per source: lenient vs exact accuracy, the share of form
misses and their kinds, the share of other accepted forms (not repaired), cuts, what was kept. Never pass `--broad`.

**D. Fits on GPUs 1-3 (trace scope about 37 min, answer scope about 32 min on this data).**

```
F="python derive/fit_dirs.py --case S --data $FIT/train.jsonl --max-minutes 40"
python $G launch --gpu 1 --name T1 -- $F --out $FIT/T1 --seed 1 --epochs 1.5
python $G launch --gpu 2 --name T2 -- $F --out $FIT/T2 --seed 2 --epochs 1.5
python $G launch --gpu 3 --name A1 -- $F --out $FIT/A1 --seed 1 --epochs 2 --scope answer --tether 5
```

Each fit logs the loss, the unlikelihood term, the tether KL and the held-out NLL; `state.pt` is written at the end
(at `--max-minutes` at the latest). Export each as it ends: `python derive/fit_dirs.py --case S --export
$FIT/T1/state.pt --name T1` (likewise T2, A1), and start its dev128 run at once (step F1) on the GPU it freed.

**E. First upload, as soon as the first candidate is exported (by about 1:15).**

```
E=A1                                          # the first exported candidate
python derive/report.py --case S --chosen $E --diag $FIT/train.jsonl.diag.json --fits $FIT/$E \
  --notes /workspace/work/notes.md --out /workspace/work/REPORT.md
python -m wsp.upload S $E --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/S__$E/edit --task task.json        # must say passed: true, or do not upload
python -m wsp.upload S $E --task task.json --report /workspace/work/REPORT.md
```

The upload references the start it stacks on (`parent/edit.json` or `champion/edit.json`); `wsp.upload` writes it so.
In the rehearsal `axonry-check` passed with 96 tensors, a 2.4 MB payload and a global relative change of 0.007.

**F. Select (pre-registered; about 50 min).**
1. Run T1, T2 and A1 on dev128 (`wsp.case run S <edit> dev128 --n 2`, one per GPU; about 15 min with the model
   build), then `python derive/exact.py paired $WSP_WORK/runs/S/base__dev128.jsonl
   $WSP_WORK/runs/S/{T1,T2,A1}__dev128.jsonl --out /workspace/work/res/dev128.json`.
2. Eligible: exact gain over all sources minus its SE > 0, and no source with an exact gain below minus its own SE.
3. The two eligible with the highest exact gain run on devB (`run S <edit> devB --n 2`; paired file `res/devB.json`).
   The choice is the one with the higher pooled exact gain:
   `python derive/exact.py pooled /workspace/work/res/dev128.json /workspace/work/res/devB.json`.
   One eligible: it. None eligible: the candidate with the highest exact gain on dev128, said plainly in the report.
4. Re-upload (step E commands) if the choice is not the uploaded edit.
5. Holdout only confirms: `run S <choice> hold192 --n 2`, paired file `res/hold192.json`. It never changes the choice.
6. Control: export `SHUF` (`--export $FIT/<choice>/state.pt --name SHUF --shuffle-alphas`) and run it on dev128.

**G. Second round (only if at least 2 h remain when the choice is made).** Sample the chosen edit on shards 0 and 1
of FIT (`--model $WSP_WORK/models/S__<choice>`, out `$FIT/r2_0.jsonl`, `$FIT/r2_1.jsonl`), rebuild the training set
from all generations (`--runs $FIT/gen_*.jsonl $FIT/r2_*.jsonl --out $FIT/train2.jsonl`), and continue the chosen
fit with its own flags: `--data $FIT/train2.jsonl --out $FIT/R2 --init $FIT/<choice>/state.pt --seed 4 --epochs 1`.
Export R2 and run it on dev128 and devB. R2 replaces the choice only if its pooled exact gain is higher; then
re-upload and run its holdout.

**H. Deliver.** Final REPORT.md with `report.py` (all `res/*.json`, all fits, notes: the start chosen and why, what
did not work). The final upload is done by T-45 min at the latest. After the verified final upload, if no step
still running can finish before T-45 min, stop all jobs (`gpu.py stop`) and run `axonry-done`; the GPU clock then
stops.

## Timeline (6 h, 4 GPUs; durations measured in the rehearsal on 4090s: vLLM start 1-2 min; at the 8k cap a shard of
356 mixed problems x 4 in 7-8 min, 282 `gsm8k_train` problems x 4 in about 9 min; dev128 x2 of a start 12-14 min, of an
edit 13-16 min with its build; on about 3300 training answers a trace-scope fit about 85 steps/min (1.5 epochs in
about 37 min), an answer-scope fit about 125 steps/min (2 epochs in about 32 min); an export 2 s; axonry-check 30 s)

| time | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| 0:00-0:10 | setup, two cases, build both starts | | | |
| 0:10-0:25 | P on dev128 x2 | K on dev128 x2 | sample shard 2 | sample shard 3 |
| 0:25-0:40 | sample shard 0 | sample shard 1 | S on devB x2 | S on hold192 x2 |
| 0:40-0:45 | build_data, probe | | | |
| 0:45-1:25 | | fit T1 | fit T2 | fit A1 (ends ~1:15) |
| 1:15-1:30 | first upload (A1) | | | A1 dev128 |
| 1:25-1:45 | | T1 dev128 | T2 dev128 | SHUF of the leader, dev128 |
| 1:45-2:00 | | top-1 devB | top-2 devB | |
| 2:00-2:15 | choice holdout, re-upload | | | |
| 2:15-3:45 | | round 2: sampling, R2 fit, R2 dev128 + devB | | |
| by 4:15 | final upload, `axonry-done` | | | |

## What did not work (rehearsal)

- **Repairing every accepted form** (`build_data.py --broad`: also the "other accepted forms" of `math_train`, with
  the same unlikelihood): all three such fits collapsed, -33 to -35 exact points on dev128, `gsm8k_train` from 75% to
  3-17%. In those repairs the original's first departing token is often the box's own closing brace (`45` closed where
  the reference goes on with `^\circ`), so the unlikelihood taught the model not to close its boxes. `fit_dirs.py`
  now never puts unlikelihood on such a repair, and the method does not build them.
- **A stronger dose**: an answer-scope fit exported with every alpha x 1.5 scored -0.5 +- 1.5 pooled (-10.9 on
  `mmlupro` and -9.4 on `tuluif` on devB). Export at dose 1.0.
- **The plain next-token loss** (control `Z1`, section 2): -3.9 +- 3.0 on dev128.

## What to expect, and what to watch

- The gain shows on the exact score of `gsm8k_train` (+6 to +11 points on the rehearsal's devB and holdout), with the
  lenient score flat or slightly up; `tuluif` is the source to watch (the rehearsal's fits lost 4-8 points on some
  splits): the selection rule rejects a candidate that loses beyond its SE on any source, and the report says how
  `tuluif` moved for the uploaded edit.
- The fit's alpha rms stays near 0.07 and the global relative change far below the limit (`axonry-check` prints it).
- If form misses are rare on the start (`train.jsonl.diag.json`), the layer has little to learn: say so in the
  report; the selection rule of step F still decides which candidate is uploaded.
