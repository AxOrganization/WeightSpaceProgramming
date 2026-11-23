# Online RL on the champion's gains, with length-neutral credit

This entry stacks on the champion (`entry.json`: `"parent": "champion"`, the champion when the run starts). It runs
rounds of on-policy reinforcement learning on the start model's own answers: fresh practice problems every round,
eight answers each at the profile's decoding and token limit, a reward for a right and finished answer, and one
data-parallel update of about 70k gains per round. The edit is the gains of the last rounds, averaged in log space and
exported as `scale_rows` / `scale_cols` ops.

## 1. What fails, and why a plain policy gradient does not fix it

**The model knows more than it says once.** For roughly 40% of fresh practice problems, the start's eight answers
disagree: the right answer is in its distribution but is not drawn reliably. Reinforcement on its own answers can
move probability toward the answers that are right.

**A plain policy gradient learns to stop early.** Within a problem, the right answers are shorter than the wrong
ones, so the raw credit (RLOO on right-and-finished) pushes every trace to end sooner. On this model's gains that is
a strong, cheap direction. In our practice run one plain update (largest gain change 0.02) halved the thinking on every
source: word problems 7.0k -> 4.3k tokens, math 16.5k -> 8.7k, instruction requests 4.8k -> 2.7k. On the same fresh
problems it scored 11.7 points lower on word problems and 15.4 lower on math than the length-neutral update below.
Shorter thinking is not better thinking.

**Length-neutral credit.** Among the finished answers of a problem, the part of the reward that is linear in the
answer's length is removed (least squares within the group), and the residual credits what differs between answers of
similar length. Answers cut at the limit keep reward 0 and stay out of the fit: a fit through them would credit the
longest, looping answers (in our run that variant raised the cut share on word problems from 0.4% to 5.7% and on
multiple choice from 0% to 8.6% within three rounds). With this credit the lengths and cut shares stay at the start's
level round after round.

**What we measured on practice** (our run on 0009554110-0010, the champion when this was written; learning rate 2e-3,
10 rounds; edit = gains of rounds 8-10 averaged; paired with the start on fresh practice problems at the profile limit,
gain +- SE by problem):

| source | n | start | edit | gain |
|---|---|---|---|---|
| math_train | 300 x 4 | 66.5 | 72.0 | +5.50 +- 1.08 |
| gsm8k_train | 500 x 4 | 77.8 | 78.5 | +0.65 +- 0.89 |
| mmlupro | 200 x 2 | 41.8 | 44.5 | +2.75 +- 2.25 |
| ifsynth | 150 x 2 | 92.7 | 89.7 | -3.00 +- 1.65 |

A higher learning rate (3e-3) drifted to longer math thinking in its later rounds (cut share 4% -> 17%) and lost on both
screens, so the rate is 2e-3.

## 2. Parameters (`derive/rl_common.py`)
One gain per MLP unit (`scale_rows` on `mlp.up_proj`), per attention head (`scale_cols` on the head's column block of
the attention output: `self_attn.o_proj` or `linear_attn.out_proj`) and per key / value row (`scale_rows` on `k_proj`,
`v_proj` and the k, v rows of `linear_attn.in_proj_qkv`), layers 4-21, all starting at 1 on the start's weights. This
is the parametrisation of the published self-distill-gains `rft.py`. Gains are clamped to [0.7, 1.4].

## 3. One round (`derive/rl_loop.py`, `derive/rl_train.py`)
1. 160 fresh practice problems (gsm8k_train 64, math_train 64, tuluif 16, mmlupro 16), outside the case's dev and
   holdout and the screen sets, never reused.
2. 8 answers each from the current model at the profile's sampling and 32768-token limit (`derive/dtgen.py`; it writes
   the answers before it stops the engines, so an engine that hangs in teardown cannot lose a finished generation).
3. Reward 1 if right and not cut; the length-neutral residual among the finished answers; RLOO advantage.
4. Answers with a nonzero advantage train: loss = -sum A * log pi(token) / tokens of the minibatch, log pi at the
   profile's temperature, sequences cut to 16000 tokens; 4 minibatches = 4 Adam steps (lr 2e-3, state kept across
   rounds), data-parallel over the 4 GPUs (`torchrun`).
5. The next round's model is the start's weights times the gains (`derive/rl_build.py`).
A round takes about 18-24 minutes on 4 x RTX 4090. Every round's accuracy is measured on new problems, so
`rl/A/log.jsonl` is an honest learning curve (noisy: 64 problems per source).

## 4. Running it
Work in `/workspace/work`. Copy `submission`, `task.json` and the start's directory (`<sid>/` as `task.json` names it,
with everything it stacks on nested inside) there. Then, in `/workspace/work`:
```
E=$(python submission/derive/start.py)
python -m wsp.case init main --profile qwen35-0.8b-think --parent $E \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512
M0=$(python -c "from wsp import case; print(case.model_path('main', 'base'))")   # build the start once
python submission/derive/sets.py --case main --out sets
cp submission/derive/*.py .   # the loop calls its siblings by path
```
**The loop** runs as short jobs: each call stops after `--minutes` and the next call resumes from the last finished
round (gains, Adam state, problem positions). Run it detached with a log, check it every 10-15 minutes, and start the
next call when one ends:
```
nohup python rl_loop.py --tag A --gpus 0,1,2,3 --start $M0 --case main --kv --detrend --lr 2e-3 --minutes 40 \
    --exclude sets/scr.json sets/guard.json sets/ifs.json > rl_A.log 2>&1 &
```
Stop starting new calls once 3 hours of loop time have passed (about 9 rounds), or earlier if fewer than 2 hours of
the run would remain afterwards.

**First upload after round 4** (a complete upload is then always in place):
```
python rl_export.py --case main --tag A --name R4 --last 3
python -m wsp.upload main R4 --report REPORT.md --task task.json --dry-run    # then axonry-check, then the upload
```
**After the loop:**
1. `python rl_export.py --case main --tag A --name RL --last 3` (rounds n-2..n), and `--name RL125 --k 1.25`.
   `python rl_check.py $WSP_WORK/models/rl_A_<n> $(python -c "from wsp import case; print(case.model_path('main', 'RL'))")`
   confirms that the exported spec builds the trained model (quantisation differences only).
2. Paired screens, one model at a time on GPUs 0-3, same salt for every model (`base` is the start):
   ```
   for m in base RL RL125 R4; do
     MD=$( [ $m = base ] && echo $M0 || python -c "from wsp import case; print(case.model_path('main', '$m'))" )
     python dtgen.py --model $MD --ids sets/scr.json --n 4 --salt scr --out runs/${m}__scr.jsonl
     python dtgen.py --model $MD --ids sets/guard.json --n 2 --salt scr --out runs/${m}__guard.jsonl
     python dtgen.py --model $MD --ids sets/ifs.json --n 2 --salt scr --out runs/${m}__ifs.jsonl
   done
   ```
   (`base` and `RL` first; `RL125` and `R4` only while at least 75 minutes remain.)
3. `python pair.py table runs RL RL125 R4` and `python pair.py choose runs RL RL125 R4` (rule fixed below); upload the
   choice if it is not R4. Uploads: `python -m wsp.upload main NAME --report REPORT.md --task task.json --dry-run`,
   then `axonry-check` on the package's `edit/`, then the upload; one at a time, none with fewer than 10 minutes left.

**The choice** (`derive/pair.py`, fixed before the run): a candidate is eligible if neither screen source (gsm8k,
math) is below the start by more than max(1 SE, 1 point) and no guard source (ifsynth, mmlupro, tuluif) by more than
max(2 SE, 4 points); among the eligible, the largest score = word-problem gain + math gain; ties within 0.25 go to the
one listed first. With none eligible, the largest score.

**task.json** holds the upload credentials: keep its contents out of logs, PROGRESS.md and the report.

**At the end.** Finish `REPORT.md` (section 5), written to a temporary file and moved into place; upload the final
choice with the finished report no later than 20 minutes before the deadline; if `status` shows that upload and
nothing is left worth trying, run `axonry-done`.

## 5. REPORT.md
- The learning curve (`rl/A/log.jsonl`): per round, accuracy, cut share and mean tokens per source, mixed problems,
  training examples, largest gain change.
- The export: rounds averaged, ops, payload, the `rl_check.py` line.
- The screen table (`pair.py table`), the choice and its reason (`runs/choice.json`), what was uploaded and when.
- What was skipped and why.

## 6. Rules kept
Practice sets and the model's own answers only; choices on practice screens outside dev and holdout; holdout is not
used. The gains and every choice are measured in the run.
