# Sibling preference on residual channels: prefer the model's own finishing answer to its own run-on answer

This entry stacks on the published self-distillation edit (`entry.json` `parent`). It adds one new layer of
**residual-channel gains**: for each block of layers 4-21, how strongly the attention block and the MLP block read
each of the 1024 residual channels, and how strongly each writes them back. The layer is written with `scale_cols`
(on every matrix that reads a block's normed input, with the same factors) and `scale_rows` (on the block's output
matrix). It does not touch per-neuron or per-head scales.

The gains are fitted to a **preference between two answers to the same problem, both sampled from the starting
model**: the one that closed its thinking and answered correctly, and a sibling that was still deliberating at the
token limit (or that finished wrong). They are not fitted by imitating correct answers.

The edit is derived during the run from the starting model's own samples on practice problems. No arrays are
shipped. The numbers in section 4 are the author's local measurements, for orientation only.

## 1. The behaviour, and why a preference

The parent finishes more often than its own parent did, but answers are still lost at the token limit: in a local
rebuild of the parent, 30% of MATH-train answers and 13% of all practice answers were still cut at 32k tokens. For
many of these problems another sample of the same problem finishes correctly. Measure this first on the starting
model (step A, `derive/sibpref.py stats`): per practice source, the cut share and the share of cut answers whose
problem has a finishing correct sibling. If almost no cut answer has such a sibling (under a fifth on every source),
the method has little to learn from; report that.

Making correct answers more likely does not, by itself, make the run-on continuation less likely where the two
compete. A preference objective uses both answers of the same problem, and it compares them over the **same number
of tokens**: the length the finished one needed. At that point the finished sibling has closed its thinking and
stated its result, while the cut one is still going. The same prompt on both sides cancels topic, difficulty and
format, so the fit learns what separates closing from continuing in the model's own states. Finished-but-wrong
siblings are used as rejected answers too, so the preference never rewards closing on a wrong result.

**Why residual channels.** The quantity being re-weighted is how much each block takes from and adds to each
direction of the residual stream. One gain per channel and block is about 74k parameters for layers 4-21:
- the read gains of a block act exactly like rescaling that block's normed input;
- the write gains act exactly like rescaling that block's contribution, channel by channel.

This is a small, smooth family. It can only re-weight what the starting model already computes, and it cannot add
content. In local tests it was also much less sensitive to the dose than a per-neuron family on the same pairs
(section 4).

Objective per pair (`derive/sibpref.py fit`):

    loss = -log sigmoid(beta * [r(chosen) - r(rejected) - anchor * max(0, -r(chosen))]) + nll * NLL(chosen)

- r = mean over the answer tokens of (log p_edit - log p_start). The start model's log-probs are computed once,
  before training (all gains at 1).
- `anchor` (DPO-Positive style) stops the fit from buying margin by lowering the chosen answers as well. Without
  it, a single fit on the loop-removal model lost 2.5 points; with it, the same fit gained 5.5 (section 4).
- Three fits on different 80% subsamples of the pairs are averaged in log space, which cancels fit noise. The
  averaged change is then dosed with a factor k, chosen on dev.

## 2. Setup (first 15 min)

On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r parent sandbox:/workspace/work/      # -> /workspace/work/parent/edit/edit.json
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init sp --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512 \
    --subset dev128=dev:128 --subset holdout256=holdout:256
python derive/pairsets.py --case sp --out /workspace/work/sp --shards 3
python -c "from wsp import case; case.model_path('sp','base')"     # build the starting model ONCE, before any GPU job
```

Start GPU jobs only after the last command has finished: parallel jobs would race to build the same directory.

Rules for all long work:
- Run every GPU job detached (`tmux new -d -s NAME "..."` or `nohup ... &`), each with its own
  `CUDA_VISIBLE_DEVICES=i`. Run one job per GPU, each under 60 min.
- Check the jobs every 10-15 min (`tail` the logs, `nvidia-smi`) and keep PROGRESS.md current.
- Never `pkill -f` / `pgrep -f` a pattern: it matches your own shell. Stop jobs by the PID you saved.
- Use only `practice:` data and the starting model's own generations. Never load `bench:` sets.

## 3. Steps

**Step A: starting-model baseline and sibling samples (in parallel, about 45 min)**

```
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run sp base dev128 --n 2 > base_dev.log 2>&1 &
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s+1)) nohup python -m wsp.gen --profile qwen35-0.8b-think --model $WSP_WORK/models/sp__base \
    --ids /workspace/work/sp/train_ids_$s.json --n 4 --salt sp --max-tokens 12000 \
    --out /workspace/work/sp/gen_$s.jsonl > /workspace/work/gen_$s.log 2>&1 &
done
```

- `pairsets.py` draws 1024 practice problems (320 GSM8K train, 320 MATH train, 192 MMLU-Pro practice, 192 ifsynth).
  They are disjoint from the case's dev and holdout splits, and from the 1024 problems the parent was itself fitted
  on: the parent's published `make_sets.py` draws those deterministically on the same split, and `pairsets.py`
  reproduces that draw exactly. On those problems the parent repeats the answers it was trained on, and pairs from
  them pulled instruction-following down in the author's rehearsal (see section 4). `sibpref.py` drops any dev or
  holdout id again.
- The 12k-token cap only shortens sampling: an answer still open at 12k tokens counts as a run-on sibling.
- When the samples are in, run `python derive/sibpref.py stats --case sp --runs /workspace/work/sp/gen_*.jsonl`.
  Write in NOTES, per source: accuracy, cut share, recoverable share of the cut answers, the problems that give
  pairs, and `short` (finished answers under 200 tokens). In the author's runs about 300 of the 1024 problems gave
  pairs (about 600 pairs).
- `short` matters. On a few instruction-following prompts the model now and then answers with an empty thinking
  block (`</think>` first, 20-90 tokens in all). If such an answer were a chosen answer, its pair would be compared
  over a few dozen tokens and would teach one thing only: skip thinking. In one of the author's runs two such pairs
  among about 600 were enough for the fit to stop thinking on ifsynth (370 instead of 5500 tokens per answer,
  ifsynth -19 points). `sibpref.py` therefore never uses a finished answer under `--min-chosen-tokens 200` as
  chosen. Keep that default.

**Step B: three preference fits (about 25 min, GPUs 1-3 in parallel)**

```
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python derive/sibpref.py fit --case sp --runs /workspace/work/sp/gen_*.jsonl \
      --name P$i --seed $i --subsample 0.8 > fit_P$i.log 2>&1 &
done
# when all three are done:
for k in 2.0 2.5 3.0; do python derive/sibpref.py average --case sp --fits P1 P2 P3 --name S${k/./} --k $k; done
```

The defaults are the tested settings: layers 4-21, `--anchor 5 --beta 10 --nll 0.2 --lr 2e-3 --accum 8 --epochs 1
--pairs-per-problem 2 --pairs all`.

The log prints the pair count per source and the rejected kinds (cut / wrong). Every 10 steps it then prints:
- the loss;
- the mean margin;
- the mean log-prob change of the chosen and of the rejected answers.

The margin should grow and the chosen change should stay at or above zero. If the chosen change falls clearly below
zero (below about -0.01 nats per token), the fit is displacing the model's own good answers: raise `--anchor` to 10
and refit.

**Step C: admission check and early upload (by about 1 h 40 min)**

```
python -m wsp.upload sp S25 --dry-run --report /workspace/work/cases/sp/NOTES.md
axonry-check $WSP_WORK/deliver/sp__S25/edit --task task.json
python -m wsp.upload sp S25 --task task.json --report /workspace/work/REPORT.md
```

- Write REPORT.md first; without `--report`, `wsp.upload` needs `cases/sp/NOTES.md`.
- `edit/edit.json` must say `"parent": "parent/edit.json"`; `wsp.upload` writes it that way.
- The spec has a few thousand ops (each read factor is applied to every reader matrix) and a payload of about
  0.5 MB, far under the limits.

**Step D: dose on dev (about 35 min, one dose per GPU)**

```
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run sp S20 dev128 --n 2 > dev_S20.log 2>&1 &   # S25, S30 on GPUs 2, 3
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run sp base holdout256 --n 2 > hold_base.log 2>&1 &
```

- Read `cases/sp/results/<edit>__dev128.json`: `paired.gain`, `se`, cut share and `by_source`.
- Check the answer lengths: `python derive/sibpref.py lengths --runs $WSP_WORK/runs/sp/base__dev128.jsonl
  $WSP_WORK/runs/sp/S25__dev128.jsonl` (and S20, S30). Shorter answers are the intended effect. But if one source
  falls below half the starting model's tokens per answer, the fit has learned to skip thinking there. Then drop all
  doses, look for short chosen answers (`stats`), and refit.
- Keep S25, the dose that was best on the author's holdout, unless S20 or S30 beats it on dev by more than one se.
  Whichever dose you keep must meet all of these:
  - gain - se > 0;
  - a cut share no higher than the starting model's;
  - no practice source down by more than about 5 points (32 problems per source, so look at the trend over doses).
- Dev reads high for every edit: all dev comparisons share one run of the starting model. Compare doses with each
  other on dev, and confirm on holdout.
- Re-upload as soon as a dose beats the uploaded one.

**Step E: confirm on holdout, control, deliver**

```
python -m wsp.case freeze sp <best>
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run sp <best> holdout256 --n 2 > hold_best.log 2>&1 &
python derive/shuffle_gains.py --case sp --edit <best>
CUDA_VISIBLE_DEVICES=2 nohup python -m wsp.case run sp SHUF_<best> dev128 --n 2 > dev_shuf.log 2>&1 &
```

- The control keeps the edit's factors and payload but puts each factor group on randomly drawn channels of the same
  tensors. It should score like the starting model. If it scores like the edit, the effect does not come from the
  learned assignment.
- If the holdout gain is negative, fall back to the next smaller dose that was positive on dev and confirm it.
- REPORT.md covers:
  - the diagnosis numbers;
  - pairs per source and rejected kinds;
  - the fit logs (margin, chosen change) and the gain ranges per group;
  - dev and holdout gains with se and cut shares;
  - the control;
  - what did not work.
- Upload the final edit. Run `axonry-done` if more than 45 min remain and nothing useful is pending. Otherwise stop
  GPU work at the 30-minute reminder and make sure the last complete upload is the best edit.

## Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | GPU 0 | GPUs 1-3 |
|---|---|---|
| 0:00-0:20 | setup, case init, pairsets, build the starting model | (idle) |
| 0:20-1:05 | starting model on dev128 x2 | sibling samples, 3 shards (12k cap) |
| 1:05-1:35 | starting model holdout256 x2 | fits P1 / P2 / P3, then average -> S20, S25, S30 |
| 1:35-1:45 | admission check + **first upload (S25)** | |
| 1:45-2:20 | (free) | dev128 x2: S20, S25, S30 |
| 2:20-3:00 | shuffle control on dev128 | holdout256 x2 of the dev winner |
| by 5:00 | final upload (REPORT.md), then `axonry-done` if nothing is running | |

## 4. What to expect

These are the author's local runs, for orientation only.

**Starting model.** A local rebuild of the parent from its published method and code: three fits on 1024 practice
problems x 4, log-averaged, k = 1.25. On 512 practice holdout problems x 4 it had 64.6% accuracy, and 13.1% of its
answers were cut at 32k tokens (MATH 29.9%).

**Data.**
- 1024 practice problems x 4 samples of the starting model at the 12k cap: about 20 min on two H200s.
- 313 problems gave 626 pairs: GSM8K 200, ifsynth 140, MATH 164, MMLU-Pro 122.
- Rejected answers: 61% cut, 39% finished wrong.
- Share of cut answers with a finishing correct sibling: GSM8K 51%, ifsynth 42%, MMLU-Pro 25%, MATH 19%.

**Fits.** About 10 min each on an H200 or an RTX 5090, and 22 min on an L40S. The margin rises to a few thousandths
of a nat per token, and the chosen answers' log-prob change stays positive.

**Holdout.** 512 practice problems never used for any choice, x 4 samples, paired with the starting model:

| dose k | all | GSM8K | ifsynth | MATH | MMLU-Pro | cut share | tokens / answer |
|---|---|---|---|---|---|---|---|
| 2.0 | +2.2 ± 1.0 | +1.0 | +0.8 | +7.0 | +0.2 | 13.1% -> 6.3% | 11.0k -> 8.5k |
| 2.5 | **+3.1 ± 1.0** | +2.5 | +1.0 | +6.2 | +2.7 | 13.1% -> 5.7% | 11.0k -> 8.2k |
| 3.0 | +3.0 ± 1.0 | +2.1 | +3.3 | +5.5 | +1.0 | 13.1% -> 4.7% | 11.0k -> 7.8k |
| 3.5 | +1.3 ± 1.1 | +0.8 | -2.9 | +5.3 | +2.1 | 13.1% -> 4.2% | 11.0k -> 7.3k |

- The gain comes from answers that no longer run to the limit. Most rescued answers are MATH problems that the
  starting model had cut.
- Dev (256 x 2) read 2-3 points higher than holdout for every edit, the control included, because one baseline run
  is shared by all dev comparisons.
- Control: the same factors on shuffled channels did not shorten the answers (10.6k -> 10.7k tokens per answer). It
  scored like the starting model within that dev offset (+2.3 ± 2.0 on dev, where the edit itself had +5.5).

**A full rehearsal of this method** (fresh samples, 4 x RTX 4090, the commands above) took 32 min for sampling and
18 min for the three fits. Without the short-answer rule, two pairs whose chosen answer had an empty thinking block
made the fit stop thinking on ifsynth: 370 instead of 5500 tokens per answer, ifsynth -19.3, all practice -3.8 on
holdout. With the rule (the same samples, refitted) the thinking stayed normal, but ifsynth was still down 6.2 (all
-0.3). Those samples also covered about a fifth of the ifsynth and MMLU-Pro problems the parent had been fitted on.
Leaving those out (without the rule) gave ifsynth -1.2 and all +0.7. The author's two best runs drew no such
problems and had no empty-thinking chosen answer: +3.1 and +2.1 on holdout (ifsynth +1.0 and +1.8, MATH +6.2 and
+4.5). Both safeguards are now the defaults.

**What did not help** (same pairs; holdout unless noted):
- Per-neuron and per-head gains with the same objective: +2.1 to +2.6, and more sensitive to the dose (k = 3.5:
  +0.2).
- No anchor: on the loop-removal model, a single unanchored fit (per-neuron gains, k = 2) lost 2.5 on dev, while
  the same fit with the anchor gained 5.5. On the starting model the unanchored per-neuron fits gave +1.8.
- Anchor 20 (per-neuron gains): weaker than anchor 5 on dev.
- Weighting the sources equally: +1.5 to +1.7.
- Twice the problems: +1.9 to +2.7.
- A second round, sampled from the edited model and fitted on top: only +0.3 to +0.5 more, with MATH up and
  MMLU-Pro down. It is not worth its time.
- Read gains alone (one fit): +3.1 on dev, against +5.9 for read and write gains together (three fits averaged).

## 5. What not to do

- Do not train on dev or holdout problems, and never on `bench:` data.
- Do not compare a cut answer over its full length with a short finished one. The comparison runs over the finished
  sibling's length, which is what makes the pair about closing rather than about length alone.
- Do not drop the anchor, and do not lower `--min-chosen-tokens`: a chosen answer without thinking makes the fit
  skip thinking.
- Do not chase a bigger effect with a higher learning rate: with one gain per channel, a higher rate mostly adds
  drift. Dose the averaged fits with k instead.
- Do not choose by teacher-forced numbers. Choose by paired generation on dev, and confirm on holdout.
