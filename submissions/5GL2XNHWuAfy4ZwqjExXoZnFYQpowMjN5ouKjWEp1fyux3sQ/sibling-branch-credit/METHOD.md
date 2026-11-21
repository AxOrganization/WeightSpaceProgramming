# Sibling-branch credit on word-problem thinking, on singular-value scales

This method stacks on the published entry named in `entry.json` `parent`, 0009552724-0008 (PR #79). That entry builds
on 0009550031-0005 (PR #76), which builds on our own 0009547524-0008 (PR #66). The start model is their work and they
get the credit for it. This method adds one new layer on top and describes only that layer. `task.json` -> `starts`
lists the start by its sid. Use the start with that sid, not the champion start, if the task offers both.

The layer rescales singular values of the start model's residual writers. Each rescaling is one `dir_ablate` op whose
direction is a singular vector of the start's own weight, so the edit only re-weights directions the start model
already uses. Nothing is shipped: every scale is fitted in the run on the start model's own samples of practice
problems. The numbers in section 7 are the author's practice measurements, for orientation only.

## 1. Diagnosis

Sample the start model 4 times on fresh `gsm8k_train` practice problems (8192-token budget). About a sixth of the
problems have mixed outcomes: some samples are right and some are wrong. The majority answer of the 4 samples is right
noticeably more often than a single sample is, so the model often can solve these problems but does not commit to the
solving path every time.

Two own traces of the same problem usually agree on the setup and then part at one step. One sample commits to a
quantity or an operation that the other does not, and the rest of the trace follows from that choice. A reward given to
the whole trace credits hundreds of tokens that were not at fault (the shared setup, the arithmetic after the choice),
so the update spreads over everything the thinking does.

The method localises the credit with the start model's own continuations:

- **Sibling branches.** Cut an own trace's thinking at a line break at 20%, 40% and 60% of its length. Let the start
  model continue the thinking from that fixed prefix 4 times (full answer, graded). Siblings share every token before
  the cut, so their outcomes differ only through what was written after it. This is the Monte Carlo estimate of the
  value of a reasoning state (Sutton & Barto, ch. 5 and 7; VinePPO, arXiv 2410.01679; Math-Shepherd, arXiv
  2312.08935), used here directly as a group baseline.
- **Credit the step right after the cut.** Each continuation gets its leave-one-out advantage
  A_j = r_j - mean of its siblings' rewards (RLOO, arXiv 2402.14740; REINFORCE with a baseline, Sutton & Barto ch. 13).
  The advantage applies only to its first 384 tokens, the step taken from the shared state.
- **Only the forking tokens.** Inside that window, only the 20% of tokens where the start model's own next-token entropy
  is highest carry the gradient. Low-entropy tokens (copying a number, finishing a word) do not decide the branch
  (arXiv 2506.01939).
- **A trust region per token.** The surrogate is the clipped ratio min(rho*A, clip(rho, 0.8, 1.2)*A), with
  rho = p_edit / p_start on the sampled token (PPO, arXiv 1707.06347; DAPO, arXiv 2503.14476). With negative advantages,
  an unclipped log-likelihood gradient keeps pushing a failing token's probability down without bound. The clip stops
  each token once it has moved by 20%.
- **Anchors on other sources.** KL(start || edit) on 384-token windows of the start model's own thinking on
  `math_train` and `mmlupro` problems, each cut at a random line break. The scales may not change how the model thinks
  where the word-problem reward is silent.

What this layer does not do: it uses no checkpoint values, no rows of trace segments, no likelihood target on the
start's own text and no direction removal. The reward of each sampled continuation is compared with its siblings', on
the forking tokens of the step right after a shared prefix, in the singular-value coordinates of our PR #66 and under
a clipped ratio. PR #79's own METHOD measures values at checkpoints and weights trace segments by their change; we cite
it as related work on step-level credit. Its uploaded edit, the start of this method, removes two residual directions
from attention outputs.

## 2. What is changed (`derive/branch_credit.py`)

For each writer matrix W (attention `o_proj`, linear-attention `out_proj`, MLP `down_proj`) of every decoder layer
except the first four and the last two (`--layers 4-21` on this 24-layer model):

- exact SVD of the start weight, W = U S V^T, keeping the top `--k 32` left singular vectors u_i;
- one trainable scale per singular value, s_i -> (1 + z_i) s_i, with z_i = 0 at the start. In the forward pass this is
  y -> y + U diag(z) U^T y;
- export: `dir_ablate`, side `out`, u = u_i, alpha = -z_i x dose. The u_i of one matrix are orthonormal, so the ops
  commute. Every fit uses the same basis (it depends only on the start), so fits average by their z.

About 1100 ops (fp16 vectors), payload about 2.4 MB. No norm, embedding, head, `A_log`, `dt_bias` or convolution
weight is touched.

## 3. Rules for the whole run

- Use only practice data and the start model's own generations. Never load `bench:` sets. Download nothing.
- Run one job per GPU, each detached (`nohup ... > log 2>&1 &`) with its own `CUDA_VISIBLE_DEVICES=i`, each under
  40 minutes. Check logs every 10 minutes. Write PROGRESS.md after every step: what finished, the numbers, the next
  command. If you are asked to continue, read PROGRESS.md first.
- Stop jobs by the PID you saved, never by a name pattern.
- `axonry-check` copies the start's `<sid>/edit/` beside the upload. If its only complaint is `<sid>/edit/FLAT.json`,
  rerun it with `--parent` pointing to a copy of that directory without FLAT.json. Never upload FLAT.json.

## 4. Setup (about 15 minutes)

`S` = the start's sid (`0009552724-0008`, listed in `task.json` -> `starts`).

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp W=/workspace/work/br; mkdir -p $W/scr
python -m wsp.case init br --profile qwen35-0.8b-think --parent S/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512 \
    --subset dev128=dev:128 --subset holdout256=holdout:256
MP=$(python -c "from wsp import case; print(case.model_path('br','base'))")
python derive/branches.py ids --case br --source gsm8k_train --n 4096 --shards 16 --out $W/ids
python derive/branches.py ids --case br --source gsm8k_train --n 512 --shards 1 --exclude-files $W/ids_*.json --out $W/scr_g
python derive/branches.py ids --case br --source math_train --n 192 --shards 1 --out $W/anc_m
python derive/branches.py ids --case br --source mmlupro --n 192 --shards 1 --out $W/anc_c
```

`scr_g` is the gsm8k screen set. It is never sampled for training.

## 5. Steps

**A. Own samples (about 45 min, 4 GPUs).** On GPU i of 4, shards i, i+4, i+8 and i+12, back to back:
`python derive/branches.py sample --model $MP --ids $W/ids_<j>.json --out $W/own_<j>.jsonl`.
Each 256-problem shard takes about 10 minutes. On the GPU that finishes first:
`--ids $W/anc_m_0.json --n 1 --out $W/anc_m.jsonl`, the same for `anc_c`, then
`python derive/branches.py anchors --own $W/anc_m.jsonl $W/anc_c.jsonl --out $W/anchors.jsonl`.

**B. Sibling branches (about 40 min, 4 GPUs).** Start branching on the first shards while later ones still sample:
`python derive/branches.py prep --own $W/own_<j>.jsonl ... --out $W/traces_<g>.jsonl` per group of 4 finished shards,
then on a free GPU `python derive/branches.py branch --model $MP --traces $W/traces_<g>.jsonl --out $W/br_<g>.jsonl`.
Expect about 600 mixed problems and 1200-1400 sibling groups with mixed outcomes over all shards. Fewer than about 1000
mixed groups gives fits that agree across seeds but not across problem sets: collect more before fitting.

**C. Fits (about 20 min, 4 GPUs in parallel).** Four seeds on all branches:

```
CUDA_VISIBLE_DEVICES=i nohup python derive/branch_credit.py fit --model $MP --branches $W/br_*.jsonl \
  --traces $W/traces_*.jsonl --anchors $W/anchors.jsonl --seed i --out $W/fit_i.pt > $W/fit_i.log 2>&1 &
```

The log prints the policy term, the anchor KL and the z spread every 200 items. z rms ends near 0.017. The anchor KL
stays near 0.001 nats per token.

**D. Candidates and the first upload.** Take the mean of the fits at three doses:
`python derive/branch_credit.py export --case br --name D2 --fits $W/fit_*.pt --dose 2`, plus `D15` at dose 1.5 and
`D3` at dose 3. The export prints the mean z correlation between fits (expect about 0.75). Upload D2 at once
(`python -m wsp.upload br D2 --task task.json --report REPORT.md`, then `axonry-check`), so a valid edit exists early.

**E. Screens (paired: every model gets the same sampling seeds).** One job per GPU, for `base`, `D15`, `D2` and `D3`:
`python derive/branches.py screen --case br --edit <name> --ids $W/scr_g_0.json --n 4 --out $W/scr/<name>_g.jsonl`.
Then `python derive/branches.py compare $W/scr/base_g.jsonl $W/scr/<name>_g.jsonl`. Add
`python -m wsp.case run br <name> holdout256 --n 2` for the other sources, using the default token limit. Math screens
at a 16k budget mostly measure how many answers finish before the cut, not whether they are right, so judge math
only at the full budget.

**F. Choice.** A candidate is eligible if no holdout256 source drops by more than one se. Among the eligible ones, take
the best gsm8k screen gain. Keep D2 unless another dose beats it by more than one se. Re-upload only when the choice
changes. In REPORT.md, write the counts (problems, mixed problems, traces, sibling groups, mixed groups), the fit
correlation, the screen table and the reason for the choice.

## 6. What does not work on this lineage (measured on practice, do not repeat)

- A fit on about 600 mixed groups: four seeds agree with each other (z correlation 0.8), but fits on two different
  problem sets do not (about 0), and the gsm8k screen does not move at any dose from 1 to 3. Pool the data first.
- Raising the dose to make up for little data: thinking gets shorter, and math moves only where answers were being cut
  at a short budget.
- Lowering the sampling temperature of the start model (0.45, 0.3) does not raise gsm8k accuracy on the screen. The gap
  to the majority vote is not a matter of sharpness.
- An answer-span policy gradient through MLP units gated open after `</think>`, without the ratio clip: answers
  degenerate into reasoning-style text.

## 7. What to expect (author's practice runs on this start; paired, 512 fresh gsm8k_train problems x4)

- Samples: about 4,300 problems x4 gave 1383 mixed sibling groups (from 874 + 509 traces in two pools). Three seeds
  correlate at z 0.76.
- Dose 2 of the three-seed mean, on two independent screen runs: gsm8k +0.73 +- 0.83 and +1.12 +- 0.79. The cut share
  at 16k fell from 4.0% to 3.4%.

## References

- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.): ch. 5 and 7 (Monte Carlo and n-step targets), ch. 13
  (REINFORCE with baseline).
- RLOO: Ahmadian et al. 2024, arXiv 2402.14740. PPO: Schulman et al. 2017, arXiv 1707.06347. DAPO: arXiv 2503.14476.
- VinePPO (Monte Carlo values of intermediate reasoning states): arXiv 2410.01679. Math-Shepherd: arXiv 2312.08935.
  Process supervision: arXiv 2305.20050.
- High-entropy forking tokens: arXiv 2506.01939.
- Singular-value fine-tuning (Transformer^2 / SVF): arXiv 2501.06252. Our PR #66 for these coordinates.
- Start model: PR #79 (0009552724-0008), PR #76 (0009550031-0005), PR #66 (0009547524-0008).
