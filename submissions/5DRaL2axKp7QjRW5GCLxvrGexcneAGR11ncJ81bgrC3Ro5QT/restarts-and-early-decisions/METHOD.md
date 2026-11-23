# Restarts and early decisions: group credit from restarts inside the thinking, then early-decision credit, as two gain layers on the mended-pairs entry

This entry names the finished entry `0009565562-0012` (mended pairs) as its parent in `entry.json` and starts from
the parent's model: the champion it stands on, the parent's directions, its unit and its instruction fit are inherited
as they are and are not derived again. The entry adds two layers, both fitted in the run on the parent's own answers
to practice problems, and delivers them as one edit.

| layer | what | source |
|---|---|---|
| 1 | **group credit from restarts**: group-relative credit for what a hard maths answer does after a point of its own thinking (restart groups), together with whole-answer groups, as per-column gains | this entry (`derive/branch.py`, `branch_lp.py`, `bgrpo.py`, `grpo2.py`) |
| 2 | **early-decision credit**: preference gains on same-prefix continuation pairs from early points of right and wrong answers, the loss restricted to the 2048 tokens after the branch | method and scripts of `0009568722-0012` (`derive/pref_gains2.py`, `fit_gains.py` unchanged; its branch points, pair rule, fit settings and strength rule as published); its branches are drawn by this entry's sampler (section 3) |

Every figure below is a measurement on practice data, paired by problem and seed, for orientation. No arrays are
shipped.

## 1. What the start loses on hard problems, and where

**Half of the hard problems are won on some draws and lost on others.** The start answers 1,000 fresh `math_train`
problems of levels 4 and 5 eight times each, at the full token limit:

| reading | value |
|---|---|
| answers right | 48.5 % |
| finished with a wrong result / finished without a result / cut at the limit | 41.0 / 7.8 / 2.8 % |
| problems with both a right and a finished wrong answer | 50.4 % (504 problems) |
| right answers shorter than the wrong ones of the same problem | in 72 % of those problems |

**Where a lost answer goes wrong.** Take a finished wrong answer of such a problem, cut its thinking at a quarter, a
half or three quarters (at the end of a line), and let the start go on from there four times (10,920 continuations,
the token limit counting the prefix):

| continued from a wrong answer at | ends right | cut at the limit |
|---|---|---|
| (the same problems from scratch) | 50.4 % | |
| 25 % of its thinking | 31.0 % | 2.9 % |
| 50 % | 23.7 % | 3.4 % |
| 75 % | 19.6 % | 3.6 % |

A quarter into a wrong answer the chances have already fallen from one half to one third, and they keep falling, but
even late prefixes still lead to a right result one time in five. 41 % of the four-continuation groups have both
outcomes. The source entry of layer 2 measured the same on its own model from right and wrong prefixes: the gap
between them is 10 points at 10 % of the thinking and triples by the middle. Most of the outcome is set in the first
half of the thinking, and a prefix still leaves room for both outcomes.

## 2. The two layers

**Layer 1: restart and whole-answer groups.** Two kinds of groups from the start's own answers, each scored by the
practice problem's reference answer (1 right, 0 finished wrong; an answer cut at the limit is on neither side and is
left out):

- **restart groups**: the four continuations of one prefix of a wrong answer (25 / 50 / 75 %). The loss covers the
  continuation tokens only; the prefix is context.
- **whole-answer groups**: the eight answers of one pool problem, over the whole answer.

Only groups with both outcomes train. The advantage of an answer is its reward minus the group mean, divided by the
group's standard deviation; the loss is minus that advantage times the mean log-probability of its scored tokens.
Parameters: one multiplicative gain per group of input columns of the attention (`q`, `k`, `v`, `o`) and MLP (`gate`,
`up`, `down`) matrices of decoder layers 4 to 21, all starting at 1 (the parametrisation of the toolkit's `wsp.fit`,
written by its spec writer as `scale_cols` ops). AdamW, learning rate 2e-3, 16 answers per step, one pass. Four fits
on four problem shards, averaged in log space, dose 1. This is edit `G`.

**Layer 2: early-decision credit** (method of `0009568722-0012`).
1. **Branches.** Up to 200 mixed pool problems whose chosen wrong answer and one right answer both have at least 3000
   thinking tokens; each of the two is cut at 10 %, 25 % and 50 % of its thinking and continued 4 times.
2. **Pairs.** At each branch point a continuation that ends right is paired with one that does not (a continuation
   cut at the limit counts as not right), at most 2 pairs per point; a pair whose two continuations agree over the
   first half of the window is left out.
3. **Fit.** Per-unit MLP gains and per-head attention gains of layers 4-21, DPO-style preference (beta 0.1) plus an
   NLL term (0.2) on the right continuation, **the loss restricted to the first 2048 tokens after the branch point**.
   Four fits, seeds 1-4, 2 epochs, on the start's model (the model the branches were drawn from).
4. **Strength, chosen in the run.** The four fits' log gains are averaged and exported at strengths 0.5, 0.75 and
   1.0 **on top of layer 1** (`fit_gains.py export --parent G.json`). Layer 1 and each candidate answer the same 256
   fresh `math_train` problems of levels 3-5 once; the largest strength whose mean thinking length is at most 1.18
   times layer 1's and whose share cut at the limit is at most 2 points above layer 1's (or at most 6 %, if that is
   more) is used; if none passes, 0.5 (`derive/ecdose.py`).

Why the two layers fit together: layer 1 shortens the thinking (answers 5-8 % shorter, fewer answers that run to the
limit or close the thinking without a result); the early-decision credit lengthens it. The strength rule of layer 2
is measured against layer 1, so it keeps the combined length in the source entry's limits.

## 3. The branches, drawn once for both layers

`derive/branch.py` draws all continuations, with two properties:
- **The continuations sample like full answers.** The profile's presence penalty applies to generated tokens only,
  and a prefix put into the prompt would lose it. `derive/branch_lp.py` (a vLLM logits processor) applies the same
  penalty to every prefix token that the continuation has not generated yet; the token limit counts the prefix, as
  in one full answer.
- **Shared prefixes are drawn once.** Layer 2's wrong answer of a problem is the one the restart groups use (the same
  draw), and its 25 % and 50 % cuts are the same prefixes: those continuations are copied from the restart run
  (`--reuse`), and only the 10 % cuts of wrong answers and the cuts of right answers are sampled again.

## 4. Steps (4 x RTX 4090, 6 h; times measured in our full rehearsal of this script on such a machine)

Initialise the case on the parent (`python -m wsp.case init main --profile qwen35-0.8b-think --parent
0009565562-0012/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth tuluif bbhsynth --dev 320
--holdout 640`), then, in the directory that holds `task.json`, run `python submission/derive/run.py main --upload`
(`--status` lists what is done; running it again resumes; it uploads after each layer). Its steps, each written once:

| step | what | GPUs | time |
|---|---|---|---|
| `pool.py` | the lists: 500 level-4/5 `math_train` problems; screens of 300 `math_train` (drawn at random), 200 `gsm8k_train`, 150 `ifsynth`, 150 `tuluif`, 200 `bbhsynth`; 256 level-3/5 `math_train` for the strength; none in dev or holdout | none | 1 min |
| `pool8` | the start x 8 on the pool, full limit | 4 | 85 min |
| `rs` | one finished wrong answer of each mixed problem cut at 25 / 50 / 75 %, x 4 continuations | 4 | 48 min |
| `ec` | layer 2's branches (section 2), the shared prefixes copied | 4 | 63 min |
| `G_s0..3`, `G` | layer 1's four fits and their average; **upload 1** (layer 1; its check runs beside the next step) | 1 each | 36 min |
| `EC_s1..4` | layer 2's four fits (only if at least 100 minutes remain; otherwise upload 1 stands); layer 1's model is built on the CPU meanwhile | 1 each | 51 min |
| `EC050`, `EC075`, `EC100`, strength | the exports and the strength screen (only if at least 45 minutes remain; otherwise 0.5); **upload 2** (layers 1 and 2) | 4 | 40 min |
| screens | start and delivered edit on the five screens, the same seeds (2 answers each, 1 when less than 75 minutes remain, none under 40) | 4 | 30-60 min |

In the rehearsal, upload 2 passed `axonry-check` 14 minutes before the deadline, and the screens were left out by
their time guard; building layer 1's model during the fits and checking upload 1 beside them saves about ten more
minutes. Each upload is packaged by `wsp.upload --dry-run` and passed through `axonry-check` before it is sent, and
upload 2 starts only after upload 1 has finished. The delivered edit stays unless a screen source is below the start
by more than two standard errors; then layer 1 alone is uploaded again. Each parallel job has its own kernel cache (`TRITON_CACHE_DIR`). If a step fails, read its log, fix the cause
and run `run.py` again.

## 5. What to expect (practice data)

| | against | reading |
|---|---|---|
| layer 1, fresh `math_train` levels 1-3 (6 against 8 answers) | the start | +1.38 +- 0.71 |
| layer 1, fresh `math_train` levels 4-5 | the start | +1.09 +- 0.91 |
| layer 1, fresh `ifsynth` / `tuluif` (4 against 4) | the start | +1.38 +- 1.14 / -0.62 +- 0.71 |
| layer 1, `gsm8k_train` | the start | +0.08 +- 1.13 |
| layer 2 on its source entry's model, held-out `math_train` / `gsm8k_train` | that model | +1.87 +- 1.04 / +0.83 +- 1.56 (as published) |

- Layer 1 gains by fewer answers that close the thinking without a result (5.7 % -> 2.9 %) or run to the limit
  (1.4 % -> 0.4 %); finished wrong answers stay where they were.
- Layer 2 raises the accuracy of the answers that finish and lengthens the thinking; the strength screen keeps the
  share cut at the limit in check.
- In our rehearsal on the parent: continuations from right / wrong prefixes ended right 55.2 / 41.8 % (10 %), 64.4 /
  31.6 % (25 %), 71.5 / 23.6 % (50 %); 1148 pairs; on the 256 strength-screen problems (one answer each) layer 1
  alone 53.9 %, with layer 2 at 0.5 / 0.75 / 1.0: 54.3 / 60.9 / 57.4 % (1.0 cut 10.2 % at the limit, over the
  cap; 0.75 chosen, thinking 1.06 times layer 1's, share cut unchanged at 5.9 %).

## 6. What did not work (measured, do not repeat)

- Whole-answer groups alone: +0.43 and -0.57 on the maths screen (doses 1 and 2), answers 11-14 % longer.
- Restart groups alone: +1.43 on the screen, -0.15 +- 0.67 on fresh problems.
- Layer 1 at dose 2: answers 13 % shorter and more finished wrong answers; at dose 0.5: +0.92.
- The source entry of layer 2 reports: no gain without the 50 % branch point, a much smaller effect from a smaller
  branch set, and no length matching needed (the window removes the continuation's length).

## 7. Rules kept

Practice sets and the start's own generations only; dev and holdout are not used for any choice; no `bench:` set is
loaded. Both layers are derived in the run on the parent's model.
