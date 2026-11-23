# Puzzle RL on quiet neurons, many seeds: on-policy neuron gains for multi-step state tracking

This method stacks on the published entry `0009565562-0012` (`entry.json` `parent`); only that entry's ops are
inherited. It is built on one diagnosis: the start finishes multi-step state-tracking puzzles but, on some samples, loses
track part-way.

How the edit is learned:
- **Reward.** It comes only from freshly generated puzzles with exact answers, and it is plain right/wrong (exact match),
  with no length term. No practice problem earns reward; the model's own math and instruction answers act only as KL
  anchors.
- **Trainable set.** The MLP neurons that are quiet on the start's own correct math and instruction reasoning, minus the
  units the start's own edit writes. The change is written as column gains of `down_proj` (`scale_cols`).
- **Iterations.** Each is the mean of 8 independent fits on random halves of the puzzles, each under a PPO ratio clip
  and a KL to the iteration's start. Each iteration is sampled from the edit built so far, and the iterations multiply
  into one edit on the start.
- **Uploads.** Two iterations are planned, a third runs if the clock allows, and an edit is uploaded after every
  iteration.

Everything is derived during the run, from the generators in `derive/puzzles.py`, the toolkit's practice sets and the
start model's own generations. No arrays or data files are shipped. The numbers below are the author's practice
measurements, given for orientation only; section 6 says which start each comes from.

## 1. The behaviour to change, and why it is there

The author sampled the start (`0009565562-0012`) on 760 fresh puzzles of 19 families, 8 samples each at 20k tokens.
The families cover:
- who holds what after a sequence of swaps;
- orderings from pairwise clues;
- a free slot in a schedule;
- adjective order;
- dates after offsets;
- navigation;
- sorting words;
- and others.

On that run:
- accuracy was 76.9%, and only 5.1% of the answers were cut at the token limit;
- 352 of the 760 puzzles had mixed outcomes, 370 were always solved and only 38 were never solved.

The errors are stochastic, not systematic. The right procedure is already in the model's distribution, and the model
loses track part-way on some samples. That is the setting where on-policy RL with a verifiable reward helps: it moves
probability toward the model's own successful traces and imports nothing.

The weak families on that run were:
- word ordering (27%, 18% cut);
- temporal schedules (48%);
- ordering seven objects from clues (52%, 30% cut);
- adjective order (61%);
- shapes from path commands (71%);
- five-object orderings, seven-object swap tracking and dates (75-77%).

Arithmetic, boolean evaluation, truth-teller chains and table lookups were at 89-98%.

Why per-neuron gains, quiet neurons, and many seeds:
- **The edit form.** A gated MLP neuron `i` adds `down[:, i] * h_i` to the residual stream. The gain
  `h_i -> (1 + g_i) h_i` is an exact column scale of `down_proj`, so the trained form is the uploaded form.
- **The trainable set.** The trainable neurons are those at or below the median of mean |activation| on the start's own
  correct math and instruction traces (layers 4-21). Neurons that carry those skills are frozen, and KL anchors on the
  same traces penalise any remaining drift.
- **Disjoint from the start's edit.** Neurons that the start's own edit already rewrites are removed from the trainable
  set (`derive/mask_disjoint.py`), so the new gains never stack on a unit the start depends on. On this start, 57 of
  those neurons were in the quiet set.
- **Many seeds.** One fit is mostly seed noise: fits differ in which half of the puzzles they see and in their order. On
  the author's earlier runs, the cosine between two fits' gains was 0.12-0.31. Averaging turns the fits into a stable
  edit. This method therefore runs **8 fits per iteration** and uses 2x their mean (dose 2/n per fit).

## 2. Data (all generated during the run)

- **Puzzles** (`derive/puzzles.py`).
  - The families are the kinds of the toolkit's practice set `bbhsynth` (`wsp/puzzles.py`), with the author's own
    templates and word lists; each generator also computes its exact answer.
  - `bbhsynth` is a frozen set (128 items per kind), but RL needs fresh, disjoint draws for every iteration plus a
    screen and a holdout. So the method draws them from `derive/puzzles.py` and keeps `bbhsynth` as an untouched
    second screen.
  - Any item whose `wsp.data.fingerprint` is in the toolkit's benchmark fingerprints, or that repeats an earlier set,
    is dropped and redrawn.
  - Seeds fix five disjoint sets: train1, train2 and train3 (40 per family, seeds 106, 107 and 108), screen (12 per
    family, seed 103) and holdout (12 per family, seed 104). Answers use the toolkit's `exact` task and grader.
- **Anchors**: the start's own answers (2 samples, 8k-token cap) to 600 practice problems: gsm8k_train 220,
  math_train 160, ifsynth 90 and tuluif 130, all disjoint from the case's dev and holdout (`derive/make_anchor_ids.py`).
  Finished, correct traces are used for the neuron mask (`stats`) and as KL-only anchors. They earn no reward.
- **Guards and screens**:
  - the case's dev128 split (gsm8k, math, ifsynth, tuluif) via `wsp.case run`;
  - the own puzzle screen;
  - after the choice, the puzzle holdout and up to 12 items per kind of the toolkit's practice set `bbhsynth`
    (`derive/bbhsynth_screen.py`, fixed seed). These are never used for training.

## 3. The objective (`derive/neuron_rl.py fit`)

The leave-one-out policy gradient on the model's own samples follows the published entry `0009547524-0008`
(answer-credited group-relative policy gradient). Here the reward is an exactly graded generated puzzle and the whole
trace is credited.

For each puzzle with 8 own samples and mixed rewards (exact answer = 1; wrong or cut = 0):
- leave-one-out advantage (RLOO) per sample;
- the whole own trace is credited (`--span all`: thinking and answer);
- a per-token PPO-style ratio clip (0.2) against the iteration's start, plus k3 KL to it (`--kl 0.02`) on the credited
  tokens;
- KL-only anchors: 450 own math, gsm8k and instruction traces with weight `--anchor-kl 5`. Math and instruction
  sources get no policy gradient (`--src-weight ... = 0`);
- one gain per trainable neuron, AdamW lr 0.01, 1 epoch on a random half of the puzzles (`--subsample 0.5`, a
  different half per seed); gains are clamped to |g| <= 0.5 and traces to 16k tokens.

Export (`derive/combine.py`):
- Within an iteration the fits add, each with dose 2/n (n = fits, so 0.25 for 8).
- Across iterations the per-neuron factors multiply: `f = prod_iter (1 + sum dose * g)`.
- The 12000 neurons with the largest |f - 1| are kept, f is quantized to 0.002, and one `scale_cols` op is written per
  (layer, factor). The payload is about 0.1 MB.

## 4. Setup (first 15 min)

From the unpacked bundle, copy `submission`, `task.json` and the start's directory to `/workspace/work`, as the brief
says. The start's directory is `<sid>` = `0009565562-0012`, holding `edit/` with anything it stacks on nested inside, so
that `/workspace/work/0009565562-0012/edit/edit.json` exists. `python submission/derive/start.py` prints that path and
stops with an error if it is missing.

The parent already stacks on the current champion `0009554110-0010`. Compare only with the case's base (the parent);
there is nothing to adopt.

**task.json holds upload credentials. Keep it out of logs, PROGRESS.md and REPORT.md.**

```
cd /workspace/work && mkdir -p pz logs
D=/workspace/work/submission/derive
START_EDIT=$(python $D/start.py); START_DIR=$(dirname $START_EDIT)
python -m wsp.case init pz1 --profile qwen35-0.8b-think --parent "$START_EDIT" \
    --practice gsm8k_train math_train ifsynth tuluif --subset dev128=dev:128 --subset holdout256=holdout:256
python $D/puzzles.py pz/train1.jsonl 40 106
python $D/puzzles.py pz/train2.jsonl 40 107 --avoid pz/train1.jsonl
python $D/puzzles.py pz/train3.jsonl 40 108 --avoid pz/train1.jsonl pz/train2.jsonl
python $D/puzzles.py pz/screen.jsonl 12 103 --avoid pz/train1.jsonl pz/train2.jsonl pz/train3.jsonl
python $D/puzzles.py pz/holdout.jsonl 12 104 --avoid pz/train1.jsonl pz/train2.jsonl pz/train3.jsonl pz/screen.jsonl
python $D/make_anchor_ids.py --case pz1 --out pz/anchor_ids.json
START=$(python -c "from wsp import case; print(case.model_path('pz1','base'))" | tail -1)   # build the start ONCE, before any GPU job
W=/workspace/work/pz
cat > /workspace/work/env.sh <<EOF
export D=$D PZ=$W START_EDIT=$START_EDIT START_DIR=$START_DIR START=$START PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PUZZLE_ITEMS=$W/train1.jsonl:$W/train2.jsonl:$W/train3.jsonl:$W/screen.jsonl:$W/holdout.jsonl
export FIT="--span all --if-reward strict --src-weight gsm8k=0 math=0 mmlupro=0 ifsynth=0 tuluif=0 --anchor-runs $W/anchors.jsonl --anchor-n 450 --anchor-kl 5.0 --kl 0.02 --clip 0.2 --lr 0.01 --gmax 0.5 --epochs 1 --subsample 0.5 --max-tokens 16000 --layers 4-21"
cd /workspace/work
EOF
```

Begin every later command with `source /workspace/work/env.sh`, because the shell state does not survive between
commands. Without `PUZZLE_ITEMS` a fit finds no puzzle groups and stops with that message.

Ground rules:
- Run one job per GPU, each detached with `setsid nohup ... &` and its own `CUDA_VISIBLE_DEVICES`. Append every PID to
  `logs/pids` and stop jobs only by PID (never `pkill -f` / `pgrep -f`).
- Check the jobs every 10-15 minutes and keep PROGRESS.md current. Never wait more than 15 minutes in one command.
- Use practice data and own generations only; never load a `bench:` set.
- Make choices on the practice guards and the own puzzle screen. Read the holdouts only after the choice.

## 5. Steps and time plan (6 h, 4 RTX 4090)

The author measured iteration 1 (steps A and B) on 4 A100 80GB GPUs:
- 22 minutes per sample shard of 190 puzzles x 8;
- 10 minutes for the anchors;
- 17 minutes per fit.

Iteration 2's sampling took 24 minutes on 4 RTX 5090 GPUs. On an RTX 4090 expect 30-45 minutes of sampling per
iteration (about 48M generated tokens) and 23-32 minutes per fit. Write the measured times in NOTES as you go.

**A. Iteration-1 samples and anchors (0:15-1:10).**
```
source /workspace/work/env.sh
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s setsid nohup bash -c "source /workspace/work/env.sh; python \$D/sample.py --case pz1 \
   --items \$PZ/train1.jsonl --n 8 --salt it1 --max-tokens 20000 --shard $s --nshard 4 --out \$PZ/it1_$s.jsonl" \
   > logs/it1_$s.log 2>&1 & echo $! >> logs/pids; done
CUDA_VISIBLE_DEVICES=3 setsid nohup bash -c 'source /workspace/work/env.sh
python -m wsp.gen --profile qwen35-0.8b-think --model $START --ids $PZ/anchor_ids.json --n 2 --salt anc --max-tokens 8000 --out $PZ/anchors.jsonl
python $D/neuron_rl.py stats --case pz1 --runs $PZ/anchors.jsonl --out $PZ/mask1.pt --q 0.5 --n 200 --layers 4-21
python $D/mask_disjoint.py $PZ/mask1.pt $PZ/mask1d.pt $START_DIR
python $D/sample.py --case pz1 --items $PZ/train1.jsonl --n 8 --salt it1 --max-tokens 20000 --shard 3 --nshard 4 --out $PZ/it1_3.jsonl' \
   > logs/gpu3_A.log 2>&1 & echo $! >> logs/pids
```
Expect accuracy of about 75-80%, about 5% cut, and about half of the puzzles mixed. Write the per-family accuracy into
NOTES.

**B. Iteration-1 fits (1:10-2:15): 8 seeds, two chained on each GPU.**
```
source /workspace/work/env.sh
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "source /workspace/work/env.sh; \
   python \$D/neuron_rl.py fit --case pz1 --runs \$PZ/it1_*.jsonl --mask \$PZ/mask1d.pt --name F$((g+1)) --seed $((g+1)) \$FIT; \
   python \$D/neuron_rl.py fit --case pz1 --runs \$PZ/it1_*.jsonl --mask \$PZ/mask1d.pt --name F$((g+5)) --seed $((g+5)) \$FIT" \
   > logs/F_g$g.log 2>&1 & echo $! >> logs/pids; done
# when all 8 have ended:
python $D/combine.py --case pz1 --name L1 --topn 12000 --parts $(for i in 1 2 3 4 5 6 7 8; do echo -n "pz1:F$i:0.25 "; done)
```
Each fit runs 90-92 steps on about 290 mixed puzzles (about 9.5M tokens). **Upload L1** (step H) before going on.

**C. Iteration 2 (2:15-3:45).** The start of iteration 2 is L1.
```
source /workspace/work/env.sh
python -m wsp.case init pz2 --profile qwen35-0.8b-think --parent $WSP_CASES/pz1/edits/L1.json \
    --practice gsm8k_train math_train ifsynth tuluif --subset dev128=dev:128
python -c "from wsp import case; case.model_path('pz2','base'); case.model_path('pz1','L1')"
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s setsid nohup bash -c "source /workspace/work/env.sh; python \$D/sample.py --case pz2 \
   --items \$PZ/train2.jsonl --n 8 --salt it2 --max-tokens 20000 --shard $s --nshard 4 --out \$PZ/it2_$s.jsonl" \
   > logs/it2_$s.log 2>&1 & echo $! >> logs/pids; done
# when they end:
python $D/neuron_rl.py stats --case pz2 --runs $PZ/anchors.jsonl --out $PZ/mask2.pt --q 0.5 --n 200 --layers 4-21
python $D/mask_disjoint.py $PZ/mask2.pt $PZ/mask2d.pt $START_DIR
```
Then the 8 fits as in step B, with `--case pz2 --runs \$PZ/it2_*.jsonl --mask \$PZ/mask2d.pt`, names G1-G8 and seeds
11-18 (G$((g+1)) with seed $((g+11)), then G$((g+5)) with seed $((g+15))). When all 8 have ended:
```
source /workspace/work/env.sh
P1=$(for i in 1 2 3 4 5 6 7 8; do echo -n "pz1:F$i:0.25 "; done); P2=$(for i in 1 2 3 4 5 6 7 8; do echo -n "pz2:G$i:0.25 "; done)
python $D/combine.py --case pz1 --name C2 --topn 12000 --parts $P1 $P2
```
**Upload C2.**

**D. Iteration 3, if `axonry-time` shows at least 2 h 10 min left when C2 is uploaded.** As step C, from C2:
- `wsp.case init pz3 --parent $WSP_CASES/pz1/edits/C2.json` (same flags);
- samples of `$PZ/train3.jsonl` with salt `it3` into `$PZ/it3_*.jsonl`;
- mask `mask3d.pt`;
- fits H1-H8 with seeds 21-28.

Then:
```
source /workspace/work/env.sh
P1=$(for i in 1 2 3 4 5 6 7 8; do echo -n "pz1:F$i:0.25 "; done); P2=$(for i in 1 2 3 4 5 6 7 8; do echo -n "pz2:G$i:0.25 "; done)
P3=$(for i in 1 2 3 4 5 6 7 8; do echo -n "pz3:H$i:0.25 "; done)
python $D/combine.py --case pz1 --name C3 --topn 12000 --parts $P1 $P2 $P3
```
**Upload C3** as soon as it is combined. If at least 35 minutes remain then, run step E on C3; otherwise keep C3
uploaded without the guard and say so in REPORT.md. If the third iteration's fits are not all finished 25 minutes
before the deadline, combine the finished ones at dose 2/n, or keep C2. With less than 2 h 10 min left at C2's upload,
skip D: the pick is C2.

**E. Guard and choose (about 30 min).** `<pick>` = C3 if iteration 3 ran, otherwise C2. Run one job per GPU:
```
source /workspace/work/env.sh
CUDA_VISIBLE_DEVICES=0 setsid nohup python -m wsp.case run pz1 base dev128 --n 2 > logs/g_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup python -m wsp.case run pz1 <pick> dev128 --n 2 > logs/g_pick.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup python $D/sample.py --case pz1 --model base --items $PZ/screen.jsonl --n 2 --salt scr \
   --max-tokens 20000 --out $PZ/scr_base.jsonl > logs/scr_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup python $D/sample.py --case pz1 --model <pick> --items $PZ/screen.jsonl --n 2 --salt scr \
   --max-tokens 20000 --out $PZ/scr_pick.jsonl > logs/scr_pick.log 2>&1 &
# when they end:
python $D/screen.py $PZ/scr_base.jsonl $PZ/scr_pick.jsonl
```
Read `cases/pz1/results/<pick>__dev128.json` for `paired.gain`, `se` and `by_source`.

Rules:
- Keep the pick unless one of these happens:
  - dev128 falls by more than about 2 points overall;
  - one source (gsm8k, math, ifsynth, tuluif; 32 problems each) falls by more than about 6;
  - the cut share or mean tokens rises sharply;
  - its puzzle-screen gain is clearly negative (more than about one standard error below 0).
- If the pick fails and it was C3, upload C2 instead. Upload C2 only if its own dev128 is within the same limits;
  otherwise keep the last edit that passed, or L1.
- Freeze only the final pick, after its dev128 guard: `python -m wsp.case freeze pz1 <pick>`.

**F. Fallbacks.**
- If a fit fails, combine the fits that finished at dose 2/n each.
- From 30 minutes before the deadline, start no new experiments. The last upload must be complete and checked.

**H. Upload commands** (for L1, C2, C3 and the final pick; no freeze is needed to upload). Write a short
`/workspace/work/REPORT.md` before the first dry run (wsp.upload stops with "no report" without it); write it to a
temporary file and then `mv` it into place.
```
source /workspace/work/env.sh
python -m wsp.upload pz1 <edit> --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/pz1__<edit>/edit --task task.json
python -m wsp.upload pz1 <edit> --task task.json --report /workspace/work/REPORT.md
```

After the last upload, if time is left, run the start and the pick on the holdout and on the bbhsynth screen, and report
both:
```
source /workspace/work/env.sh
python $D/bbhsynth_screen.py $PZ/bbh_screen.jsonl
python $D/sample.py --case pz1 --model base --items $PZ/bbh_screen.jsonl --n 2 --salt bbh --max-tokens 20000 --out $PZ/bbh_base.jsonl
```
Run the same for `<pick>`, then `screen.py`. The holdout works the same way, with `holdout.jsonl` and salt `hold`.
Then run `axonry-done`.

## 6. What to expect (author's practice runs, for orientation)

- On this start (`0009565562-0012`), iteration 1 (760 puzzles x 8) gave:
  - about 290 mixed puzzles per fit;
  - gains in the range -0.21..+0.20;
  - 12000 neurons in about 1700 `scale_cols` ops, payload about 96 KB;
  - global relative change 0.0065.
- After one iteration, the model's accuracy on fresh practice puzzles was unchanged within noise: 76.1% on the next
  set, against 76.9%. The change comes over iterations.
- On the author's earlier runs of this objective on another start:
  - two iterations of 4 fits each (2x the mean), combined, gained +3.3 +- 0.9 on the puzzle screen and +3.6 +- 1.0 on
    the untouched puzzle holdout. The gains sat where that start was weak: swap tracking, temporal schedules, dates.
  - The instruction anchors kept instruction following in place: without them, one iteration cost about 0.9 points
    there.

## 7. REPORT.md

Report the following:
- the start's per-family accuracy and cut share;
- mixed puzzles per iteration;
- the gain range and the neurons kept;
- how many of the start's edited neurons left the mask;
- the own puzzle-screen table (base vs the pick) and, after the choice, the holdout and bbhsynth-screen tables;
- the dev128 guards by source;
- which iterations ran;
- what was uploaded and when;
- what was rejected and why.

## 8. Credits

- **The parent.** The published entry `0009565562-0012`, whose ops are inherited unchanged. It stacks on
  `0009554110-0010`.
- **The policy gradient.** The leave-one-out policy gradient on the model's own samples follows the published entry
  `0009547524-0008` (answer-credited group-relative policy gradient). Here the reward is an exactly graded generated
  puzzle and the whole trace is credited.
- **The edit form.** One gain per MLP unit, written as `scale_cols`, follows the published entry `0009538688-0019`
  (gains fitted on the model's own answers).
- **Keeping off other units.** Keeping a fit off another edit's units follows the published entry `0009550031-0005`,
  whose trainer can exclude another edit's neurons. Here the trainable set is the neurons at or below the median
  activity on the start's own correct math and instruction thinking, minus every unit the start's edit writes.
- **The toolkit (`wsp`).** Cases, generation and grading, edit building, upload, and the practice set `bbhsynth`.
- **New in this entry:**
  - the generated, exactly graded puzzles as the RL task, with unlimited fresh draws (train sets disjoint per
    iteration);
  - the quiet-neuron mask, disjoint from the start's edit;
  - the 8-seed averaging rule;
  - the multiplicative on-policy iterations;
  - the KL-only anchors on own math, gsm8k and instruction traces.
