# Same thinking, different answer: average several preference fits on the model's own branches

## The problem
A thinking model often plans an answer and then writes one that breaks the request: a format, a count, a keyword,
a casing or a length rule the thinking had noted. On instruction-following practice problems (ifsynth, tuluif) the
starting model's finished answers fail the checker 14-24% of the time.

The decisive measurement holds the model's own thinking fixed. Take a finished answer, keep everything up to and
including `</think>`, and sample only the answer again, 4 times:
- the outcome after the same thinking is mixed (some answers pass, some fail) for 12-21% of the thinkings;
- for about a third of the failed answers, another answer after the very same thinking passes.

So the model can write the passing answer from where it stands, but does not do so reliably.

Measure this first, on the starting model (step 3), per source: the mixed share and how often a failed answer has a
passing sibling. The method applies when at least ~8% of the thinkings are mixed; if they are not, say so and stop.

## Why we think it happens, and what the edit learns
Two answers written after identical thinking differ only in how the answer is written, so the checker's verdict on
them is clean credit: this way of answering keeps the constraints, that way loses them. A preference between the
two (DPO with an NLL term on the passing one) teaches answer discipline. Measured: the gain is spread over every
family of constraints (format, keywords, length, case, start/end, content: +9 to +19 points each). The effect is not
confined to the answer tokens. Holding the thinking's likelihood fixed during training turned the gain into a loss,
so the same change also shapes how the thinking prepares the answer.

A single fit is unreliable, and this is the central finding. Fits from the same pairs land in different places.
Some also lengthen the thinking (answers cut at the token limit go up 1.5-2x) and lose math and multiple choice;
others shorten it. Averaging several fits of the same neurons keeps what they agree on, the answer discipline, and
cancels what they do not. In an independent replication, a single fit lost 7.5 points and the average of 4 fits of
the same data gained 4.5.

## What is new here
- Pairs share a prefix taken from the model's own trace (its whole thinking). Only the answer after it differs, and
  the label is the practice item's own checker. They are not two independent answers to the same problem.
- The objective is a preference between those two branches, scored on the branch tokens only.
- It is not fine-tuning on the model's own correct whole traces, whether picked at random or by shortness: nothing
  is learned from the thinking. The signal is the checker's verdict on two answers written after identical thinking.
- The edit is the average of several fits of one neuron set. The size of the averaged change (a dose k) is chosen
  on a check set.

## Approach
Everything runs in the sandbox with 4 GPUs. Copy `submission/` and the bundle's published edits to `/workspace/work`.
`D=/workspace/work/submission/derive`. Work in `/workspace/work`.

0. **The starting model: the champion.** This entry's parent is `champion`. The subnet resolves it to the champion
   when the run starts, the same model you are judged against, so every point the edit adds counts against it. The
   bundle holds that model as the run's start, under its id: `<id>/edit/`, with anything it stacks on nested inside
   (`task.json`: `starts`, `judge.champion`). The method needs an open answer stage after the thinking, which a
   thinking model has (step 3 measures it). It was developed on stop-thinking (`0009538706-0022`) and
   self-distill-gains (`0009538688-0019`, the champion when this method was written); see What to expect.
   ```
   S=$(python3 -c "import json; t=json.load(open('task.json')); print(t['judge']['champion'] or t['starts'][0]['sid'])")
   ```

   Copy `task.json` and the start's directory `$S/` (all of it: anything it stacks on is nested inside) to
   `/workspace/work` in the sandbox, under the same names. `axonry-check` looks for the start beside `task.json`.
   Then:
   ```
   python -m wsp.case init main --profile qwen35-0.8b-think --parent /workspace/work/$S/edit/edit.json \
     --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 200 --holdout 200
   START=$(python -c "from wsp import case; print(case.model_path('main', 'base'))" | tail -1)
   ```
   `dev` and `holdout` hold 40 problems per source and are never used for training.
1. **Baselines first** (they take longest). Run them detached on GPUs 0 and 1. Every later comparison is paired with
   them.
   `CUDA_VISIBLE_DEVICES=0 python -m wsp.case run main base dev --n 2`, and the same on GPU 1 for `holdout`.
2. **The starting model's own answers** on the instruction-following problems outside dev/holdout, 2 per problem,
   on GPUs 2 and 3 (about 20 minutes on 24 GB GPUs):
   ```
   python3 $D/rollouts.py pick --spec ifsynth=all,tuluif=all --exclude $WSP_CASES/main/sets/dev.json \
     $WSP_CASES/main/sets/holdout.json --out ids.json
   python3 $D/rollouts.py run --model "$START" --ids ids.json --out roll --gpus 2,3 --caps ifsynth=6144,tuluif=6144 --more 0
   ```
3. **Branch the answers.** For each problem, take one finished answer whose thinking closed, keep the thinking, and
   sample 4 new answers after it. Each is graded by the item's checker.
   ```
   python3 $D/branch.py plan --mode answers --rows roll/rows.jsonl --out ta.jsonl --sources ifsynth,tuluif --k 4 --per-problem 1
   python3 $D/branch.py run --model "$START" --tasks ta.jsonl --out ba --gpus 2,3
   python3 $D/branch.py pairs --mode answers --tasks ta.jsonl --results ba --out pairs.jsonl --max-pairs 2
   ```
   `pairs` prints the diagnosis per source; write it into the report. Expect about 350-450 pairs from about 1,800
   thinkings (14-21% mixed on the starts measured so far). Read a few: both answers must follow the same thinking and differ in whether
   a constraint holds.
4. **Four fits of one neuron set.**
   - Fit 1 chooses ~650 MLP neurons (gate/up rows and the down column of each) by the gradient of the pass/fail
     margin, never in the last layer, and trains them.
   - Fits 2-4 train the same neurons from the same start with other seeds (another order of the pairs).
   - Use 2 epochs and a **global batch of 4 pairs**: 2 GPUs, `--accum 2`. The number of steps sets how far a fit
     moves, so keep the batch.
   The fits' specs name the case's copy of the start as their parent (`PREF`, as `case.json` records it), and
   `edit_tools.py package` turns that into the start's id for the upload.
   ```
   PREF=../$(python3 -c "import json; print(json.load(open('$WSP_CASES/main/case.json'))['parent'])")
   T="torchrun --standalone --nproc_per_node 2 $D/train_sparse.py --model $START --data pairs.jsonl \
      --parent-ref $PREF --objective dpo --select margin --epochs 2 --accum 2 --max-len 12288"
   CUDA_VISIBLE_DEVICES=2,3 $T --out-spec $WSP_CASES/main/edits/F1.json --seed 0
   for s in 14 21 28; do CUDA_VISIBLE_DEVICES=2,3 $T --out-spec $WSP_CASES/main/edits/F$s.json --seed $s \
     --neurons-from $WSP_CASES/main/edits/F1.json; done
   (cd $WSP_CASES/main/edits && python3 $D/avg_sparse.py --specs F1.json F14.json F21.json F28.json --out AVG.json)
   ```
   Each fit takes about 5 minutes on two 24 GB GPUs. When the baselines have finished, run two fits at once on
   GPUs 0-1 and 2-3.
5. **Doses.** For k in 0.35, 0.5 and 0.7:
   `python3 $D/dose_sparse.py --spec $WSP_CASES/main/edits/AVG.json --start "$START" --k 0.35 --out $WSP_CASES/main/edits/AVGk035.json`
   (and `AVGk05.json`, `AVGk07.json` the same way).
   This sets every trained row and column to start + k × (averaged − start).
6. **Upload early: AVG at k = 0.35.** On stop-thinking, the smallest dose kept math and multiple choice and was
   positive on both check sets. Larger doses gained no more on instruction following and cost math, multiple
   choice and gsm8k (see What to expect). Write a first REPORT.md, then:
   ```
   python3 $D/edit_tools.py package --spec $WSP_CASES/main/edits/AVGk035.json --out up/AVGk035 --report REPORT.md
   axonry-check up/AVGk035/edit --task task.json
   python3 $D/edit_tools.py upload --dir up/AVGk035 --task task.json
   python3 $D/edit_tools.py verify --dir up/AVGk035 --task task.json
   ```
   `verify` reads the upload back from storage and runs axonry-check on exactly what is stored. It must print
   `VERIFIED`. Use these commands for every later upload.
7. **Choose the dose on dev, confirm on holdout.**
   - Run `AVGk035`, `AVGk05` and `AVGk07` on dev, one GPU each, paired with the baseline.
   - Read per-source gain and SE:
     `python3 $D/failures.py compare $WSP_WORK/runs/main/base__dev.jsonl $WSP_WORK/runs/main/AVGk035__dev.jsonl`.
   - Take the best dev gain. Keep the **lowest** dose whose dev gain is within one SE of it and that loses no source
     by more than about 8 points. Each source has 40 problems, so a few points are noise, and a larger dose is
     taken only when it is clearly better.
   - Run the choice on holdout. If it is not 0.35, upload it (step 6's commands) only if its holdout gain is
     positive and at least that of 0.35 (run 0.35 on holdout too).
8. **If more than 2 hours remain:** fit four more seeds on the same neurons and average all eight. Check it on dev and
   holdout, and upload it only if it beats the uploaded edit on both sets together.

## Time plan (6 hours, 4 GPUs)
- 0:00-0:20: copy, case init, start model; start the baselines (step 1).
- 0:20-0:50: rollouts and branching (steps 2-3).
- 0:50-1:20: the four fits, average and doses (steps 4-5). **Upload and verify AVG k = 0.35 (step 6).**
- 1:20-2:30: dose choice on dev, holdout confirmation, upload and verify (step 7). Update REPORT.md after each one.
- 2:30-5:00: the eight-fit average (step 8) if time allows.
- From 30 minutes before the deadline: no new experiments. The last upload must be complete and verified.

## What to expect
These numbers come from check sets that played no part in choosing anything, with new rollouts, new pairs and new
draws of dev and holdout each time. The edit is the 4-fit average, paired with its starting model, with 2 samples
per problem:

| starting model | dose | dev (200 problems) | holdout (200) |
|---|---|---|---|
| **stop-thinking (`0009538706-0022`)** | **0.35** | **+5.0 ± 2.4** | **+6.2 ± 2.2** |
| the same | 0.5 | +6.0 ± 2.6 | +2.5 ± 2.5 |
| the same | 0.7 | -0.2 | +2.8 |
| our short-answer edit (V13) on loop-removal, run 1 | 0.5 | +6.5 ± 2.4 | +3.0 ± 1.9 |
| the same, run 2 | 0.5 | +5.8 ± 2.1 | +5.5 ± 2.4 |
| the same recipe from another run (`0009538688-0022`) | 0.5 | +7.5 ± 2.4 | +3.8 ± 2.2 |
| self-distill-gains (0019) on loop-removal | 0.7 | +3.8 ± 2.4 | +4.5 ± 2.6 |

- Instruction following gained +6 to +24 on every run. On stop-thinking at 0.35: tuluif +18.8 / +20.0, ifsynth
  +12.5 / +6.2 (dev / holdout).
- Math, gsm8k and multiple choice moved -5 to +8, about 0 to +3 on average. On stop-thinking at 0.35, math was
  0.0 / +5.0 and multiple choice -3.8 / +7.5. At 0.5 they fell on dev (math -3.8, multiple choice -7.5) and gsm8k
  fell on holdout (-8.8). Hence the lowest dose within one SE of the best (step 7).
- What the start already does keeps working: stop-thinking's gate still closed the thinking (answers cut at the
  token limit stayed at 1-3%).
- A single fit of the same pairs ranged from -7.5 to +2.5.

## What did not work (measured, do not repeat)
- **A single fit** (see above), and **too large a dose**: k = 1.0, k = 0.7 on a short-answer start, or k = 0.5 and
  more on stop-thinking, cost math and multiple choice.
- **Holding the thinking's likelihood fixed during training** (a tether): -4 to -10, instruction following -6 to -21.
- **DPO-Positive** (penalising any drop of the passing answer's likelihood): smaller side effects, but an even
  smaller instruction-following gain. Overall below plain DPO.
- **Plain fine-tuning on the passing answers only** (no preference): weaker and less consistent.
- **Replay anchors** (the model's own correct math / multiple-choice answers mixed into training): the gain vanished.
- **Keeping the neurons most used by other skills out of the edit**: no consistent change.
- **Branching inside wrong math answers, or answer branching on math / multiple choice**: almost no mixed outcomes
  on an ungated start (the answer there is decided by the thinking), so nothing to learn.
- **Adding math, gsm8k and multiple-choice pairs on stop-thinking.** Its gate closes the thinking early, so 4-10% of
  those answers become mixed (122 pairs, mostly a right answer that is sometimes lost). Training on them together
  with the instruction-following pairs cut the gain to about 0 (dev +0.5, holdout +1.0) at every dose.
- **Next-step fragment pairs and conclude-or-continue pairs from run-on traces:** harmful, or too few.

## What to report (REPORT.md)
- The starting model and its sid.
- Step 3's diagnosis per source, and the pair count.
- The neurons per layer.
- Each dose's dev result per source with SE, and the chosen dose's holdout result.
- What was uploaded, and when (verify output included).
- Everything that did not work.
