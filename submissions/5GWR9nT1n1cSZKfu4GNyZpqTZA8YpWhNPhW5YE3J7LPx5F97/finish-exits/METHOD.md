# Teach the model to finish: refit a few MLP units' output weights on its own concise answers and its own exits from loops

## The problem
The starting model (the parent edit, whose loop-removal edit already took out the main loop direction) still often
fails to reach an answer:
- a quarter to a half of its math and multiple-choice answers are cut at the 32k-token limit, most of them in a loop;
- finished answers are far more accurate than answers overall;
- most problems that are cut in one sample are solved, much more briefly, in another sample.

The ability is there. The model does not reliably take its own path to the end. The chain below measures all of this
first on DEV, which uses practice problems only, never the benchmarks.

## Why we think it happens
- **The model does not conclude.** After the reasoning has done its work, the model keeps re-checking. It then either
  falls into a loop that restarts the same block, or keeps doubting ("Wait, let me re-read...") until the limit.
- **Removing loops alone does not help.** Suppressing repetition turns the loops into long non-repeating doubt that
  still hits the limit. What is missing is the decision to close the thinking and answer.
- **The model can make that decision.** When its thinking is closed at the point where it starts to repeat itself,
  it often still answers correctly. Those continuations are its own way out of the loop.

A small set of MLP units should carry the choice between going on and concluding. Rewriting their output weights,
fitted to the model's own concluding continuations, should shift it.

The toolkit's `conclude` direction edit (playbook: tried and not confirmed) ablated one direction, and that lowered the
closing continuation too. Here no direction is removed: the selected units' output weights are fitted to the model's
own closing continuations.

## Approach
Code in `derive/`, using the toolkit. The chain `derive/chain.py` carries out the skill's steps 1-6 for this method,
and `python submission/derive/chain.py ax upload` (through `derive/deliver.py`) is its step 7. Apart from the step-0
`init` below, run none of the skill's own commands for these steps (`run`, `pairs`, `extract`, `edit`, `selectivity`,
`freeze`, `wsp.upload`). The skill's ground rules still hold: practice data only, choices on DEV, HOLDOUT only after
the freeze.

All GPU work runs in one detached chain, `derive/chain.py` (step 4). Never start a GPU job by hand. Steps 1-3, 5 and 6
are run by that chain; what they do is listed so that you know what runs and can report it. Work in `/workspace/work`.
From the unpacked bundle, copy the three parts there:
```
scp -F ~/.axonry/ssh/config -r submission parent task.json sandbox:/workspace/work/
```
so that `/workspace/work/parent/edit/edit.json` and `/workspace/work/task.json` exist.

0. **Case**, stacked on the parent, with the four practice sources the toolkit ships:
   - `gsm8k_train` and `math_train`: the GSM8K and MATH train splits;
   - `mmlupro`: MMLU-Pro questions outside the evaluated 500, which the toolkit checks share no question with the
     benchmarks;
   - `ifsynth`: GSM8K train questions with IFEval-style constraints.

   You run this one yourself:
   ```
   cd /workspace/work && python -m wsp.case init ax --profile qwen35-0.8b-think --parent parent/edit/edit.json \
     --practice gsm8k_train math_train mmlupro ifsynth --subset holdout256=holdout:256
   ```
   It prints `{'dev': 256, 'holdout': 512, 'holdout256': 256}`. DEV is the case set `dev` (64 problems per source).
1. **The model's own answers** (run by the chain):
   - `mkdata.py ids --per-source 300`, then `shard_gen.py --model base --n 4 --max-tokens 16384`: 1200 practice
     problems outside DEV and HOLDOUT, 4 samples each, on all GPUs;
   - `evaluate.py <case> base dev --n 2`: the starting model's DEV run, which pairs every comparison;
   - `ifcheck`: 200 further ifsynth problems outside DEV, HOLDOUT and the pool, kept for step 5.
2. **Exits from loops** (`exits.py --per-source 300 --n 2 --sources gsm8k math mmlupro`):
   - every answer cut in a loop is cut at the sentence before the point where it starts repeating itself, and the
     thinking is closed there (`</think>`, the only inserted text);
   - the model answers twice;
   - the continuations that the toolkit's grader marks correct against the practice problem's own answer become
     records: the model's own way out of that loop.

   Instruction-following problems get no exits. In development, exits taken from them taught the model to close its
   thinking early on every instruction-following answer, before it checked the constraints. Without them,
   instruction following stayed level, and the exits from math and multiple-choice loops still carried over.

   No benchmark item is read at any step.
3. **Records** (`mkdata.py data --extra exits.jsonl --answer-only-sources gsm8k math mmlupro --loop-ratio 0`):
   - the exits, learned from the exit on;
   - each problem's shortest correct answer: its answer part only for gsm8k, math and mmlupro, and in full,
     thinking included, for ifsynth.

   Learning the whole thinking of the math answers made the model more deterministic and loop more. Learning no
   thinking at all made it skip the planning that instruction following needs. This split avoids both.
4. **Run the chain: find the units and solve for their new output weights, once per seed.** Right after step 0, from
   `/workspace/work`:
   ```
   python submission/derive/chain.py ax start
   ```
   - It returns at once. The chain runs detached in its own session and writes one line per event to
     `logs/status.log`.
   - It reads the deadline from `/etc/axonry-deadline`. If `start` says the deadline is not usable, start it as
     `DL=$(date -d '<deadline date and time from the brief> UTC' +%s) python submission/derive/chain.py ax start`, with
     the date and time only (the word UTC once).
   - For seed s = 0, 1, 2, ..., one GPU job at a time on all GPUs:
     ```
     torchrun --nproc_per_node 4 submission/derive/train.py --case ax --data train.jsonl --name S<s> --seed <s> \
       --alphas 1,1.5,2 --lr 5e-5 --epochs 3 --batch 64 --ema 0.95 --time-limit 20
     python submission/derive/evaluate.py ax S<s>_a1.5 dev --n 2
     ```
     Each is followed by the rule of step 5 (`derive/rank.py`).

   What train.py does:
   - **Units.** The first stage takes the gradient of the objective with respect to every down_proj column of every
     decoder layer except the last. That exclusion is the playbook's rule: the last layer's MLP is entangled with
     next-token prediction. The 1024 columns with the largest gradient are the units whose output weights matter
     most for this choice. Report where they are (`[select] units per layer` in `logs/train_S<s>.log`).
     `--units` sets how many units, never which: 2 KiB of payload per unit in float16, so 1024 units use half of the
     4 MiB limit.
   - **Fit.** Only those columns change (`set_cols`). They are fitted to raise the likelihood of the records under
     the model's own sampling distribution: the profile's temperature and presence penalty, so the fit does not just
     make the model more deterministic. Everything else is frozen.
   - **Averaging.** A low learning rate over 3 epochs, and an average of the solution over the last steps
     (`--ema`). Without the average, whether the exit was learned depended on the order of the records.
   - **Strengths.** `--alphas` exports the solution at several strengths without retraining: `S<s>`, `S<s>_a1.5`
     and `S<s>_a2`. Strength 1.5 is screened; the others are used only as step 5 says.

   Why several seeds:
   - The seed sets the order of the records and the records whose gradient picks the units.
   - In development, with this recipe, every seed learned the exit. At strength 1.5: HOLDOUT (never used for any
     choice) +5 to +8 points, cut share down by about a third, instruction following up. With earlier recipes, some
     seeds did not.
   - The chain therefore keeps training seeds while the time allows, and picks by a fixed rule.

   Timing:
   - About 26 minutes per seed: 11 to train, 13-15 for DEV.
   - A seed starts only if its DEV run can end before the time that steps 5 and 6 need.
   - Every step has a time limit, so a hung job is killed and the chain goes on.
   - Every output is kept: `start` again resumes where the chain stopped, and a second chain is refused.
5. **Choose by a fixed rule** (`derive/rank.py`). The chain runs it after every DEV run, and
   `python submission/derive/chain.py ax status` shows its table. For each candidate, in points, from its DEV result paired
   with the starting model's DEV run:
   - r = the edit's cut share divided by the starting model's: how strongly the exit is learned;
   - g = the overall DEV gain, plus the change per source; nonIF = the mean change on gsm8k, math and mmlupro.

   Classes:
   - **GOOD**: 0.25 <= r <= 0.80, g >= 3, no change on gsm8k, math or mmlupro below -9, and ifsynth not below -12.
   - **OVER**, r < 0.25: the model closes its thinking too early. The chain then also evaluates the seed's strength-1
     export (`S<s>`), which enters the table as its own candidate.
   - **WEAK** with 0.80 < r <= 1: the exit was not learned. The chain then also evaluates `_a2`.
   - **SAFE**: not GOOD or OVER, g > 0, the same two floors.

   The floors are about 2 standard errors per source (128 samples each), so they only remove collapses.

   PICK is the best GOOD by g, else the best SAFE, else the best g. `final.txt` always names it.

   **IF check.** Instruction following is the benchmark this edit puts most at risk.
   - The two best GOOD candidates of different seeds run on `ifcheck` (2 samples, 400 answers), after the starting
     model.
   - IFp = the ifsynth change on DEV and on ifcheck, pooled by sample count.
   - T = the mean answer length on ifcheck divided by the starting model's. A collapse of the thinking shows as a
     small T.
   - IF-OK means IFp >= -4 and T >= 0.35.
   - FINAL is the IF-OK finalist with the larger S = 0.5·nonIF + 0.5·IFp. If no finalist is IF-OK, FINAL is the one
     with the larger IFp. With no GOOD seed, FINAL is PICK.

   No other strength, recipe, seed average or manual choice is ever used.

   Uploads. The chain uploads with `deliver.py` (step 7), CPU only, in the background, one upload at a time, and
   never with less than 50 minutes left:
   - the first PICK that is GOOD or SAFE;
   - the first GOOD, if the uploaded edit is not GOOD;
   - FINAL when it is chosen.

   The control is the seeds or strengths that do not pass. Report them next to the chosen edit.
6. **Freeze and confirm** (the chain does this):
   - `python -m wsp.case freeze ax <FINAL>`;
   - if the time allows, HOLDOUT: `evaluate.py ax base holdout256 --n 2`, then the same for FINAL. HOLDOUT is for the
     report only and never changes the choice;
   - `python -m wsp.case report` writes `cases/ax/RESULTS.md`;
   - the chain writes `DONE` to `logs/status.log`.
7. **While the chain runs: the report and the upload.**
   - **Check on the chain** every 10 minutes, with one command of at most 9 minutes:
     ```
     cd /workspace/work && python submission/derive/chain.py ax status --wait 8
     ```
     It returns as soon as the chain reports something new, after 8 minutes at most. If your shell tool stops
     commands earlier, use a smaller `--wait`. Keep working between checks (the report, PROGRESS.md) and do not end
     your turn just to wait.
   - **PROGRESS.md.** Keep it current; its first line is the status. Write the start command and the upload command
     `python submission/derive/chain.py ax upload` into it.
   - **Never** do any of these:
     - start a GPU job by hand;
     - change the chain's settings;
     - choose on HOLDOUT;
     - upload with `python -m wsp.upload`. Its export (`wsp/edits.py`, `export`) copies only the arrays behind `u`,
       `cols` and `rows`, so a set_cols edit would arrive without its `values` and fail admission ("edit arrays
       unreadable");
     - print task.json, which holds credentials.
   - **If `status` says `chain NOT running` before `DONE`**, read the last lines of `logs/status.log` and the log it
     names. Fix only an environment problem (a missing package, a full disk), then run `start` again.
   - **REPORT.md.** Write `/workspace/work/REPORT.md` during the screen. Write it to a temporary file, then `mv` it
     into place, because the chain uploads it. It covers:
     - the failure targeted (the starting model's DEV: cut share, loops, accuracy of finished answers);
     - the records and the exit accuracy per source (`logs/exits.log`, `logs/data.log`);
     - the units per layer;
     - the table of every seed with the rule (`logs/select.log`);
     - the IF check (`logs/final.log`);
     - HOLDOUT;
     - the control;
     - what did not work.

   **Uploads go through `derive/deliver.py`** (`chain.py ax upload` calls it):
   - It follows the brief's upload protocol (axonry.upload/1). It calls `wsp.upload`'s own `package`, `target` and
     `upload` unchanged, and uploads only `edit/edit.json`, `edit/edit.npz` and `edit/REPORT.md`, then
     `manifest.json`.
   - It makes one addition: it copies the `values` arrays of the set_cols ops into `edit/edit.npz`.
     `python -m wsp.upload` exports only the arrays behind `u`, `cols` and `rows`. A set_cols edit uploaded with it
     would point its `values` at a file that is not uploaded. Admission rejects it, and as the last complete upload
     it would replace the good one.
   - It runs `axonry-check` on the package and uploads only if it passes; the check's output is in
     `logs/deliver.log`. One upload runs at a time (`upload.lock`).

   Every upload of this method's edits is made with `python submission/derive/chain.py ax upload`: the chain's early
   uploads, the final one, and any upload the brief or a reminder asks for. It follows the brief's protocol and sends
   the same three files and manifest to the same place as `python -m wsp.upload`; it adds only the `values` arrays and
   the `axonry-check` run above.

   **At the end**, once the chain says `DONE` (or at the 30-minute reminder at the latest):
   1. Finish REPORT.md.
   2. Run `python submission/derive/chain.py ax upload`, which uploads FINAL with your report. Start it no later
      than 20 minutes before the deadline. An upload still in progress at the deadline does not count.
   3. Run `axonry-done`.

## Time plan (6 h, 4 GPUs)
| Elapsed | Step |
|---|---|
| 0:00-0:10 | Copy, case, `chain.py ax start` |
| 0:10-1:05 | Pool sampling (1200 problems x 4) |
| 1:05-1:20 | DEV of the starting model; ifcheck ids |
| 1:20-1:30 | Exits; records |
| 1:30-~4:20 | Seeds, about 26 min each, each followed by the rule. First upload as soon as one passes |
| ~4:20-4:45 | IF check of the two best GOOD seeds; FINAL; upload |
| ~4:45-5:15 | Freeze; HOLDOUT (report only) |
| 5:15-5:40 | REPORT.md, `chain.py ax upload`, `axonry-done` |

## What to expect
- Far fewer answers cut at the limit.
- Large gains on math and multiple choice, where most cut answers were.
- Correct answers several times shorter.
- Instruction following level.
- In our measurements on practice DEV and HOLDOUT, about +5 to +9 points overall, with math gaining most.
