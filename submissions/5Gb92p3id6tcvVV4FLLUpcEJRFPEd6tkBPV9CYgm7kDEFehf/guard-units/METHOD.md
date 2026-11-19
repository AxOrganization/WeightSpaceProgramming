# One temperature cannot serve loops and answers, and a forbidden comma slips at the draw

This method stacks on a published edit (`entry.json` `parent`). It changes the next-token draw in two states that
the model itself marks, and nowhere else: inside a loop it adds a vector that favours no token and only makes the
draw hotter, and in the answer to a request that forbids commas it takes a fixed number of logits from the comma
tokens. Nothing is trained and no direction is removed: every number in the edit is computed during the run from
traces of the starting model on practice problems (a mean, a covariance, an eigenvector, a linear solve) and
written into two or four hardly used MLP units of the last decoder layer (`set_rows` on `gate_proj` and `up_proj`,
`set_cols` on `down_proj`).

The figures below are the author's measurements on practice problems, paired with the starting model by problem
and seed, on two published starts: the parent named in `entry.json`, whose answers no longer loop, and another
published edit whose answers still do ("the looping start"). Each figure says which start it belongs to. The method
has a branch for each kind of start, and step A measures which kind the run's start is. On another start the figures
will differ; the steps say what to do then.

## 1. The behaviour to change, and why it is there

**(a) Loops, and the one temperature of the sampler.** On the looping start, with the profile's sampler, 76 of 1280
practice answers ended in a loop at the token limit. Drawing the same model at other temperatures (a diagnostic
only):

| temperature | accuracy | looping cuts | other cuts | accuracy of finished answers |
|---|---|---|---|---|
| 0.3 | 64.5 | 209 | 20 | 78.4 |
| 0.45 | 67.6 | 133 | 30 | 77.4 |
| 0.6 (profile) | 68.0 | 76 | 62 | 76.1 |
| 0.8 | 70.1 | 25 | 69 | 75.6 |
| 1.0 | 66.5 | 12 | 99 | 72.7 |

A hotter draw leaves loops; a cooler draw slips less everywhere else. One temperature for every position cannot
have both: the profile's value is a compromise. Suspected cause: inside a loop the repeating continuation is far
more likely than any way out, so the draw needs more randomness exactly there and nowhere else. The model marks the
state: on practice traces a linear read of the last layer's MLP input fires on 92% of the positions inside a loop
and on 0.5% of the others.

**(b) Commas against the instruction.** Some practice requests (`ifsynth`, `tuluif`) forbid commas in the answer.
On the looping start 36% of the finished answers to them fail, and 36 of those 44 failing answers hold only one to
three commas. On the parent named in `entry.json`, 66 of 149 such answers in the building traces fail, every one of
them holds a comma, and 59 hold only one to three. The thinking shows that the model has read the instruction;
while it writes the answer, the habit token wins a few draws. This state is marked too: a linear read fires on 97%
of the answer positions of such requests and on 0.9% of all other positions. `rule.py` prints these counts for the
run's start before it builds anything, and stops when there is nothing to repair.

**Why the last layer.** Both writes are defined on the final residual `h` itself. The heat write moves no token by
itself: it works only because the final norm divides by `rms(h + w)`, and that division happens once, after the
last layer. The comma write is a solve against the head as that norm scales it, with `rms(h)` measured at the
positions where the gate fires. Neither quantity exists at an earlier layer. The playbook says never to edit the
last layer's MLP, because it is entangled with next-token prediction: that entanglement is the lever here, and the
control of section 2 measures its price on the problems the rule should leave alone. The playbook also lists a
global change of the effective temperature as tried without gain. Section 7 confirms it: the same write at every
position gains nothing that can be told from noise. The gain comes from the gate.

## 2. The two units

**Heat (`derive/thermo.py --gated`).** The logits are `E (g * h / rms(h))`: the head applied to the final residual
`h` after the last RMS norm (`g` is the norm's gain, read off the module). The norm divides by the size of `h`. Add
to `h` a vector `w` that the head reads as nearly the same number for every token: no token is preferred, but
`rms(h + w) > rms(h)`, so every logit difference shrinks by `s = rms(h) / rms(h + w)`. With the sampler unchanged
this is a draw at temperature `T / s` and presence penalty `pp / s`, at the positions where the unit fires.
- Neutral direction `v`: the unit vector that minimises the variance of `E (g * v)` over the vocabulary (weights:
  half the frequency of the generated tokens, a tenth on the tokens that end the thinking or the answer or break a
  line, the rest uniform) plus a small penalty on `E[(h . v)^2]`: the smallest eigenvector of a 1024 x 1024 matrix.
- Size: `beta` such that the mean of `rms(h) / rms(h + beta v)` over sampled positions is the chosen scale `s`.
- Gate (the toolkit's `wsp.gate.build`): LDA read of loop positions (positions of looping answers where at least
  70% of the last 64 16-grams had occurred before in the same answer) against positions of right answers and
  positions whose repeat score is under 0.2; threshold above 99.5% of those.

**Comma rule (`derive/rule.py`).** The instruction itself names what is banned: every token whose text holds a
comma. Nothing is searched or ranked. The gate is one linear read. Its positives are the answer positions (after
the thinking has closed) of requests whose instruction list holds it; its negatives are the thinking of the same
requests, where the model may plan with commas, and all positions of requests without the instruction and of the
other practice sources. The requests with the instruction are split by a hash of their id: 60% to build the unit,
40% kept out to test it. Write: the vector that lowers the logits of the 24 most frequent banned tokens by exactly
`delta` and moves the other tokens as little as possible (a frequency-weighted least-squares solve with 24 equality
constraints, through the final norm and the head).

Each write is capped by a second unit (same read, shifted threshold, opposite write): a looping position cannot
get hotter than the cap, and every firing answer position gets the same comma push.

**Which units.** Each builder overwrites two units that the starting model hardly uses: among the units whose
largest write at the sampled trace positions is in the lowest tenth of the layer, the two with the smallest mean
write (about 0.1% of the residual norm on average). The control below measures what silencing them costs.

Before any generation the builders print what to check: `recall` and `neg_fire` of the gate, the realised scale
and its spread, the logit shift of the comma tokens, the largest shift any other frequent token gets, and the size
of the new gate rows and write columns relative to their matrices.

**The control (`derive/null.py`).** Every problem is drawn with fixed seeds, and any change of the weights changes
which answers those seeds give. An edit that acts on few problems is therefore compared with the starting model
across re-drawn answers on all the others, and where the starting model's one draw was lucky, every edit looks
worse. `null.py` makes the same two units silent and writes nothing: the start, re-drawn. The comma rule is judged
against it on the problems it should leave alone.

## 3. Setup (first 20 min)

The bundle holds the entry's parent in `parent/edit/` and, when the champion is another edit, that one in
`champion/edit/`. On the agent machine, from the unpacked bundle (the brief says to copy `task.json` to the sandbox):

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r parent sandbox:/workspace/work/        # gives /workspace/work/parent/edit/edit.json
if [ -d champion ]; then scp -F ~/.axonry/ssh/config -r champion sandbox:/workspace/work/; fi
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`):

```
cd /workspace/work && mkdir -p w
cat > w/env.sh <<'EOF'
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
T="w/prof_0.jsonl w/prof_1.jsonl"
C=gparent                                  # the chosen case; step A may change this line to gchampion
FIRST=C10                                  # the first upload; step B changes this line to G35 on a start that loops
BEST=C10                                   # the edit to deliver; step D sets this line
EOF
. /workspace/work/w/env.sh
for S in parent champion; do [ -f $S/edit/edit.json ] || continue
  python -m wsp.case init g$S --profile qwen35-0.8b-think --parent $S/edit/edit.json \
      --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 320 --holdout 640 \
      --subset holdout320=holdout:320
  python -c "from wsp import case; case.model_path('g$S','base')"   # build each start ONCE, before any GPU job
done
```

**Run `. /workspace/work/w/env.sh` first in every new shell, ssh call and tmux window**: the commands below rely on
`$C`, `$T`, `$FIRST` and `$BEST` and on the working directory (bash; the two file names in `$T` are split on the
space). Both cases get the same problems (same seed), so their runs are paired by problem and sample. Read the
`REPORT.md` next to each start's `edit.json`. Rules for all long work:

- Every generation job runs detached (`nohup ... &` as written below, or `tmux new -d`) with its own
  `CUDA_VISIBLE_DEVICES`, one job per GPU (a `wsp.case run` given two GPU ids splits its problems over both). Its
  output file appears only when the job ends.
- Before each start, look at `nvidia-smi`: the GPUs you name must hold no memory. An engine left over from a job
  that died keeps its memory, and the next job on that GPU fails while it starts. Stop a leftover by its PID
  (`nvidia-smi` lists it), never by a name pattern.
- Check the jobs every 10 to 15 min (`tail` the log) and keep `PROGRESS.md` current. A job whose log ends in a
  traceback is started again once; if it fails the same way, go on without it and say so in REPORT.md.
- Use only `practice:` data and the starting model's own generations.

## 4. Steps

**Step A: pick the start, then draw the traces (about 70 min)**

```
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run gparent base dev --n 2 > dev_parent.log 2>&1 &
if [ -d cases/gchampion ]; then CUDA_VISIBLE_DEVICES=2,3 nohup python -m wsp.case run gchampion base dev --n 2 > dev_champion.log 2>&1 & fi
```

With one start (no `cases/gchampion`) there is nothing to choose and GPUs 2 and 3 are free: run the two
list-making lines of the block further down now and start its GPU 2 and GPU 3 jobs at once; its GPU 0 and GPU 1
jobs follow when the dev run has ended. With two starts, wait for both runs (about 30 min; each `[gen] ... tok/s`
line of a log is the generation rate of one GPU, see section 5), then:

```
python derive/pair.py $WSP_WORK/runs/gparent/base__dev.jsonl $WSP_WORK/runs/gchampion/base__dev.jsonl
```

`pair.py A B` prints the gain of B over A with its standard error. The parent is the start this method was
prepared on: stay with it unless the champion's case beats it by more than that standard error. Only then change
the `C=` line of `w/env.sh` to `C=gchampion` and source it again.

Read for the chosen case, from `cases/$C/results/base__dev.json`: `summary.acc`, `summary.cut_share`,
`summary.by_source` and `summary.kinds`. **`summary.kinds.cut_loop` (absent means 0) decides the path of every
later step: 10 or more of the 640 answers ended in a loop: the start LOOPS. Fewer: it does NOT LOOP**, there is no
heat unit, and of every pair of command blocks below the one headed "does NOT LOOP" applies. Then draw the traces,
one job per GPU:

```
python derive/make_sets.py --case $C --out w --shards 2          # 800 practice problems, never dev / holdout
python derive/rule_sets.py --case $C --out w                     # build / test lists of the comma rule
CUDA_VISIBLE_DEVICES=0 nohup python derive/sample.py --case $C --ids w/train_ids_0.json --out w/prof_0.jsonl --n 2 > t0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python derive/sample.py --case $C --ids w/train_ids_1.json --out w/prof_1.jsonl --n 2 > t1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python derive/sample.py --case $C --ids w/if_build.json --out w/if_build.jsonl --n 2 --salt rb > t2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/sample.py --case $C --ids w/no_comma_test.json --out w/comma_base.jsonl --n 4 --salt rt --max-tokens 32768 > t3.log 2>&1 &
```

`sample.py` is `python -m wsp.gen` with a token cap (8000 by default: it only shortens the drawing) and, with
`--edit NAME`, on an edit of the case instead of its starting model. Each output file appears when its job ends.
Step B needs three of them (`ls -la $T w/if_build.jsonl`); `w/comma_base.jsonl`, drawn at the full token limit, is
needed in step D only and may still be running.

**Step B: build (one GPU, one command after the other)**

The builders run forward passes over the traces and generate nothing: on the author's GPU the whole block of a
looping start took 4 min and `rule.py` alone 1 min. Run them inside `tmux` if your connection may drop.

A start that LOOPS (set `FIRST=G35` in `w/env.sh` and source it):

```
CUDA_VISIBLE_DEVICES=0 python derive/thermo.py --case $C --runs $T --name H35 --scale 0.35 --gated --neg-quantile 0.995 --clamp 1.5 --max-tokens 8000
CUDA_VISIBLE_DEVICES=0 python derive/thermo.py --case $C --runs $T --name H20 --scale 0.2 --gated --neg-quantile 0.995 --clamp 1.5 --max-tokens 8000
CUDA_VISIBLE_DEVICES=0 python derive/rule.py --case $C --runs w/if_build.jsonl $T --exclude w/no_comma_test.json --name C10 --delta 10 --neg-quantile 0.99 --avoid H35 H20
python derive/stack.py --case $C --name G35 --edits H35 C10
python derive/stack.py --case $C --name G20 --edits H20 C10
```

A start that does NOT LOOP (`FIRST` stays `C10`):

```
CUDA_VISIBLE_DEVICES=0 python derive/rule.py --case $C --runs w/if_build.jsonl $T --exclude w/no_comma_test.json --name C10 --delta 10 --neg-quantile 0.99
```

Note what each builder prints. Measured by the author: heat gate `recall` 0.92 with `neg_fire` 0.005, `scale_std`
0.04 to 0.05; comma gate `recall` 0.97 to 0.99 with `neg_fire` 0.008 to 0.009, `scale_where_it_fires` 0.93 to 0.94,
and in its `diagnosis` line 44% of the building answers failing with a comma. The printed `rel_gate` and
`rel_down` are the size of one new row or column against its matrix, not the relative change that the limits
measure: `axonry-check` (step C) gives that. These values are for comparison only: unless a builder stops by
itself, go on whatever it prints. The tests of step D decide.

- `thermo.py` stops with `NO HEAT GUARD` when the traces (cut at 8000 tokens) hold too few loop positions or the
  gate reaches under half of them. Then there is no heat unit: set `FIRST=C10` and go on as for a start that does
  not loop, from the `rule.py` line of that block (its timeline applies from here).
- If `thermo.py` prints `WARNING: no direction keeps the scale ...`, keep the heat unit only if its `scale_std` is
  below 0.15.
- `rule.py` stops with `NO COMMA GUARD` when the traces hold too few answers to requests with the instruction, when
  the start hardly ever breaks it, or when the gate reaches under half of the answer positions. Then there is no
  comma rule. On a start that loops, deliver the heat unit alone: set `FIRST=H35`; in step D run only `H35` and
  `H20` on dev (no `C10` run, no comma test), and a heat edit qualifies if its `cut_loop` is at most half of the
  start's and its `paired.gain` is not below minus twice its `paired.se`. On a start that does not loop there is
  nothing to deliver, and REPORT.md says so.
- `H35` and `H20` take the same two units (they are alternatives); `--avoid` keeps the comma rule off them.
  `stack.py` refuses a stack in which two edits write the same unit.

**Step C: admission check and first upload (by about 1 h 50 min)**

`$FIRST` is `G35` on a start that loops and `C10` on one that does not. Write `/workspace/work/REPORT.md` first
(what was done and found so far; `wsp.upload` reads it), then:

```
python -m wsp.upload $C $FIRST --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/${C}__$FIRST/edit --task task.json
python -c "import json; print(json.load(open('$WSP_WORK/deliver/${C}__$FIRST/edit/edit.json'))['parent'])"
python -m wsp.upload $C $FIRST --task task.json --report /workspace/work/REPORT.md
```

The third line must print `parent/edit.json` when `C=gparent` and `champion/edit.json` when `C=gchampion`: the
toolkit writes the start the case stands on; do not edit it, and stop to find the cause if it prints anything
else. The values are inline in the spec, so there is no `edit.npz`. Measured by the author, the largest relative
change of a tensor is 0.17 to 0.20 for one pair of units and 0.26 to 0.30 for two. If `axonry-check` reports
a tensor above the per-tensor limit for `G35`, upload `C10` instead (one pair of units) and say so; if it does for
`C10` alone, the gate rows of this start are too large for the limit and there is nothing to deliver: report the
figure. This first upload is a safeguard; step D decides what stays.

**Step D: test**

The `for E` line of each block builds every edit's model once, before two jobs use it.

A start that LOOPS (about 70 min; start the GPU 3 job only when `w/comma_base.jsonl` of step A exists):

```
for E in G35 G20 C10; do python -c "from wsp import case; case.model_path('$C','$E')"; done
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run $C G35 dev --n 2 > dev_G35.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run $C G20 dev --n 2 > dev_G20.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m wsp.case run $C C10 dev --n 2 > dev_C10.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/sample.py --case $C --edit C10 --ids w/no_comma_test.json --out w/comma_C10.jsonl --n 4 --salt rt --max-tokens 32768 > t5.log 2>&1 &
```

A start that does NOT LOOP (about 40 min):

```
python derive/null.py --case $C --from C10 --name Z
for E in C10 Z; do python -c "from wsp import case; case.model_path('$C','$E')"; done
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run $C C10 dev --n 2 > dev_C10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run $C Z dev --n 2 > dev_Z.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python derive/sample.py --case $C --edit C10 --ids w/no_comma_test.json --out w/comma_C10.jsonl --n 4 --salt rt --max-tokens 32768 > t5.log 2>&1 &
```

Readings:

- The comma test: `python derive/pair.py w/comma_base.jsonl w/comma_C10.jsonl` (same salt and sample count as the
  reference, so the answers are paired by seed). The rule works if the gain is above twice its standard error.
- Each dev run, from `cases/$C/results/<edit>__dev.json`: `paired.gain`, `paired.se`, `summary.kinds.cut_loop`,
  `summary.by_source`. These are fractions (0.027 is 2.7 points); `pair.py` prints points.
- Without loops, the side effects, against the control (the line `(none)` holds the problems without any
  instruction, which the rule should leave alone):

```
python derive/pair.py $WSP_WORK/runs/$C/Z__dev.jsonl $WSP_WORK/runs/$C/C10__dev.jsonl --by instruction
```

Choose the edit to deliver, write it into the `BEST=` line of `w/env.sh` and source it. On a start that LOOPS, in
this order:

1. A heat stack (`G35`, `G20`) qualifies if its `cut_loop` is at most half of the start's and its `paired.gain` is
   not more than twice its `paired.se` below that of `C10` (the same rule without the heat).
2. `BEST` is the qualifying stack with the higher `paired.gain`; `G35` on a tie within half a point. If neither
   qualifies, the heat does not help on this start and `BEST` is `C10`.
3. If the comma test shows no gain above twice its standard error, the comma rule does nothing here. Deliver the
   heat edit alone: `BEST` is `H35` if step 2 chose `G35`, `H20` if it chose `G20`. Run it on dev once
   (`CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C $BEST dev --n 2 > dev_heat.log 2>&1 &`): step E
   needs that run. If step 2 chose `C10`, neither unit helped on this start: `BEST` stays `C10`, and REPORT.md
   says so.
4. If `BEST` differs from the uploaded `G35`, upload it now (the four commands of step C with `$BEST` in place of
   `$FIRST`).

On a start that does NOT LOOP, `BEST` is `C10`, with two checks:

1. If the comma test shows no gain above twice its standard error, the rule does nothing on this start. There is
   nothing better to deliver: leave the upload as it is, say so in REPORT.md, and go on with step E.
2. The dev gain of `C10` against the starting model is mostly the noise of a re-draw (expect about -4 to +4 points
   at two samples). What counts is the `(none)` line against the control `Z`. If `C10` is below `Z`
   there by more than twice the standard error printed on that line, build the rule again with fewer stray firings
   and test it (about 45 min more):

```
CUDA_VISIBLE_DEVICES=0 python derive/rule.py --case $C --runs w/if_build.jsonl $T --exclude w/no_comma_test.json --name C10b --delta 10 --neg-quantile 0.999
python -c "from wsp import case; case.model_path('$C','C10b')"
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run $C C10b dev --n 2 > dev_C10b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python derive/sample.py --case $C --edit C10b --ids w/no_comma_test.json --out w/comma_C10b.jsonl --n 4 --salt rt --max-tokens 32768 > t6.log 2>&1 &
python derive/pair.py w/comma_base.jsonl w/comma_C10b.jsonl                                  # after both have ended
python derive/pair.py $WSP_WORK/runs/$C/Z__dev.jsonl $WSP_WORK/runs/$C/C10b__dev.jsonl --by instruction
```

   `BEST` is `C10b` only if it passes both checks (comma test above twice its standard error, `(none)` line not
   below `Z` by more than twice its standard error): set the `BEST=` line to `C10b` then and upload it with the
   commands of step C (`$BEST` in place of `$FIRST`). In every other case keep `C10` and report both readings.

**Step E: confirm on holdout, then the control of the heat unit, deliver**

```
python -m wsp.case freeze $C $BEST
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C base holdout320 --n 2 > hold_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m wsp.case run $C $BEST holdout320 --n 2 > hold_best.log 2>&1 &
```

Wait until both have ended (`paired` in the result file is empty while the run of the start is missing), then
read `cases/$C/results/${BEST}__holdout320.json` as in step D. Only if `$BEST` holds a heat unit, its control:

```
CUDA_VISIBLE_DEVICES=0 python derive/thermo.py --case $C --runs $T --name HCONST --scale 0.75 --max-tokens 8000
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C HCONST dev --n 2 > dev_const.log 2>&1 &
```

`HCONST` writes a heat vector at every position instead of inside loops. It removes loops too, but adds cut and
wrong answers elsewhere. It should come out behind `$BEST`, which shows that the gain needs the gate:

```
python derive/pair.py $WSP_WORK/runs/$C/HCONST__dev.jsonl $WSP_WORK/runs/$C/${BEST}__dev.jsonl
```

If `$BEST` holds a heat unit and its holdout gain is below minus twice its standard error, upload `C10` instead and
say so. For `C10` alone the holdout gain against the starting model is a report figure, not a test: it is a re-draw
on nearly every problem (section 2), and the rule was tested in step D. The final REPORT.md covers: which start
was chosen and why; its accuracy, cut share and looping cuts; for each unit the builder's checks, its test and the
decision; dev and holdout gains with standard errors; the controls; what was dropped and why. Upload again with the final REPORT.md
(the last command of step C with `$BEST`), so that the last complete upload is `$BEST`. After this upload nothing
of the method is left to run: end the run as the brief says.

## 5. Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

A start that loops:

| time | GPUs 0-1 | GPUs 2-3 |
|---|---|---|
| 0:00-0:20 | setup, case init for each start, build the starts | |
| 0:20-0:50 | parent on dev x2 | champion on dev x2 (if it differs) |
| 0:50-1:30 | traces shard 0 / shard 1 | `if_build` traces / comma test reference |
| 1:30-1:50 | build H35, H20, C10, stacks; admission check; **first upload (G35)** | |
| 1:50-3:00 | G35 on dev x2 / G20 on dev x2 | C10 on dev x2 / comma test of C10 |
| 3:00-3:10 | choose `BEST`, upload it if it is not G35 | |
| 3:10-3:50 | start on holdout320 x2 | `BEST` on holdout320 x2 |
| 3:50-4:40 | HCONST on dev x2, REPORT.md, final upload | |

A start that does not loop: the same until 1:30; C10 built and **uploaded by 1:45**; C10 and Z on dev and the
comma test until 2:25; holdout320 of the start and of C10 until 3:05; REPORT.md and final upload by 3:20.

These times assume about 3,300 generated tokens per second per GPU; they were not measured on 24 GB cards. Each
`[gen] ... tok/s` line of `dev_parent.log` is the real rate of one GPU (a two-GPU run prints two lines). Below
2,000 per GPU, drop `G20` from steps B to D and `HCONST` from step E; nothing else depends on them. Below 1,200,
also give every dev run two GPUs (two at a time instead of four) and leave out the holdout runs of step E if they
would not end an hour before the deadline: the delivered edit is then the `BEST` of step D, and REPORT.md says
that no holdout was run. Whatever the rate, the first upload of step C goes out before anything of step D is
waited for, and it stays the delivered edit if step D cannot be finished.

## 6. What to expect

Measured by the author on the looping start (76 and 102 looping cuts in 1280 dev answers, in two drawings) and on
the parent named in `entry.json`.

- **Heat unit, on the looping start.** Scale 0.35 with the cap, stacked with the comma rule (`G35`), on dev with 4
  samples: looping cuts 102 -> 9 of 1280 answers, other cuts 59 -> 51, accuracy +3.5 +- 1.3 against the start
  and +3.8 +- 1.2 against the comma rule alone. On holdouts: +3.6 +- 1.7 (320 problems x 2) and, for the heat
  unit alone, +2.0 +- 0.8 (640 problems x 4, cut answers 10.1% -> 5.3%). The answers that leave a loop are right
  less often than the others, and the gain moves between sources from one set to the next. Expect a gain of two to
  three points overall and a clear drop of `cut_loop`.
- **A start that does not loop** has 0 to 2 looping cuts in its 640 dev answers; `thermo.py` would stop with
  `NO HEAT GUARD`.
- **Comma rule.** On the 50 requests kept out of the building (4 samples each): 63.0% -> 85.0% of the answers pass
  on the looping start (+22.0 +- 4.2) and 67.0% -> 97.0% on the parent (+30.0 +- 5.5). Over 960 dev and holdout
  problems of the parent (4 samples) the requests with the instruction gain +28.4 +- 7.0, no other instruction
  type moves beyond its standard error, and the 576 problems without any instruction are at -0.3 +- 0.8.
  Against the control, the problems without any instruction read -2.1 +- 1.2 on dev at two samples and
  +1.5 +- 0.9 on 384 holdout problems at four samples, for the same build: the `(none)` line of step D is this
  noisy.
- **The control.** On the parent, 640 holdout problems x 4 samples: the control against the starting model is
  +0.7 +- 0.8 overall, with single sources between -2.5 and +3.3. That is the size of a re-draw, and it is why the
  comma rule alone read -3.3 +- 1.6 against the starting model on dev at two samples in one run and +1.1 +- 1.6
  on holdout320 in the same run.
- One pair of units changes the gate matrix of the last layer by about 0.2 of its norm, the write matrix by 0.01
  and the model by 0.02 overall; its payload is 24 KiB. Two pairs: 0.3, 0.02 and 0.03, 48 KiB.

## 7. What does not work, or not reliably (measured, do not repeat)

Dev set of the looping start, 320 problems x 4 samples, points against the starting model; standard errors 1.1 to
1.4.

| edit or setting | effect |
|---|---|
| The heat vector at every position (constant unit `HCONST`, scale 0.75) | +0.9 +- 1.4: looping cuts 102 -> 13, other cuts 59 -> 78; 2.7 +- 1.3 behind the gated unit |
| Sampler: temperature 0.8 with presence penalty 2.0 (what a uniform logit scale of 0.75 equals) | -0.5 |

- Scaling everything a late layer writes (factors 0.8 to 1.5) changes the logit scale by under 10%: the norms
  absorb it. There is no cheap way to make the draw cooler; the heat unit only ever makes it hotter.
- Lowering the end-of-text logit where the heat unit fires did not help; the cap on the write is enough.
- Choosing the two units by their mean write alone can pick one that is silent almost everywhere and writes 15% of
  the residual norm at a few positions. The builders also require the largest write to be small.
- Teacher-forced checks do not predict the effect of the heat unit; choose by paired generation only.
