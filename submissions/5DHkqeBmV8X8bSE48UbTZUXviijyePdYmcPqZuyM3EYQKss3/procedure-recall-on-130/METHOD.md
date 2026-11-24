# Procedure recall for maths, on the published model of 0009576150-0005 (#130)

This entry adds one layer, **R**, to the published model of its declared parent `0009576150-0005` (#130,
"own-close-carried-as-change"; `entry.json`), and uses that model as it is. R is derived in the run from the toolkit's
practice sets and the model's own generations: the model's thinking is started with one authored sentence that states
a way of working; its own right continuations, placed after the plain question, are distilled into 128 rewritten MLP
units, so that it starts its maths thinking that way unprompted, and nowhere else. The upload is R alone, stacked on
#130's published edit. One driver, `derive/run_a.py`, runs every step and the upload (section 4).

```
Fixed settings (derive/run_a.py and the scripts it calls; the run sets only --gpus)
start: the declared parent's published edit, 0009576150-0005/edit/edit.json, as it is (START_FROM_PARENT = True)
R    : sentence 'concise'; 1024 practice maths problems x2; rows of 160 tokens, at most 2 per problem; keep rows on
       keepNM (383) + keepPZ (456), weight 3; 128 units outside every unit the start's chain rewrote, never the last
       layer; lr 1e-4, accum 8, S = 2 x training rows / 8
gate : math_hold > 0, puzzle_hold >= -max(1 point, 2 se), dev256 >= -2 points, each paired with the start
other: no other layer is fitted (NO_VO = True); if R is not GO, nothing is uploaded
```

## 1. The start

The start is the edit that `0009576150-0005` (#130) published, read from the bundle's `0009576150-0005/edit/` together
with the chain it stands on. It is used as it is: none of its ops is changed, left out or applied a second time. The
case of the run starts from it, and the upload names it as its parent, so the delivered edit is R alone on top of it.

What that model is, in our words (#130's METHOD describes it in full; its REPORT.md is in the bundle). #130 stands on
the champion of its time, `0009570269-0024` (#116, "procedure-workings"). It takes the published layer of
`0009569144-0010` (#111, "keep-the-answer-own-close": fixed rules - wrap-up units silenced, exit, break and commit
units, four direction removals on attention outputs - and a fitted round that favours answers which keep the result
they found and close their own thinking) and adds it to the champion as a change to the weights rather than as their
values, so that neurons touched by both layers keep both changes; the round goes in at half size and stays off the rows
the champion's own set ops write. On that merged model it fits one gain per MLP unit and per attention head that
prefers instruction answers keeping their request: the request-gain method of `0009552831-0015` as published by
`0009568726-0005` (#106). Its own practice checks decide what its final upload holds; the bundle holds that edit. The
champion of the moment is not used: the driver only checks that its directory is present and names it in the log.

## 2. The failure R addresses, and why a recalled procedure

**The behaviour.** On the harder practice maths problems, the starts of this line that we explored work longer than
the problem needs: they check a step they have already checked, again and again, and run out of room, or keep going
after a result is checked instead of writing it in `\boxed{}`. #130's layers act on where and how the thinking
closes and keeps its result (the carried layer) and on the visible answer to instruction requests (the request gains);
R acts earlier, on how the maths working is set up from its first line.

**Why.** A way of working is missing, not knowledge: the same model gets more maths problems right when its thinking
starts from a short procedure in its own voice, "First, recall how to work efficiently: I keep the working short, do
not repeat a check I have already done, and once a result is checked I write it in \boxed{}." (`derive/recall_gen.py`,
`PREFIX['concise']`). The sentence states a general way of working; it names no question, option or hint. Forced on
other kinds of problem it costs, generated puzzles most, so the edit has to carry it on maths only: that is the job of
the keep rows (section 3, R4) and of the puzzle screen (R5).

**Practice evidence** (our exploration, practice splits only, each model paired with its start on the same problems
and seeds). Measured on the start this layer was first written for, the published edit `0009572333-0018`, unless
marked as the in-run rehearsal: that ran these steps on the champion `0009570269-0024` with the first twelve ops of
`0009569144-0010` (its fixed rules) placed on it, the twelve ops #130 carries at their full change. On this entry's
start, R5 and R6 decide.
- *Forced recall, no fit* (two sets of 384 maths problems x2): +3.0 and +4.6 points with this sentence. A longer
  wording (restate what is asked, keep exact values, check against every condition) did about as well, 65.8% -> 69.3%
  (+3.5 +- 1.5) and 62.2% -> 65.5% (+3.3 +- 1.3), with its gain on the hard problems (level 4: 62% -> 68%, level 5:
  36% -> 42%) and fewer answers running out of room (5.6% -> 3.3%); the short sentence is used because it did as well
  and is shorter.
- *Forced on everything*: generated puzzles 89.1% -> 77.3%; multiple choice about the same (+0.4 / -3.3 on two sets of
  256); word problems about the same (-0.1 on 512); instruction requests lower (a constraint procedure there: -5.3).
- *The fit, first version* (concise): 1,213 fit rows from 677 problems and 384 keep rows (the non-maths answers; the
  456 puzzle keep rows came later, to keep the procedure off puzzles); held-out loss 1.21 -> 0.15.
- *In-run rehearsal* (64 units): forced recall right on 66.8% of 2,048 answers; 1,203 fit rows from 678 problems and
  839 keep rows; held-out loss 1.24 -> 0.157 (the 64- and 128-unit fits end alike there, 0.153 / 0.152; this entry
  fits 128). Its in-run screens: `math_hold` 66.1 -> 68.5 (+2.4 +- 1.1), `puzzle_hold` 89.3 -> 88.7 (-0.7 +- 0.8).
- *The fitted units* (concise, 64): maths (384 x4) +3.5 +- 1.1; generated puzzles about the same (+1.7 / +0.3 for two
  fits, 256 x4); the practice mix +0.0 (256 x4); 512 word problems +0.3. The edited model writes the procedure on none
  of the puzzles.
- *Did not work*: the start's own right maths answers distilled without the sentence (mixed hold-out -1.3 / -2.2,
  maths lower too); group-relative credit on its own maths answers (-0.5 to -2.2, more answers out of room); a
  procedure for instruction requests (-5.3) or for word problems (-1.2); units chosen without excluding the start's
  units (the fit overwrites part of the start).

**What sets it apart.** The fitted target is the model's continuation of an authored procedure, not a choice between
its own right and wrong answers: no pair, no preference and no answer of the model without the sentence enters the
fit; the sentence is what carries it (without it the same pipeline lowered the mixed hold-out, above). It leaves the
start's layers in place: its units lie outside every unit the start's chain rewrote, and each row it writes is the
start's own row (any gain of the start included) plus the fitted change. **What is changed:** the `gate_proj` and
`up_proj` rows and the `down_proj` column of 128 MLP units (`set_rows` / `set_cols`).

## 3. How the edit is found (R1-R6, as `derive/run_a.py` runs them)

`$S` = `0009576150-0005/edit/edit.json` (the driver resolves it to an absolute path); `R/` = `$WSP_WORK/recall/`;
`<runs>` = `$WSP_WORK/runs/mp`; `<start model>` = the path R1's build prints; `<gpus>` = the driver's GPU list
(`0,1,2,3`); minutes on 4 x RTX 4090.

- **R1** (20; mostly the start's build on the CPU). `python -m wsp.case init mp --profile qwen35-0.8b-think --contrast
  conclude --parent $S --practice gsm8k_train math_train mmlupro ifsynth bbhsynth --dev 512 --subset dev128=dev:128
  --subset dev256=dev:256 --subset holdout256=holdout:256`; the start's build (`case.model_path('mp', 'base')`);
  `python derive/recall_sets.py mp`, outside the case's dev and holdout splits: `recall_pool` (1024 `math_train`
  problems), `math_hold` (384 other maths problems, none with a pool problem's text), `keepNM` (383 not maths: 96 word
  problems, 128 instruction requests, 64 multiple choice, 5 puzzles of each of the 19 families), `keepPZ` (24 more
  puzzles per family, 456), `puzzle_hold` (10 per family, 190; never fitted or kept). Check: "R1 start built"; the
  five sizes.
- **R2** (40). Not started with fewer than 265 minutes left (section 5). Per GPU i of n, together:
  `CUDA_VISIBLE_DEVICES=i python derive/recall_gen.py mp base recall_pool R/RECALL.i.jsonl --n 2 --var concise --shard
  i/n`: the start's thinking is begun with the sentence, the model continues freely with the profile's sampling, and
  every answer is graded as usual. The shares are joined into `R/RECALL.jsonl`; `python derive/answer_kinds.py
  R/RECALL.jsonl` counts right, finished wrong and cut answers. Check: about two thirds right.
- **R3** (10). `python derive/prun.py mp base keepNM --n 1 --gpus <gpus>`, the same for `keepPZ`: the start's own
  answers. Check: "R3 keep answers: keepNM 383, keepPZ 456".
- **R4** (30-50; the start's screens set the pace).
  *Rows*: `python derive/recall_rows.py mp R/RECALL.jsonl R/ROWS --keep <runs>/base__keepNM.jsonl
  <runs>/base__keepPZ.jsonl --keep-w 3 --tokens 160 --per 2 --drop-mentions`. The right answers are placed after the
  PLAIN question (no sentence in the prompt); the target is the sentence plus the first 160 tokens of the model's own
  thinking, at most 2 per problem; answers that refer to "the hint", "the note" or "the given rule" are dropped; a
  tenth of the problems is held out to measure the fit. The keep answers become keep rows, a KL to the unedited start
  with weight 3: they hold the rest of the model in place, the start's instruction answers and puzzles included.
  *Fit*, on the first GPU: `python derive/units_fit.py train <start model> R/ROWS.train.jsonl R/FIT --heldout
  R/ROWS.held.jsonl --rank 0 --units 128 --skip-last-mlp --exclude-units $S --lr 1e-4 --accum 8 --steps S --warmup 2
  --max-tokens 1536 --eval-every 20 --log-every 10`, S = max(4, 2 x training rows / 8). The 128 units are those with
  the largest loss gradient on the rows, outside the last layer and outside every unit that the start's chain rewrote
  (`--exclude-units` reads `$S` and every link under it: units written by `set_rows` / `set_cols`, or by a `dir_ablate`
  restricted to their rows or columns). Beside the fit, on the other GPUs, the start's three screens one after the
  other: `prun.py mp base math_hold --n 4`, `dev256 --n 2`, `puzzle_hold --n 4`.
  *Export*: `python derive/units_fit.py export R/FIT mp PROC128 --model <start model>` (each written row and column is
  the start's own plus the fitted change), the toolkit's model digest, the build of `PROC128` on the CPU. The driver
  also initialises a second case, `kr`, on `PROC128` and builds it on the CPU; nothing in this entry reads it.
  Check: about 1,200 fit rows from about 680 problems and about 840 keep rows; "excluded units of the parent chain: N
  scores zeroed" with N > 0; the held-out loss falls from above 1 to below 0.2.
- **R5** (30). With fewer than 185 minutes left R is dropped here (section 5). `prun.py mp PROC128 math_hold --n 4
  --gpus <gpus>`, then `puzzle_hold --n 4`, each followed by `python derive/grid.py table mp <set>`: the toolkit's
  paired result against the start on the same problems and seeds. GO = `math_hold` above the start, and `puzzle_hold`
  not below it by more than max(1 point, 2 standard errors of the paired difference), so that a screen's noise alone
  does not drop the layer. Check: "R5 PROC128 vs the start: math_hold ..., puzzle_hold ... -> GO".
- **R6** (15). `prun.py mp PROC128 dev256 --n 2 --gpus <gpus>` and its table: the practice mix must not lose more than
  2 points against the start; then `python -m wsp.case freeze mp PROC128`. Check: "R6 dev256 ...: frozen PROC128;
  R = GO".
- **Upload** (about 20, on the CPU), as soon as R is GO. The report `$WSP_WORK/runa/REPORT_R.md` (the start; R's
  forced-recall share, rows, fit line and screen tables; `wsp.upload` adds the case's results tables); `python -m
  wsp.upload mp PROC128 --report $WSP_WORK/runa/REPORT_R.md --task task.json --dry-run` (the package); the driver's
  check that the package names `0009576150-0005` as its parent and carries exactly R's ops; `axonry-check
  $WSP_WORK/deliver/mp__PROC128/edit --task task.json`; the same upload command without `--dry-run`. Check: "package
  mp/PROC128 ok: parent 0009576150-0005, ... stacks on 0009576150-0005"; "axonry-check mp/PROC128: passed";
  "UPLOADED mp/PROC128"; then "DONE: mp/PROC128 ...".

## 4. Running it (4 x RTX 4090, 6 hours)

**Copy** (agent machine, from the unpacked bundle; each start directory under its id, next to task.json):
```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/      # the declared parent: the start
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/        # the champion, when task.json lists it
```
task.json holds the upload credentials: it stays out of PROGRESS.md, notes and logs.

**Launch** (sandbox; it sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`, otherwise
`export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp`):
```
cd /workspace/work && mkdir -p "$WSP_WORK" "$WSP_CASES"
setsid nohup python derive/run_a.py --gpus 0,1,2,3 > "$WSP_WORK/run_a.out" 2>&1 < /dev/null &
tail -n 5 "$WSP_WORK/runa/status.log"
```
The driver writes one line per step to `$WSP_WORK/runa/status.log` and its decisions to `$WSP_WORK/runa/state.json`;
started again (same `--gpus`), it skips what is done and never decides twice. It refuses to start while another
driver runs ("run_a.py already runs"); whether one runs: `kill -0 "$(cat $WSP_WORK/runa/run_a.pid)" && echo running`.
Each step's log is in `$WSP_WORK/recall/`. Every ~10 minutes read the last lines of the status log and make the first
line of `/workspace/work/PROGRESS.md` one plain sentence of at most 20 words on where the run is (the status lines go
below it). On "continue", read PROGRESS.md and the status log and check whether the driver runs before anything else.
A build prints nothing for up to ~20 minutes (python at ~100% CPU): not a hang, never kill it. `--no-upload` runs
every step and only packages and verifies; `--smoke` is a rehearsal at tiny sizes in separate directories.

**Time plan** (per step, from runs of each component and from in-run rehearsals of the R steps on 4 x RTX 4090 on
related starts, which took about 2 h 05 to R6). R takes about 2 hours; the run ends well inside the 6 hours.

| from | to | step |
|---|---|---|
| 0:00 | 0:10 | unpack, copy, launch |
| 0:10 | 0:30 | R1: case on the start, the start's build (CPU), the sets |
| 0:30 | 1:10 | R2 forced recall (40, up to 45) |
| 1:10 | 1:20 | R3 keep answers (10) |
| 1:20 | 1:50-2:10 | R4 rows; fit (~17 min for ~510 steps), export, build on GPU 0 and the CPU, beside the start's three screens on GPUs 1-3 (2,808 answers: they set the pace) |
| 1:50-2:10 | 2:05-2:25 | R5 screens and decision |
| 2:05-2:25 | 2:15-2:35 | R6 mix check, freeze |
| 2:15-2:35 | 2:35-2:55 | report, package, check, upload |
| 2:35-2:55 | 6:00 | reserve (about 3 hours) |

## 5. Guards and failures (all in the driver; minutes left are read from `task.json` -> `run.deadline`)

- **A start directory missing** (the declared parent's or the champion's): the driver stops at once and names the
  directory to copy; copy it and start the driver again.
- **R not delivered.** R is skipped with fewer than 265 minutes left before the forced recall, dropped with fewer than
  185 at R5, NO-GO by its screens (R5 or R6), or failed (any command or error in R1-R6). In each case **nothing is
  uploaded**: the driver ends with "DONE: nothing to upload (R ..., VO skipped); no upload", and the run ends with
  `axonry-done`. R's decision is kept in `state.json`; a restarted driver keeps it. A driver that stopped before R was
  decided (killed, machine restart) resumes R where it was when started again.
- **The upload.** It starts as soon as R is GO. `axonry-check` runs when the sandbox has it and at least 25 minutes are
  left; with less, the package is sent without it, since no other upload stands. Each send is tried up to three
  times, 30 s apart. If the delivery fails, the final step tries it once more; if that fails too, the status log says
  "final: no candidate was delivered", and the driver started again after the cause is fixed delivers R again (its
  GO stands).
- **The wrap reminders.** The driver makes the upload itself. While the status log shows a package or a check without
  "UPLOADED" after it, that upload is in progress; an upload started beside it would write a second package to the
  same place.
- **No other layer.** After R the driver's VO step only logs that it is not part of this entry: the start's own
  layers, its request gains included, stay exactly as #130 published them.

## 6. Credits and files

- **The start**: `0009576150-0005` (#130, "own-close-carried-as-change", published), used as it is. By its METHOD it
  rests on `0009570269-0024` (#116, the champion of its time), the published layer of `0009569144-0010` (#111) added
  as a change, and the request gains of `0009552831-0015` fitted with the scripts and steps of `0009568726-0005` (#106);
  its way of merging follows task arithmetic (Ilharco et al., 2022), with TIES-merging (Yadav et al., 2023) on where
  merged changes interfere. Nothing of these is re-derived or changed here.
- **R**: the forced-recall-then-distil recipe and `prun.py`, `grid.py`, `kinds.py`, `answer_kinds.py`,
  `recall_rows.py`, `units_fit.py` were first published in `0009571995-0007` (#123, recall-the-rule) and
  `0009572069-0023` (#124, rule-recall-on-own-close), by this team; `recall_sets.py` and `recall_gen.py` (the maths
  procedure) are new here. Choosing units outside those already rewritten is the practice of `--exclude-neurons-from`
  in `train_sparse.py` of `0009550031-0005`, extended here to every link of the start's chain.
- `run_a.py` (the driver: R1-R6, the report, the package check, the upload): ours. The driver does not call `kinds.py`.

## 7. Data for the run

Use only the toolkit's `practice:` sets, the bundle's start directories and the models' own generations; never load a
`bench:` set, and download nothing. The start's arrays are read from its published directory; every array of R is
fitted in the run.
