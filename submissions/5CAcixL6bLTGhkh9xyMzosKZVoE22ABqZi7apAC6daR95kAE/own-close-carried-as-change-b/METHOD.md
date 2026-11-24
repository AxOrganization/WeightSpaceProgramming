# The declared parent's fitted round carried onto the champion as a change at a practice-chosen dose, then the published request-gain recipe refitted in the run on the merged model (second run: request-gain seeds 5-8)

This entry names the published entry `0009569144-0010` ("keep-the-answer-own-close") as its parent in `entry.json`
and starts from the **champion of the moment** (task.json `starts`, role `"champion"`; written while
`0009570269-0024`, "procedure-workings", held that place). Its final upload is ONE edit on the champion, with two parts (an earlier upload holds the
carry alone, section 2):

| part | what | source |
|---|---|---|
| carry | the parent's own layer, **added as a change**: for every row and column its `set_rows` / `set_cols` ops write, the champion's weight plus what the parent's ops did to the weight of the model below it, its fitted preference round at the dose of section 1 (k = 0.5; not on the rows / columns the champion's set ops write, section 1), its fixed unit rules in full; its four `dir_ablate` ops unchanged | the declared parent's published ops, openly taken from the bundle; the way they are added, and the dose, are this entry's (`derive/carry.py`) |
| BR | answer-branch preference gains for instruction requests (one gain per MLP unit of layers 4-21, one per attention head), fitted in the run on the carried model; no published gain values are taken or copied: the gains are fitted in the run from this model's own answers | the published method of `0009552831-0015`, run with the published scripts and steps of `0009568726-0005` (`derive/sets.py`, `fit_gains.py`, `branch2.py`, `pref_gains2.py` unchanged) |

What this entry adds, plainly: two published layers edit the same neurons; this entry merges them by **adding the
change instead of overwriting the values**, at a dose fixed in this text from practice data (section 1), checks on practice
data that neither layer's behaviour drops beyond a floor, and then fits the request gains on the merged model. Everything is derived in the run from the bundle's published edits, the toolkit's
practice sets and the models' own answers. No arrays or data files are shipped.

The only fit in the run is the request gains of step 5 (one gain per MLP unit and per attention head). No gain value
of any published entry is used: the gains are fitted in the run from this model's own answers. No new neuron
layer is fitted and no math layer is fitted: the change to how answers close and keep their result is the parent's
fitted round, carried.

The carried ops are the published ops of the declared parent `0009569144-0010`, read from its edit.json in the bundle;
this entry derives no closing rule and no answer round of its own. New in this entry: (1) the declared parent's layer
added to another start as a change, not as values, at the dose of section 1; (2) the overlap of the two layers
measured from the published files (109 of 664 neurons); (3) the published request-gain recipe applied to the merged
model rather than to either start (the fit itself is that recipe, unchanged).

This is the author's second entry of this method on this start. The first, `own-close-carried-as-change` (same author,
committed before this one), is the same procedure, with the parent's round carried at k = 0.5 off the rows the
champion's set ops write and the request gains fitted with seeds 1, 2, 3, 4. One fit of the request gains is the mean of
four seeded fits, and the seeds are a choice the run cannot make for itself (the check of section 3 only accepts or
rejects what it is given), so this entry is a second run of the same method: it fits the request gains with the next
four seeds, 5, 6, 7, 8 (`BR_SEEDS` in `derive/run_e1.py`). Nothing else differs: the dose (k = 0.5, by the practice rules
of section 1), the carry, the lists, the check and its rules, the time guards and the uploads are the same in both.

## 1. The behaviour, and why the plain copy breaks it

The parent `0009569144-0010` stands on `0009554110-0010`. Its own layer (78 ops, published) holds four direction
removals on attention outputs and 74 `set_rows` / `set_cols` ops that write whole MLP neurons: two wrap-up units of
`0009554110-0010` silenced, an exit pair, break/commit pairs, and a preference round on 656 neurons over 22 layers
(answers that keep the answer they found and close their own thinking; its REPORT.md in the bundle,
`<parent sid>/edit/REPORT.md`, describes it). These ops
store **absolute values**: the rows as the parent computed them from `0009554110-0010`'s weights.

The champion also stands on `0009554110-0010`, with its own layers on top (program-written workings for fixed-form
puzzles, an instruction layer, a dead-end unit). Applied to the champion as they are, the parent's set ops would put
back `0009554110-0010`'s rows plus the parent's change on every neuron they touch, and so erase the champion's own
change wherever the two layers share a neuron. On the published files they share **109 of the 664 neurons the parent
writes (327 rows / columns)**: the plain copy would undo the champion's own edit on those neurons.

Adding the change keeps both: for a row `r` the parent writes,

    new[r] = champion[r] + k * (parent_ops(base_of_parent)[r] - base_of_parent[r])

where `base_of_parent` is the model the parent stacks on (`0009554110-0010`), `parent_ops(...)` is that weight after
the parent's own set ops (in float32, in the parent's order), and `k` is the dose (below). On a neuron the champion did
not touch, k = 1 gives exactly the parent's value; on a shared neuron, both changes stay. Measured against the champion
the upload stacks on, each written row changes by exactly k times the parent's change; the champion's value in it is
only the starting point the set op must restate, not a carried component.

This is task arithmetic (Ilharco et al., 2022, "Editing models with task arithmetic": a fine-tune's change is a vector
that can be added to another model) applied to two sparse layers; TIES-merging (Yadav et al., 2023) shows that what
goes wrong when such changes are merged is interference where they share parameters, which is why the shared neurons
are checked below on the behaviour of both layers rather than assumed to add.

**The dose.** The parent's layer has two kinds of ops (its REPORT): ops 0-11 are fixed unit rules found by its own
decisions (two silenced wrap-up units, four direction removals, the exit pair, the break/commit pairs), and ops 12-77
are its fitted preference round ("step H", delivered at dose 0.45 and, by its own REPORT, not read on dev before the
deadline). This entry carries ops 0-11 in full (the four `dir_ablate` alphas unchanged) and the round's change at
**k = 0.5**, but not on the rows / columns the champion's set ops write (below); on a row written by both kinds, the full change.

**Where the carried change is not added.** The champion's edit (every link of its chain as the bundle ships it; a flat
archive: its one edit) writes whole rows / columns of 72 tensors with `set_rows` / `set_cols`: 1,962 rows / columns,
which belong to 656 MLP units. This entry does not apply the round's change on any row / column that those set ops
write (`carry.py --keep-start-units`; the test is per row or column of the same tensor and the same axis, not per neuron: a `set_cols` of the champion does
not count against a `set_rows` of the round): the
champion's values stay there. On this champion that leaves out 378 rows / columns, which make up 126 whole carried
neurons (gate row, up row and down column); 538 neurons (1,610 rows / columns) remain, 3,316,884 bytes. Only the
champion's `set_rows` / `set_cols` count: rows it changes otherwise (its direction removals; 41 of the remaining carried
neurons differ in the champion from the parent's base that way) are not kept off. The 20 rows / columns of the
parent's ops 0-11 keep their full change (2 of them lie on the champion's rows) and its four direction removals are
carried unchanged. Said plainly: all 378 rows left out are rows written by `0009554110-0010`, the edit the parent stands
on and its round was fitted on, which the champion's archive contains; the champion's own top layer writes none of the
round's rows / columns with set ops (its direction removals do change rows / columns of 109 of the round's neurons, the
overlap above; on the 41 of them still carried, both changes are added). The parent's own round was fitted with the same
kind of care: none of its units is a unit of its own rules (`--exclude-neurons-from`, an option of `train_sparse.py`,
published work of the author of `0009550031-0005` that the parent uses unchanged; its METHOD); this entry applies that
practice to a different object, the carried round on the champion.

Why: in the author's own exploration on practice data (study 1 below) the round, carried as a change onto the
champion, cost the champion's skills more at the larger dose (puzzles and multiple choice against the champion: -1.44
and -2.94 points at k = 1, -0.86 and -0.08 at k = 0.5), and the round overlaps the rows the champion's set ops write.
Leaving the round off those rows tests whether that cost comes from them; the design is used only under the rule of
study 2.

**The rules.** The delivered round (k = 1) is the default: a smaller dose would be a choice of this entry, and as task
arithmetic chooses its coefficient on held-out data (Ilharco et al.), it is made here on practice data, by this rule:
k = 0.5 replaces k = 1 only if, on some practice set, it is better than k = 1 by more than two standard errors of the
paired difference and on no set worse by more than two; the same test then decides between 0.5 and 0.25. The carry that
leaves the round off those rows replaces the plain carry only if, at the dose this rule gives inside that
design, no practice set is below the champion by more than 2.5 standard errors and no set is worse than the plain carry
at the same dose by more than two. The request gains go on every unit, as the published recipe exports them; leaving
out the gains on the units the carry writes would replace that only under the dose test at the dose chosen.

Both studies use practice sets only: `math_train` 1,200 problems, `mmlupro` 821, the 608 test puzzles of `bbhsynth`,
`ifsynth` / `tuluif` 896 requests, `gsm8k_train` 500, all outside every training and keep list of step 5; 4 answers per
problem, 8 on `mmlupro`, 2 on `gsm8k_train`; the same sample indices and salt for every model, paired (with the check's
salt `e1chk`, so the first two answers per problem are the check's own draws). d +- se in points against the champion
(paired over problems).

Study 1, the plain carry (doses 1, 0.5 and 0.25 of the round, each alone, and at 0.5 with request gains fitted by step 5
on the k = 0.5 plain carry, exported with and without leaving out the gains on the carried units):

| model | `math_train` (1,200) | `mmlupro` (821) | puzzles (608) | requests (896) | `gsm8k_train` (500) |
|---|---|---|---|---|---|
| champion, accuracy % | 64.54 | 45.98 | 88.03 | 90.57 | 78.20 |
| carry, round at k = 1 | +3.33 +- 0.60 | -2.94 +- 0.74 | -1.44 +- 0.62 | -0.81 +- 0.61 | +1.00 +- 1.19 |
| carry, round at k = 0.5 | +3.56 +- 0.58 | -0.08 +- 0.61 | -0.86 +- 0.55 | +0.25 +- 0.55 | +0.60 +- 1.16 |
| carry, round at k = 0.25 | +3.25 +- 0.60 | +0.00 +- 0.61 | +0.45 +- 0.55 | +0.53 +- 0.50 | +0.60 +- 1.18 |
| k = 0.5 + gains, gains on every unit | +3.31 +- 0.62 | +0.17 +- 0.58 | -0.25 +- 0.62 | +1.62 +- 0.53 | +0.40 +- 1.12 |
| k = 0.5 + gains, none on the carried units | +3.25 +- 0.60 | +0.17 +- 0.60 | -0.41 +- 0.59 | +1.90 +- 0.53 | -1.10 +- 1.18 |
| contrast k = 0.5 minus k = 1 | +0.23 +- 0.55 | **+2.86 +- 0.62** | +0.58 +- 0.51 | +1.06 +- 0.57 | -0.40 +- 1.18 |
| contrast k = 0.25 minus k = 0.5 | -0.31 +- 0.55 | +0.08 +- 0.57 | **+1.32 +- 0.48** | +0.28 +- 0.50 | +0.00 +- 1.14 |
| contrast none on carried units minus every unit (at 0.5) | -0.06 +- 0.57 | +0.00 +- 0.55 | -0.16 +- 0.47 | +0.28 +- 0.44 | -1.50 +- 1.12 |

Study 2, the round left off the rows the champion's set ops write (doses 1, 0.5 and 0.25; the champion's and the plain carry's answers
are those of study 1):

| model | `math_train` (1,200) | `mmlupro` (821) | puzzles (608) | requests (896) | `gsm8k_train` (500) |
|---|---|---|---|---|---|
| round off the champion's set rows, k = 1 | +3.00 +- 0.59 | -1.37 +- 0.67 | -0.12 +- 0.56 | -0.08 +- 0.55 | +1.50 +- 1.19 |
| round off the champion's set rows, k = 0.5 | +3.29 +- 0.60 | +0.29 +- 0.60 | +0.04 +- 0.59 | +0.22 +- 0.54 | +1.30 +- 1.16 |
| round off the champion's set rows, k = 0.25 | +2.08 +- 0.59 | +0.46 +- 0.57 | +0.25 +- 0.52 | +0.25 +- 0.53 | +0.90 +- 1.23 |
| contrast k = 0.5 minus k = 1 | +0.29 +- 0.57 | **+1.66 +- 0.60** | +0.16 +- 0.48 | +0.31 +- 0.49 | -0.20 +- 1.07 |
| contrast k = 0.25 minus k = 0.5 | **-1.21 +- 0.57** | +0.17 +- 0.57 | +0.21 +- 0.46 | +0.03 +- 0.52 | -0.40 +- 1.06 |
| contrast with the plain carry at k = 0.5 (this design minus plain) | -0.27 +- 0.55 | +0.37 +- 0.58 | +0.90 +- 0.46 | -0.03 +- 0.52 | +0.70 +- 1.19 |

The rules' outcomes. Inside this design, k = 0.5 was better than k = 1 on `mmlupro` by +1.66 +- 0.60 (more than 2
standard errors of the paired difference) and worse on no set by more than 2; k = 0.25 was not better than k = 0.5 on
any set and worse on `math_train` by -1.21 +- 0.57: the dose is k = 0.5. At k = 0.5 this design is below the champion on
no set by more than 2.5 standard errors, and against the plain carry at the same dose it is worse on no set by more than
2: it replaces the plain carry. (For the plain carry alone, study 1 gave the dose 0.25 by the same dose rule.) The pair
with and without the gains on the carried units was read in study 1 at k = 0.5 on the plain carry, not on this carry, so
it cannot move the default: the published export (gains on every unit) stands. This carry with request gains was not
read in either study.

The studies' puzzles are the check puzzles of section 3; their other problems include the check problems of section 3
(256 `math_train`, 256 `mmlupro`, 256 requests); their `gsm8k_train` problems are not check problems. In the run the
dose and the design are fixed: the practice check (section 3) only accepts this carry or none, and no other dose or
design is tried.

> **Dose and seeds of this entry: k = 0.5 (the round's change, ops 12-77), kept off the rows the champion's set ops write, by the
> practice rules above; BR fit seeds 5, 6, 7, 8.** (`run_e1.py`: `CARRY_K`, `BR_SEEDS`; the driver passes
> `--keep-start-units`.)

Why the parent's layer is worth carrying on top of the champion: it targets answers that run to the token limit or
drop an answer they had. Why the
request gains are fitted after it: the gains act on the units of layers 4-21 of the model they are fitted on, so they
are fitted on the merged model, not on the champion alone.

## 2. How the edit is found (all in `derive/run_e1.py`)

1. **Start** (`start`). `derive/carry.py plan` reads task.json: the start is the champion; the parent is the declared
   parent, with the edit it stacks on nested in its directory (a former champion comes flat: one edit on the base with
   `FLAT.json`). A case `e1` is initialised on the champion; `derive/sets.py` writes the practice splits; `derive/chk.py
   ids` writes the check problems (section 3).
   **Practice rows used.** Every list the run builds by enumerating a practice set (the case splits, `sets.py`'s
   splits and the request list of step 5) uses only the first 1,024 rows, in file order, of `practice:ifsynth` and of
   `practice:tuluif`, whatever the files hold (`derive/pools.py`, constant `POOL_ROWS`; the driver runs the case init
   and `sets.py` through it). Further rows of those files, and practice sets the method does not name (such as
   `mmlucf` and `musr`), are not used. The other sets (`gsm8k_train`, `math_train`, `mmlupro`, `bbhsynth`) are used as
   the toolkit ships them.
2. **Carry** (`carry`, CPU, about 10 min with the model build). `derive/carry.py build --k 0.5 --k-ops 12-77 --keep-start-units` computes,
   for the tensors the parent's set ops write, the weights of the model below the parent, of the champion (each as
   `wsp.edits.build` gives them: every link rounded to the base dtype) and the parent's change, and writes `CARRY` into
   the case: the round's change (ops 12-77) at k = 0.5 except on the rows / columns the champion's edit writes with
   `set_rows` / `set_cols` (every link of its chain; left out: the champion's values stay), the rows written by ops 0-11 at the full change, one
   `set_rows` / `set_cols` op per tensor (values float16, row indices int16) and the parent's four `dir_ablate` ops
   (alpha unchanged). Payload rule: the own ops of the upload may hold 4,194,304 bytes; 202,752 bytes are kept free for
   step 5 (int16 indices of every unit and head column of layers 4-21, an upper bound of its export after
   `derive/compact.py`) and 4,096 bytes of margin. If the carry does not fit, the carried neurons with the smallest change
   (L2 norm of the dosed change over the neuron's gate row, up row and down column) are left out, smallest first, until
   it fits. The log line `[carry] ... trim: N neurons left out ... holding X% of the squared change` says what was dropped.
3. **Check** (`check`, section 3): the champion and `CARRY` on the check problems; the fixed rule picks `CARRY` or
   `NONE`.
4. **Upload 1** (`up1`): the chosen carry, alone, on the champion (`python -m wsp.upload e1 CARRY ...`).
5. **BR** (`br_init` ... `br_fits`): the steps published by `0009568726-0005` (reproduced in `run_e1.py`), with its settings (6 answers
   per instruction request, 4 fits, strength 1.0), on the model of step 3's choice (`wsp.case init br --parent
   cases/e1/edits/<choice>.json`; on the champion itself if the choice is `NONE`):
   - requests: every `ifsynth` id of the first 1,024 rows outside the dev, holdout and `tuluif` half-B lists of
     `sets.py`, plus `tuluif` half A;
     keep-set: 200 `gsm8k_train`, 100 `math_train`, 200 `mmlupro` training problems of `sets.py`;
   - answers by the carried model at 6,144 tokens (`derive/gen1.py`; per GPU shard one call for the requests and one
     for the keep problems): 6 per request, 1 per keep problem;
   - every request answer branched after `</think>` 4 times (`branch2.py --kinds if --if-all --k 4`: the same thinking,
     only the visible answer resampled);
   - `pref_gains2.py`: pairs of a passing and a failing answer after the same thinking, DPO (beta 0.1) plus NLL 0.2 on the
     answer tokens, keep term KL to the start on the keep answers (weight 30, first 4,096 tokens), one gain per MLP unit
     (`up_proj` row) and per attention head of layers 4-21, 2 epochs, one fit per seed of `BR_SEEDS` (5-8);
   - `fit_gains.py export --k 1.0 --levels 256`: the mean of the four fits' log gains, 256 levels per layer, as
     `scale_rows` on `mlp.up_proj` and `scale_cols` on `self_attn.o_proj` / `linear_attn.out_proj`.
6. **Compact** (`compact`): `derive/compact.py` writes `A05c`: the export's index arrays as int16; the ops and values
   are the export's (the request gains on every unit of layers 4-21, as the published recipe exports them).
7. **Check 2** (`check2`): the carry + BR model on the same check problems; the fixed rule decides (section 3).
8. **Upload 2** (`up2`): carry + BR as ONE edit on the champion (`python -m wsp.upload br A05c`: the chain champion ->
   carry -> BR, the local carry layer carried into the upload; the toolkit puts a `round` op between them, which builds
   the same model as the chain). It replaces upload 1 only if check 2 passes.

## 3. The practice check and its fixed rules

Check problems (`derive/chk.py ids`; practice data only, two answers each, same sample indices and salt `e1chk` for every
model, the profile's token limit; generated and graded by `wsp.gen`'s own function through `derive/gen1.py`, one call
per GPU on the interleaved split `wsp.gen` uses):

- `pz`: the 608 test puzzles of `practice:bbhsynth`, items 96-127 of each of the 19 kinds. The champion's workings were
  fitted on items 0-95 of each kind by its published split, so these test items are outside its fit. **This is where the
  champion's own behaviour must survive.**
- `math`: `holdout_math` of `sets.py` (256 `math_train` problems; outside every training list of step 5). **This is
  where the carried layer is meant to act.**
- `req`: the first 128 of `holdout_ifsynth` and the first 128 of `extra_tuluif_b` of `sets.py` (instruction requests that
  step 5 never trains on). **This is where the request gains are meant to act**, and where the champion's own instruction
  layer must survive the carry.
- `mc`: `holdout_mmlupro` of `sets.py` (256 multiple-choice questions, outside every training list of step 5).

`derive/chk.py cmp` prints, per set, the paired difference d (edit minus champion, points, mean over problems of the two
answers) with its standard error over problems, and the mean answer length. The rules are fixed before the run's check is read:

- **floors** (each edit against the champion, every set alike): d >= -2.5 se, where se is the paired standard error
  over problems of the same reading; one multiplier for every set (a set missing from the reading fails).
- **carry**: `CARRY` (the carry at the dose of section 1) if it passes every floor; else `NONE`: no carry, step 5
  runs on the champion, and nothing is uploaded in step 4. No other dose is read: the dose is fixed before the run.
- **final**: carry + BR replaces upload 1 if it passes every floor against the champion **and** its `req` is not below the
  upload-1 model's on the same requests (paired d >= 0). Otherwise upload 1 stands. If there was no upload 1 (`NONE`),
  BR on the champion is uploaded if it passes the floors; if it does not, the null edit (one `scale_rows` of factor 1.0:
  the champion unchanged) is uploaded so that the run ends with a complete, honest upload.

In the run nothing is tuned on the check problems: they decide, they are never fitted; in the run they are a
confirmation set, not a selection set: each edit is fixed before it is read, is read once in the run, and is never
repaired, re-dosed or re-sampled after the reading; the rules only accept or reject it. `holdout` is only the file name `sets.py` gives these lists;
its dev lists are not used in the run.

## 4. Setup (first 15-25 min)

From the unpacked bundle, put `submission/`, `task.json` and BOTH start directories (`<parent sid>/` and `<champion
sid>/`, as task.json `starts` lists them) in `/workspace/work`. **task.json holds upload credentials: keep it out of logs,
PROGRESS.md and REPORT.md.** Then, in a tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/work && python submission/derive/carry.py plan --task task.json
T_OFFSET_MIN=20 setsid nohup python submission/derive/run_e1.py all > /workspace/work/run_e1.out 2>&1 &
tail -n 5 /workspace/work/run_e1/logs/run.log
```

- `carry.py plan` prints the mode: `carry` (expected), `contained` or `parent` (section 6). It names the start (the
  champion in mode carry / contained, the parent in mode parent) and the parent's `edit.json`; if it names a missing directory, copy it from the bundle first.
- Set `T_OFFSET_MIN` to the minutes of the run used so far (20 above); it is saved on the first start, and the real
  deadline is used when the driver can read it.
- Start the driver once; if `status` shows it running, never start another (it refuses a second start while the
  first one's pid is alive).
- `run_e1.py` is resumable: a step with a `.done` marker in `run_e1/` is skipped; if the driver stopped on a failed
  step, read its log in `run_e1/logs/`, fix the cause and start the driver again (once).
- `RUN_E1_INCOMPLETE` in `run.log`: once `status` says the driver is not running, run `python
  submission/derive/run_e1.py up2` again at once (never `up1`, never `wsp.upload` by hand; it logs `RUN_E1_END` when
  the final upload landed); never run
  `axonry-done` before `RUN_E1_END`. The driver retries a failed upload 1 itself in its last step: never run `up1` by
  hand (after upload 2 was sent or started, `up1` refuses to send).
- Use only `practice:` data and the models' own generations. Never load a `bench:` set. Never `pkill -f` / `pgrep -f`
  a pattern (it matches your own shell).

**While it runs.** The driver does every step and every upload itself, for about 3.5 h. Check `python
submission/derive/run_e1.py status` every 10-15 min with short commands, never one long blocking wait, and keep the
last step done on PROGRESS.md's first line. Never run `wsp.upload` by hand and never stop the driver (except as in "A stalled step" below): an upload
started beside the driver's mixes two packages. Start no other GPU job while the driver runs (its runs take every
GPU). At the 30- and 10-minute reminders, check `status` and let the driver
finish; an upload starts its checks only with at least 36 minutes left (the null edit 28) and starts sending only
with 20 (the null edit 12; the unchecked re-send of upload 1 with 8). Run `axonry-done` only once `run.log` shows `RUN_E1_END`, never
after upload 1.

**A stalled step.** Progress is the step's log in `run_e1/logs/` growing (`run.log`, and the step's own logs such as
`gen_*`, `br_gen_*`, `br_branch_*`, `br_fit_*`). The driver stops any GPU call that runs 45 minutes (the longest
measured call took 21): it kills that call's process group, keeps the call if its output files exist (`gen1.py` writes its rows atomically; the BR scripts write theirs at the
very end of the call; a call whose outputs exist and stay unchanged for 60 s while its process lingers is stopped and
kept), and
otherwise starts it once more if the time left allows (else the step's fallback applies). If `run.log` itself shows no
new line for 60 minutes while `status` says the driver is running (an upload in progress writes only to
`logs/up1.log` / `logs/up2.log`: that counts as progress), kill the driver's process group (`kill -- -<pgid>`
of the pid in `run_e1/driver.pid`, as `ps -o pgid= -p <pid>` prints it) and start the driver again once: the lock lets
a restart through when the recorded pid is dead, it first stops the GPU calls the killed driver left running
(`run_e1/children.pgid`), and done steps are skipped.

## 5. Steps and time plan (4 x RTX 4090)

Measured in the author's rehearsal of `run_e1.py` on 4 x RTX 4090 (one GPU per job). CPU steps (model builds,
`axonry-check`, about 9 min each) were timed with 256 cores; on fewer cores allow more. `t` is the run's clock with 20
minutes of setup.

| step | what runs | measured | t at its end |
|---|---|---|---|
| setup | bundle, `carry.py plan`, driver started | (15-25) | 20 |
| `start` | case `e1` on the champion, practice splits, check ids, champion model build (CPU) | 9.6 min | 30 |
| `carry` | `carry.py build` (CPU, 40 s) and the carried model build (CPU) | 9.8 min | 40 |
| `check` | champion and carry, 1,376 check problems x2 each (2,752 answers per model) | 40.2 min (20.8 + 19.4) | 80 |
| `up1` | package, payload, `axonry-check` (builds the model on the CPU), upload | 9.0 min + upload | 90 |
| `br_init` | case `br` on the carry, training lists (1 min), its start model build (CPU) | about 10.5 min | 100 |
| `br_gens` | 1,152 requests x6 answers at 6,144 tokens, 500 keep answers | 18.5 min | 119 |
| `br_branch` | 24,700 branch rows (every request answer x4 after `</think>`) | 6.9 min | 126 |
| `br_fits` | 4 fits (one per GPU, 756 steps each) and the export | 18.6 min | 144 |
| `compact` | int16 indices (3 s) and the final model build (CPU) | 9.0 min | 153 |
| `check2` | the final model on the 2,752 check answers (the champion's are reused) | one model read, as the carry's: about 20 min | 173 |
| `up2` | package (1.2 min), payload (1.3 min), `axonry-check` (11-12 min), upload | about 14-15 min + upload | 188 |

The run ends at about t = 190 of 360 minutes. These times were measured before the BR answer calls were split into
one request call and one keep call per GPU shard and before the wait for each GPU's memory after every call (up to
about a minute per call); the dose does not change them.

Time guards in `run_e1.py`: step 5 starts only with at least 145 minutes left (else upload 1 stands); check 2 starts
only with at least 65 minutes left (else upload 1 stands). An upload starts its checks only with at least 36 minutes
left (the null edit 28) and starts sending only with 20 (the null edit 12); the re-send of upload 1 after a failed
upload 2 needs 8 minutes and is not checked again when its content equals the checked one. The driver reads the run's deadline from task.json (or
`/etc/axonry-deadline`, as `axonry-time` does) and uses the earlier of it and its own count. The judged edit is the
last complete upload before the deadline. Run `axonry-done` only after `RUN_E1_END`.

Every upload goes through `deliver` in `run_e1.py`: `wsp.upload --dry-run` (package), the payload of the packaged
`edit.json` (`wsp.edits.spec_size`, must be <= 4,194,304), `axonry-check` on the package, then `wsp.upload --task
task.json` (up to three attempts; an upload counts only when `wsp.upload` itself ends without error, which it does
only after reading the remote manifest back and checking every file's size), then `--status`, logged for
information only.
The package holds only `edit.json`, `edit.npz` and `REPORT.md`. If an upload failed, the step is not marked done; the
driver sends a missing upload 1 again in its last step. A half-finished upload makes the entry FAILED. If upload 2 fails before it starts
sending, upload 1 stands. If it fails after it started sending (the remote files may then be mixed: the manifest goes
last), the driver sends upload 1 again: the same package (its report is kept as `REPORT_up1.md`; `axonry-check` is not
repeated when the edit's content equals the one it passed). With no upload 1 it sends the null edit, and also
when upload 1 fails before sending (package, payload or `axonry-check`). If upload 1 fails while sending, the run ends
`RUN_E1_INCOMPLETE` and `run_e1.py up2` tries it again. Every upload starts only if the time left covers its checks (`CHECK_MIN`, 16 minutes) and its margin (the time
guards above).

## 6. Other starts, and failures

- **The champion of the moment already contains the parent** (its `FLAT.json` lineage or its nested chain holds
  `0009569144-0010`, or its REPORT.md says it carries it): mode `contained`. Nothing is carried (its layer is already in the start); the check reads the
  champion alone, step 5 runs on the champion, check 2 applies its floors against the champion.
- **No champion besides the parent** (the parent is the champion, or no champion is offered): mode `parent`. The start
  is the parent; nothing is carried; step 5 runs on it; same floors.
- **Another champion** that does not contain the parent: mode `carry`, the same steps on it: the parent's ops 0-11 at
  their full change, its round at k = 0.5 except on the rows / columns that champion's set ops write (every link of its
  chain; a champion with no set ops, or none on the round's rows, gets exactly the plain carry at k = 0.5, and the
  `[carry]` line says `0 rows/cols ... left out`), its four `dir_ablate` ops as they are; the check against that
  champion decides carry or none. In modes `contained` and `parent` there is no carry step and the option does not
  apply. If every row of the round lies on that champion's set rows, the rows of ops 0-11 at their full change (20
  rows / columns, 8 neurons on today's parent) and the four `dir_ablate` ops are still carried, so the carry is never
  empty. A known limit: on a champion that already holds the parent's ops 0-11 as values, the carry would apply them a
  second time; this entry was built and rehearsed for the champion named above and makes no claim for that case.
  If it holds no puzzle workings, the `pz` floor still guards whatever it does on puzzles. If that champion already
  carries the request gains of `0009568726-0005`, step 5 still fits them on the carried model; the floors of check 2
  decide.
- **The parent's layer holds ops the carry does not handle** (anything but `set_rows`, `set_cols`, `dir_ablate`, or one
  tensor written by both a set op and a `dir_ablate` op, or by both `set_rows` and `set_cols`), or the carry fails for another reason: `carry.py build` stops with the reason,
  `run_e1.py` logs it and goes on as for `NONE` (no carry; step 5 on the start).
- **A step fails or time runs out**: what is uploaded stands (upload 1 after the check; upload 2 after check 2).

## 7. What to expect (the author's rehearsal of these steps, on practice data only)

On RTX 4090 GPUs (3 for the check and 4 for step 5 in this rehearsal, 7 for check 2; the times of section 5 are from an
earlier 4-GPU rehearsal of the same steps). The carry and the check ran as the driver's steps in one work directory
(the champion's check answers reused from the earlier rehearsal); step 5, compact, check 2 and the uploads ran as the
driver's steps in a second one on the identical carry (same arrays). The carry, check and step-5 steps ran an earlier
copy of `carry.py` / `run_e1.py` whose differences from the files shipped are comments, the report's wording and how the
start's rows are read (every link of a nested chain, per tensor and axis); the shipped `carry.py` builds the identical carry from this bundle (checked),
and check 2 and the uploads ran the shipped files with `BR_SEEDS` = (1, 2, 3, 4), the seeds of the author's first entry
of this method: the step-5 fits, check 2 and upload 2 below are from those seeds. This entry's fits, with seeds 5, 6, 7,
8, are the same step with other seed numbers and were not part of this rehearsal.

- **Carry**: 538 neurons (1,610 rows / columns over 69 tensors, 4 `dir_ablate` ops); 378 rows / columns of the round
  left out on the rows the champion's set ops write; payload 3,316,884 bytes; no trim.
- **Check** (two answers per problem, salt `e1chk`, carry minus champion, points):

  | set | problems | champion % | carry % | d +- se | tokens champion -> carry |
  |---|---|---|---|---|---|
  | pz | 608 | 88.08 | 88.57 | +0.49 +- 0.66 | 2,258 -> 1,908 |
  | math | 256 | 67.58 | 64.84 | -2.73 +- 1.79 | 11,834 -> 12,381 |
  | req | 256 | 87.70 | 85.55 | -2.15 +- 1.55 | 4,051 -> 3,796 |
  | mc | 256 | 44.34 | 46.68 | +2.34 +- 2.28 | 8,914 -> 8,936 |

  Every d is above -2.5 se: the rule picks `CARRY`. With two answers per problem the check is a guard against a large
  loss; it does not resolve changes of one or two points (its standard errors are 0.7 to 2.3 points; study 2 read the
  same carry with +0.04 to +3.29 points on these sets, on far more problems).
- **BR** (the seeds 1-4 run): 1,152 requests (512 `tuluif` half A, 640 `ifsynth`), 6,912 request answers and 500 keep answers; 24,620
  branch rows; 404 preference pairs from 6,155 branch points (232 `tuluif`, 172 `ifsynth`) and 263 keep rows per fit;
  the four fits (19.9-20.1 min each) logged a running pair accuracy up to 0.97 (second-epoch means 0.87-0.88) and keep
  KL about 0.002; the export: 4,849 ops, unit gains 0.897-1.144, 403,240 bytes, 201,620 bytes after `compact`.
- **Check 2** (the same check problems, two answers each; FINAL = carry + gains):

  | set | FINAL minus champion | FINAL minus carry | tokens champion -> FINAL |
  |---|---|---|---|
  | pz | +0.33 +- 0.76 | -0.16 +- 0.65 | 2,258 -> 1,934 |
  | math | +2.34 +- 1.74 | +5.08 +- 1.87 | 11,834 -> 12,402 |
  | req | +2.93 +- 1.49 | +5.08 +- 1.63 | 4,051 -> 3,863 |
  | mc | +2.73 +- 2.14 | +0.39 +- 1.81 | 8,914 -> 9,362 |

  FINAL passes every floor against the champion and its requests are not below the carry's (+5.08): the rule picks
  `FINAL`.
- **Upload 2** (no upload, `UPLOAD=0`): carry + gains as one edit, 3,518,504 bytes (limit 4,194,304), `axonry-check`
  passed; the driver then logged `RUN_E1_END`, and its `up1` step, run after it, did not send ("upload 2 has already
  been sent"). The carry alone (upload 1's package, 3,316,884 bytes) was not checked separately in this rehearsal.

## 8. REPORT.md

`run_e1.py` writes it for each upload from its own logs: the mode and start, the `[carry]` lines (neurons, shared
neurons, payload, trim), the check table and the carry choice, BR's answer and pair counts and export line, the compact
line, the check-2 table and the final choice, each upload's payload, `axonry-check` and `--status` line, after a fixed
paragraph: the targeted failure, the carry formula with k and the op ranges, BR's source, seeds and strength, and the
controls (paired practice check, its floors). Write anything that failed, and when, in `run_e1/NOTES_AGENT.md`; the next
upload's REPORT.md includes it.

## 9. Credits and files

- **The parent**: `0009569144-0010` ("keep-the-answer-own-close", published): its own 78 ops are the layer carried here.
  Its ops are openly taken from the declared parent's published edit in the bundle; their values are recomputed as a
  change against `0009554110-0010`, the edit that parent stacks on. Carrying the declared parent's ops onto the champion
  follows `0009570269-0024`'s derive/take_unit.py, but as a change rather than as values.
- **The start**: the champion of the moment, `0009570269-0024` ("procedure-workings") when this was written; nothing of
  its own layers is taken into this entry's ops: it is the model the upload stacks on.
- **BR**: the answer-branch preference gains of `0009552831-0015`, as published with scripts and steps in
  `0009568726-0005` ("early-decision-credit-wide-br"). Of that entry only its BR scripts and steps are used; none of its published ops or values.
- **Where the round is not added**: leaving the carried round off the rows / columns the start's set ops write applies
  the parent's own practice (none of its round's units is a unit of its rules: `--exclude-neurons-from`, its METHOD) to
  the carried round; the rule of study 2 decided that it is used.
- In this second entry only the request-gain fit seeds differ from the author's first entry of this method on this
  start, `own-close-carried-as-change`.
- **The toolkit (`wsp`)**: cases, generation and grading, the practice sets, edit building, upload. The request-gain
  fit uses the published code of `0009568726-0005` carried in `derive/` (`fit_gains.py`, `branch2.py`,
  `pref_gains2.py`, `sets.py`); the toolkit offers `wsp.fit`, and this run keeps the code it was rehearsed with.
- Literature: Ilharco et al. 2022 (task arithmetic), Yadav et al. 2023 (TIES-merging).

| file | origin |
|---|---|
| `sets.py`, `fit_gains.py`, `branch2.py`, `pref_gains2.py` | `0009568726-0005`, byte-identical (its versions of the BR scripts of `0009552831-0015`; step numbers in their docstrings are that entry's) |
| `pools.py` | new: the practice rows the run enumerates (the first 1,024 rows of `ifsynth` and `tuluif`) |
| `gen1.py` | new: `wsp.gen`'s generation and grading (`wsp.gen._generate`) on one GPU, its rows written before the engine is torn down |
| `run_e1.py` | new; its steps `br_init`, `br_gens`, `br_branch`, `br_fits` and helpers `slots`, `split_ids`, `split_by_tl` are the BR steps `0009568726-0005` publishes (its step table), written in Python (the same commands and settings; shell scripts are not a package file type), with its own start replaced by this entry's carried model and the options of its other layers removed |
| `carry.py` | new: the start and the carry as a change, with its dose (`--k`, `--k-ops`) and `--keep-start-units` (the round kept off the rows / columns the start's set ops write) |
| `chk.py` | new: the check problems, the paired table and the fixed rules |
| `compact.py` | new: int16 index arrays (arrays read once per file; its option `--drop-units-of` is not used in this run) |
