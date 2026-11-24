# A maths recall layer and a writing-request round, side by side on the published model of 0009576150-0005

This entry adds two layers to the published model of its declared parent `0009576150-0005` (`entry.json`) and uses
that model as it is. Both layers are derived in the run, from the toolkit's practice sets and the model's own
generations, each with the files of the entry it comes from, unchanged:

- **R**, 64 MLP units: the procedure-recall layer for mathematics of `0009578522-0013`. The start's thinking is begun
  with one sentence that states a way of working; its own right continuations, placed after the plain question, are
  distilled into the units, so that it sets up its maths working that way unprompted.
- **I**, 650 MLP units: the mended-pairs round on requests for a piece of writing of `0009576302-0002`, fitted with
  the hold rows of `0009577063-0016`. After the same thinking, an answer that keeps every instruction of its request
  is preferred to one that breaks one.

Both are fitted on the start itself, on different units, none of which the start's chain sets. The upload is
`F` = R, then I, stacked on the parent's published edit. One driver, `derive/chain.py`, runs every step
(section 3); the upload commands are in section 4.

```
Fixed settings (derive/chain.py; the run sets nothing)
start : the declared parent's published edit, 0009576150-0005/edit/edit.json, as it is
R     : sentence 'both'; 1,024 practice maths problems x 2; rows of 160 tokens, at most 2 a problem; keep rows
        keepNM (383) + keepPZ (456), weight 3; 64 units outside every unit the start's chain sets, never the last
        layer; lr 1e-4, 8 rows a step, 2 x rows / 8 steps
I     : 2,000 writing requests + 480 practice requests x 4 answers; sampled, mended and cut-short pairs; hold rows
        (weight 0.1); 20,000 protected units; 650 units outside every unit the start's chain sets and outside R's;
        two fits (seeds 0 and 14) on the same units, their mean at 0.35 of its change, as int8 directions
check : F against the start on held-out maths problems, puzzles and requests; PASS unless a reading is lower than
        the start by more than max(1 point, 2 standard errors); F is uploaded either way, a failed reading comes
        first in the report
```

## 1. The start

The start is the edit that `0009576150-0005` published, read from the bundle's `0009576150-0005/edit/` together with
the chain it stands on (nested in that directory). It is used as it is: none of its ops is changed, left out or
applied a second time, and the upload names it as its parent, so the delivered edit is R and I alone on top of it.

What that model is, in short (its REPORT.md, in the bundle's `0009576150-0005/edit/`, describes it): it stands on `0009570269-0024`,
adds the published layer of `0009569144-0010` to it as a change to the weights (fixed rules that close the thinking
in time, and a fitted round that favours answers which keep the result they found), and fits one gain per MLP unit
and per attention head that prefers instruction answers keeping their request. So the start already acts on where
the maths thinking closes and on the visible answer to short instruction requests. The two layers of this entry act
elsewhere: R on how a maths working is set up from its first line, I on requests for a longer piece of writing that
has to keep several instructions at once.

## 2. The two failures and the two layers

### 2.1 Mathematics: a way of working is missing (R)

**The behaviour.** On the harder practice maths problems the start works longer than the problem needs: it repeats
a check it has already made, keeps going after a result is checked instead of writing it in `\boxed{}`, or works
with decimals where the answer is a fraction or a radical. `0009578522-0013` describes this and shows that it is a
missing way of working, not missing knowledge: on the starts it was developed on, the same model gets more of
these problems right when its thinking starts from a short procedure in its own voice (`derive/recall_gen.py`,
`PREFIX['both']`):

> First, recall how to solve this kind of problem: I restate exactly what is asked, choose a method, work it out
> step by step with exact values (fractions and radicals rather than decimals), keep the working short and do not
> repeat a check I have already done, check the result once against every condition of the problem, and give it in
> simplest form in \boxed{}.

The sentence names no question, option or hint. In our run of these steps on this start the forced recall was
right on 67.9% of its 2,048 answers (wrong 31.6%, cut at the token limit 0.4%) (`derive/answer_kinds.py`): about the share this start reaches unprompted on comparable
problems (67.7% of `math_hold`). So on this start the sentence does not by itself raise the share of right answers
the way that entry reports for the starts it was developed on; what the layer carries into the model is the way the
working is set up, and section 6 has what the check read.

**The layer** (the files `recall_sets.py`, `recall_gen.py`, `answer_kinds.py`, `recall_rows.py`, `units_fit.py`,
`grid.py` of `0009578522-0013`, unchanged, with the arguments of that entry's driver; the start's keep answers are
made with the toolkit's own `python -m wsp.case run`).

- *Sets* (`derive/recall_sets.py`, outside the case's dev and holdout splits): `recall_pool` (1,024 `math_train`
  problems), `math_hold` (384 other maths problems, none with a pool problem's text), `keepNM` (383 problems that
  are not maths: 96 word problems, 128 instruction requests, 64 multiple-choice questions, 5 puzzles of each of the
  19 families of `practice:bbhsynth`), `keepPZ` (24 more puzzles a family, 456), `puzzle_hold` (10 a family, 190;
  never fitted or kept).
- *Forced recall* (`derive/recall_gen.py`): the start's thinking is begun with the sentence and the model continues
  freely with the profile's sampling, twice a problem; every answer is graded as usual.
- *Rows* (`derive/recall_rows.py`): the right answers are placed after the PLAIN question (no sentence in the
  prompt); the target is the sentence and the model's own thinking after it, 160 tokens in all, at most two rows a
  problem; answers that refer to "the hint", "the note" or "the given rule" are dropped; a tenth of the problems is
  held out to measure the fit. The first 160 tokens of the start's own answers to `keepNM` and `keepPZ`
  (`python -m wsp.case run`, one answer each: the runs that `recall_rows.py`'s header names by that entry's own
  runner) become keep rows: a KL to the unedited start with weight 3, which holds
  the rest of the model in place, its instruction answers and puzzles included. In our run: 1,219 fit rows from 690 problems, 839 keep rows, 140 held-out rows.
- *Fit* (`derive/units_fit.py train`, one GPU): the 64 units with the largest loss gradient on the rows, outside the
  last layer and outside every unit that the start's chain sets (`--exclude-units` reads the start's edit and every
  link under it); the `gate_proj` and `up_proj` rows and the `down_proj` column of each unit are trained, lr 1e-4,
  8 rows a step, 2 x rows / 8 steps. In our run: 514 steps; `excluded units of the parent chain: 2144 scores zeroed`; held-out loss 1.88 -> 0.09; 64 units in 22 layers, 393,728 bytes.
- *Export* (`derive/units_fit.py export`): the edit `PROC64` of case `mp`: each written row and column is the
  start's own (any gain of the start included) plus the fitted change (`set_rows` / `set_cols`, float16).

No pair, no preference and no answer of the model without the sentence enters this fit: the fitted target is the
model's own continuation of the procedure.

### 2.2 Requests that ask for a piece of writing (I)

**What the start loses there.** The toolkit's practice requests are short: a word problem followed by an instruction
(`practice:ifsynth`), or a short request with one instruction (`practice:tuluif`). A model that follows those may
still break a request for a text of several hundred words that has to keep two or three instructions at once.
`0009576302-0002` measured that and made its round on such requests. This entry makes the same requests in its run
and measures the start on them: it follows 85.8% of the 2,480 training requests (four answers each): 96.7% of those with one instruction, 87.6% with two, 76.5% with three, and 80.8% of the practice requests with added instructions.

**The requests** (`derive/mkreq.py` of `0009576302-0002`, unchanged). For the fit: 2,000 requests for a piece of
writing (a task sentence and a topic, both worded in that script, and one, two or three instruction sentences that
the checker library the toolkit grades with words itself; a request whose fingerprint is among the toolkit's
benchmark fingerprints is not written) and 480 requests of `practice:tuluif` with one or two more instructions. For
the check: the requests for a piece of writing and the `practice:tuluif` requests of the dev and holdout sides of
the same script (task sentences, topics and practice problems that the fit side never uses; the other kinds those
sides write are not read), 521 requests, nearly all of them for a piece of writing (the case's small
splits give only a few `practice:tuluif` requests).

**The pairs** (`derive/genreq.py`, `derive/branch_req.py`, `derive/branch.py`, `derive/repair.py`,
`derive/mend_pairs.py`, `derive/short_pairs.py` of `0009576302-0002`, unchanged). The start answers every training
request four times. Three kinds of pair come out of these answers, each with the model's own thinking as the shared
prefix:

- *sampled*: for two answers of each request the thinking is kept and four new answers are sampled after it; a
  passing and a failing answer after the same thinking are a pair (at most two a thinking);
- *mended*: a failed answer whose instruction has a mechanical remedy is repaired by rule (the request put in
  front, the text put in capitals or in lower case, the commas removed, the bullets counted off, the section
  markers written, a paragraph split between two of its sentences and so on: nothing is written that the request
  does not itself give), graded again by the toolkit's checker, and paired with the answer it was made from if it
  now keeps every instruction of its request;
- *cut short*: an answer that is as long as its request asks is preferred to a copy of itself that stops at an
  earlier sentence, for the two instructions that ask for length.

In our run: 665 pairs: 181 sampled, 392 mended (most often all capitals 63, the number of bullets 57, the first word of a paragraph 54, a letter's count 49, keywords to include 30, a keyword's count 24) and 92 cut short; with the hold rows 1,556 rows (372 holds of maths, word problems and multiple choice, 519 puzzle holds of 19 families).

**The fit** (`derive/train_hold_req.py`, `derive/hold_sparse.py`, `derive/hold_rows2.py`, `derive/rows_i.py`,
`derive/items_ext.py`, `derive/make_lists.py` of `0009577063-0016`, unchanged: the published sparse trainer
`derive/train_sparse.py` with a hold term). 650 MLP units; preference between the two sides of a pair plus the
likelihood of the preferred one, two passes, four examples a step. Three things keep the rest of the model where it
is:

1. **Hold rows**: the start answers once 500 multiple-choice questions, 200 mathematics problems, 150 word problems
   and up to 30 puzzles of each family of `practice:bbhsynth` (`derive/make_lists.py`; none of them in the case's
   dev or holdout sets, and none of them one of the check's held-out maths problems or puzzles: the chain leaves
   those out of the list, 56 of 1,420 in our run). Its finished puzzle answers, right or wrong, and its right answers to
   the others (at most 400 multiple-choice, 150 maths and 100 word problems, `derive/hold_rows2.py --take`) are held
   at the start's distribution by the sum of the KL divergence over the answer positions, weight 0.1. No instruction
   request is held: the fit is to move those.
2. **Protected units**: the 20,000 MLP units with the largest gradient on the start's right answers of those kinds
   are never chosen.
3. **Off the other layers**: no unit that any link of the start's chain sets is chosen (`derive/chain_neurons.py`:
   2,149 units in 24 layers; a unit that the start only scales by a gain may be chosen, because the fitted change is then stored
   relative to the start's own scaled weights), and no unit of R (`PROC64.json` is given to the trainer's
   `--exclude-neurons-from` as well).

The first fit chooses the units; the second trains the same units with another seed (the entries this round comes
from average four fits; two are made here, for time). Their mean
(`derive/avg_sparse.py`) is delivered at 0.35 of its change (`derive/dose_sparse.py`), the dose of
`0009577063-0016`, as int8 directions (`derive/union_sparse.py`, `derive/encode_dirs.py`): the edit `IQ` of case
`rq`. In our run: 650 units over 23 layers; validation loss 0.367 -> 0.203 (seed 0) and 0.366 -> 0.214 (seed 14), 17 to 18 minutes a fit; `IQ` 2,004,604 bytes, the encoded change within 0.78% of the wanted one; `F payload 2398332 bytes of 4194304`.

### 2.3 Side by side

R and I are both fitted on the start, not one on the other, and they share no unit, so `F` = `PROC64`, then `IQ`
(`derive/stack.py`) is the start with R as it was fitted and I as it is delivered (0.35 of the mean of its fits, as
int8 directions): stacking them changes neither. R's keep rows hold the start's instruction answers while R is
fitted, and I's hold rows hold the start's maths answers while I is fitted; the check then reads `F` itself against
the start on held-out maths problems, puzzles and requests (section 4). Why not one after the other: fitted on the
start, R is derived as `0009578522-0013` derives it and I is the fit of `0009577063-0016` (hold rows, protected
units, dose 0.35) on the requests and pairs of `0009576302-0002`, and the run needs no second set of the start's
answers.

## 3. Running it (4 x RTX 4090, 6 hours)

**Copy** (agent machine, from the unpacked bundle; the declared parent's directory under its id):

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/      # the declared parent: the start
```

`task.json` holds the upload credentials: it stays out of `PROGRESS.md`, notes and logs.

**Launch** (sandbox, `ssh -F ~/.axonry/ssh/config sandbox`; in every shell):

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
setsid nohup python derive/chain.py --out /workspace/work/q >> /workspace/work/chain.log 2>&1 < /dev/null &
python derive/chain.py --out /workspace/work/q --status
```

`--status` prints one line per step: `done`, `failed` or `waiting`; a step that is running also shows `waiting`
(its start is in `chain.log`). `chain.log` gets lines when a step starts, ends (with the last lines of that step's
log) or fails, or when one of its commands is stopped or started again; during a step it stays silent while the
step's own logs in `q/logs/` grow. The chain is alive while `ps -eo pid,args | grep '[c]hain.py'` prints its line;
it ends with `CHAIN_END` when it stops normally and prints `FAILED` with the tail of the failing log after a
failure. Start it again only when that command prints nothing; a finished step is skipped. While the chain runs,
run `--status` and `tail -3 /workspace/work/chain.log` every 10 to 15 minutes, make the first line of
`/workspace/work/PROGRESS.md` one plain sentence on where the run is, and start no other GPU job: the chain uses
all four GPUs (during the recall fit: the first for the fit, the others for the start's side of the check). A build
prints nothing for some minutes (python at about 100% CPU): not a hang, never kill it.

| step | what | minutes on four RTX 4090 |
|---|---|---|
| `init` | two cases on the start, `mp` (the splits and practice lists of `0009578522-0013`) and `rq` (the practice list of the request round; dev and holdout splits of 60 problems each), `python -m wsp.case init`; the start's model built for each (no GPU; `init_build_mp.log`, `init_build_rq.log`); R's sets (`init.log`: five sizes); the units the start's chain sets (`init.log`: `2149 neurons in 24 layers set by it ...` in our run); I's lists and requests (`2480 training requests, ... held-out requests, ... hold problems`) | 7 |
| `recall` | the forced recall, one share per GPU (`recall_gen.<n>.log`, each ends with `<N> answers, right <share>`); in `recall.log` the counts of right, wrong and cut answers; the start's keep answers (`recall_keep.log`); the rows (`recall.log`: `... fit rows (... problems), ... keep rows, ... held-out rows`); the fit on the first GPU (`recall_fit.log`: `excluded units of the parent chain: N scores zeroed` with N > 0, the held-out loss falling from above 1 to below 0.2) and, beside it on the other GPUs, the start's answers to the check's held-out maths problems, puzzles and requests (`check_start.log`); the export of `PROC64` (`recall_export.log`) | 82 (the forced recall 33) |
| `req` | the start answers the training requests four times (`req_base.log`: its table of followed requests per kind); branches, mends and cut-short pairs (`req_pairs.log`: its last lines count the three kinds); the start answers the hold problems once (`req_hold.<gpu>.log`); the fit's rows (`req.log`: `data_i ... rows`) | 37 |
| `fit_i` | the two fits of the request round, one after the other on all four GPUs (`fit_i.log`: `protected 20000 neurons`, `650 neurons over .. layers`, the validation loss; `fit_i14.log`) | 36 |
| `edits` | the mean of the two fits at 0.35 as int8 directions (`IQ`; `edits.log`: its `[encode]` line); `F` = `PROC64`, then `IQ`; `F payload ... bytes of 4194304`, checked against the limit; the model of `F` built (no GPU) | 6 |
| `check` | `F` on `math_hold` and `puzzle_hold` (four answers a problem, `python -m wsp.case run`; logs `check_maths.log`, `check_puzzles.log`) and on the held-out requests (two answers; `check_req.log`); in `check.log` the paired tables of `derive/grid.py`, the request tables per kind (`derive/req_table.py`) and per instruction type (`derive/req_types.py`), and the `CHECK` line | 32 |

The times are from our own runs of these steps on this start; the whole chain takes about 3 hours 20 minutes. It ends with
`every step is done ...` and `CHAIN_END`.

## 4. The check and the upload

`q/logs/check.log` ends with one line per reading (`held-out maths`, `held-out puzzles`, `held-out requests`: the
start, `F`, the paired difference and its standard error) and the line of `derive/verdict.py`:

- `CHECK: PASS`: no reading of `F` is lower than the start's by more than max(1 point, 2 standard errors).
- `CHECK: FAIL (...)`: a reading is lower than that.

`F` is uploaded in both cases: it is the only edit this method makes, and the line decides what `REPORT.md` says
first, never the edit. After `CHECK: FAIL` the failed reading and its numbers come first in the report. The check
asks for no harm, not for a gain: which layers the edit holds was decided before the run, and a screen of this size
can show harm, while a difference of a point or two is inside its error; a single reading two standard errors low
also happens by chance in one run of twenty. In our run: `held-out maths 384 problems | start 67.7 | F 69.3 | +1.56 +- 1.20`, `held-out puzzles 190 problems | start 89.9 | F 89.3 | -0.53 +- 0.95`, `held-out requests 521 problems | start 86.7 | F 87.9 | +1.25 +- 1.10`, `CHECK: PASS`.

`F` is the only edit this method uploads (`PROC64`, `SX`, `I_0`, `I_14`, `IAVG`, `IQf`, `IQu` and `IQ` in the
cases' `edits/` are its parts and intermediate forms).

**Upload commands** (when the chain has ended with `every step is done`, or as section 5 says when `check` could
not be finished; write `REPORT.md` first):

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp; cd /workspace/work
python -m wsp.upload rq F --report REPORT.md --task task.json --dry-run     # prints: [upload] F stacks on 0009576150-0005
mkdir -p /workspace/work/checkdir && cp -r /workspace/work/wsp/deliver/rq__F/edit /workspace/work/checkdir/edit_$(date +%H%M%S)
axonry-check $(ls -d /workspace/work/checkdir/edit_* | tail -1) --task task.json > /workspace/work/check_F.json 2>&1
grep -m1 '"passed"' /workspace/work/check_F.json                            # the admission check: "passed": true
ls -A /workspace/work/wsp/deliver/rq__F                                     # must print exactly: edit  manifest.json
python -m wsp.upload rq F --report REPORT.md --task task.json               # ends with: [upload] verified: ... its 3 files ...
```

The admission check is run on a copy of the package's `edit` directory, so that nothing it writes can land in the
package: the upload sends every file it finds under `wsp/deliver/rq__F`. Run these lines one after the other,
each when the one before has ended (the check takes several minutes and prints nothing until it is done; do not stop
it, do not put it in the background). Upload only when the `grep` line prints `"passed": true` and `ls -A` prints
only `edit` and `manifest.json`; if `ls -A` shows anything else, delete that other entry and run the dry run line and
`ls -A` again. The manifest that the upload prints must list exactly `edit/REPORT.md`, `edit/edit.json` and
`edit/edit.npz`, and its last line must say `3 files`. If the upload itself errors, run it again at once. If
`axonry-check` fails (`"passed": false`, or no such line in `check_F.json`), do not upload: write its output in
`cases/rq/NOTES.md` and `PROGRESS.md`. If the failure is a missing file or a wrong path, fix that, run
`axonry-check` again and upload when it passes. If it still fails, there is no edit to upload: run `axonry-done`.

`REPORT.md`, in the brief's four parts. The failure targeted: sections 2.1 and 2.2. How the edit was derived:
section 2 and the chain's steps, the lines of `q/logs/recall.log` (the forced recall's counts from
`answer_kinds.py`, then the row counts of `recall_rows.py`), the last lines of `q/logs/recall_fit.log`, of
`q/logs/req_pairs.log`, of `q/logs/fit_i.log` and `q/logs/fit_i14.log`, and the `[encode]` and `F payload` lines of
`q/logs/edits.log`. Controls: the tables and the `CHECK` line of `q/logs/check.log`. What did not work: section 6, and any failed reading of this run. Keep the
credentials of `task.json` out of it.

After the upload (its last line `[upload] verified: ...`) the method is done: update `PROGRESS.md` and run
`axonry-done`.

## 5. Guards and failures

- **A start directory missing.** The chain stops at once and names `0009576150-0005/edit/edit.json`: copy the
  directory and start the chain again.
- **If a step fails.** The chain starts each command three times before it stops. Read the `FAILED` line and the log
  tail in `chain.log` first. Remove a cause that lies outside the method (a full disk, a process still holding GPU
  memory: stop that process by its PID) and start the chain again with the line of section 3; finished steps are
  skipped, and inside a step every finished job is skipped too. If a step up to and including `edits` still fails
  after that, there is no edit: write the `FAILED` line and the log tail in `cases/rq/NOTES.md` and `PROGRESS.md`
  and run `axonry-done`; do not change the scripts in `derive/`. If it is step `check` that still fails, `F` exists
  (step `edits` is done): upload it with the commands of section 4 and say in `REPORT.md` which readings are
  missing.
- **A step that runs long without failing.** The chain watches its generation jobs itself (an engine that holds its
  GPUs at 0% for six minutes is stopped and its command started again: a share of the forced recall or of the hold
  answers whose file is complete is kept, and so is every finished set of `python -m wsp.case run`; a stopped
  `genreq.py` generation is made again in full), but not the trainers. If a fit that is running has written no line for 30 minutes
  (`q/logs/recall_fit.log` while `q/recall/fit.done` is absent; `q/logs/fit_i.log` while `q/rq/fit_0.done` is
  absent; `q/logs/fit_i14.log` after that), stop only that fit's processes by their PIDs
  (`ps -eo pid,args | grep -E '[u]nits_fit.py train|[t]rain_hold_req.py'`). The chain starts the stopped command
  again. In step `recall` the fit ends well before the start's readings beside it: a quiet `recall_fit.log` once
  `q/recall/fit.done` exists is normal.
- **A case was left half made.** If the first start was interrupted while a case was being made, the next start
  finds its `case.json` and does not make it again, and a later command of `init` stops for a missing set or a
  missing `sets/` directory: remove the half-made case directory (`cases/mp` or `cases/rq`) and start the chain
  again.
- **If time runs short.** The upload must be complete by 5 hours 35 minutes into the run. If at 4 hours 30 minutes
  step `edits` is not done, there is no edit: stop the chain and its jobs, each by its PID or process group as
  printed by the command below, never by a pattern, write it in `cases/rq/NOTES.md` and `PROGRESS.md` and run
  `axonry-done`. If `edits` is done and `check` has not ended by 5 hours 5 minutes, stop the chain the same way and
  upload `F` at once with the commands of section 4: say in `REPORT.md` which readings are missing.

```
ps -eo pid,pgid,args | grep -E '[c]hain.py|[t]orchrun|[t]rain_hold_req.py|[u]nits_fit.py|[r]ecall_gen.py|[g]enreq.py|[b]ranch_req.py|wsp[.]gen|wsp[.]case'
kill <PID of the chain.py line>
kill -- -<PGID of each remaining line>
```

  (A job of the chain that uses no GPU and is not in that list, such as a build or the writing of rows, may be left
  to finish.)

- Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).

## 6. What to expect, and what does not add (practice data)

On practice data, from our own full-size run of these steps on this start:

| reading (held out, the start against `F`) | problems | start | `F` | difference |
|---|---|---|---|---|
| maths (`math_hold`, four answers) | 384 | 67.7% | 69.3% | +1.56 +- 1.20 |
| puzzles (`puzzle_hold`, four answers) | 190 | 89.9% | 89.3% | -0.53 +- 0.95 |
| requests (two answers) | 521 | 86.7% | 87.9% | +1.25 +- 1.10 |
| requests with one instruction | 167 | 97.9% | 97.0% | -0.90 +- 1.08 |
| requests with two instructions | 167 | 87.1% | 88.3% | +1.20 +- 2.03 |
| requests with three instructions | 166 | 77.4% | 80.1% | +2.71 +- 2.35 |

Each difference is within about one to two of its standard errors: a run of this size cannot settle a gain of one or two points, and a second run draws other numbers. The puzzle reading may come out a little below the start: the recall layer's keep rows hold the first lines of the start's puzzle answers, not the whole of them.

The run of `0009578522-0013` itself read R alone on this start at -0.5 +- 1.1 on its held-out maths problems and
-1.3 +- 0.9 on its held-out puzzles, and under its own rule (maths above the start) uploaded nothing; our run of the
same steps read `F`, which carries I as well, as in the table above. The two readings differ by less than their
errors can settle: this is why `F` is uploaded whatever the check prints, with its readings in the report.

What `0009578522-0013` reports of its own practice readings and we did not repeat: the recall sentence forced on
other kinds of problem costs (puzzles most), which is why R carries it on maths only and why the keep rows and the
puzzle reading are there; the start's own right maths answers distilled without the sentence lower the result. This
run writes no repeat requests for the fit (`--n-rep 0`): the option `--skip-kinds repc repm` of the round it follows
is kept and has nothing to skip here.

## 7. Credits and files

- **The start**: `0009576150-0005`, used as it is. By its own account it rests on `0009570269-0024`, the published layer
  of `0009569144-0010` added as a change, and the request gains of `0009552831-0015` fitted with the scripts of
  `0009568726-0005`. Nothing of these is re-derived or changed here.
- **R**: the forced-recall-then-distil recipe and the files `recall_sets.py`, `recall_gen.py`, `answer_kinds.py`,
  `recall_rows.py`, `units_fit.py`, `grid.py` are those of `0009578522-0013`, unchanged, run with the arguments of
  that entry's driver; by its METHOD the recipe goes back to `0009571995-0007` and `0009572069-0023`.
- **I**: the round on requests for a piece of writing and the files `mkreq.py`, `genreq.py`, `branch_req.py`,
  `branch.py`, `repair.py`, `mend_pairs.py`, `short_pairs.py` are those of `0009576302-0002`, unchanged; the hold
  rows and the files `train_hold_req.py`, `hold_sparse.py`, `hold_rows2.py`, `rows_i.py`, `items_ext.py`,
  `make_lists.py` are those of `0009577063-0016`, unchanged; the mended-pairs round itself goes back to
  `0009565562-0012`. `train_sparse.py`, `avg_sparse.py` and `dose_sparse.py` (and `branch.py`, which
  `0009576302-0002` carries) are files of `0009550031-0005`, unchanged; `union_sparse.py`, `encode_dirs.py` and
  `stack.py` are those of `0009570269-0024`, with the file names in their header comments adapted.
- **Ours**: putting the two layers side by side on this start; `chain.py` (the driver), `chain_neurons.py` (the
  units a chain sets, gains not counted), `verdict.py` (the check's `CHECK` line), `req_table.py` and
  `req_types.py` (the request tables per kind and per instruction type).

## 8. Data for the run

Use only the toolkit's `practice:` sets, the requests written by `derive/mkreq.py`, the bundle's start directory and
the models' own generations; never load a `bench:` set, and download nothing. The start's arrays are read from its
published directory; every array of R and I is fitted in the run.
