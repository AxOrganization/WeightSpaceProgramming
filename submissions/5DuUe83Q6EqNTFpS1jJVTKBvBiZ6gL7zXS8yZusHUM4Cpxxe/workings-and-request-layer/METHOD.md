# The procedure written out, on a published start: compact workings for the toolkit's practice puzzles and a request layer beside them, both fitted with a keep term and delivered in int8 form

This entry stacks on the published entry `0009568831-0002` (`entry.json` `parent`), which stacks on
`0009554615-0007`, which stacks on `0009552724-0008` and, below it, further published edits. Their ops are inherited
as they are. This entry adds two layers on different neurons, both fitted on the start and added to it:

- **W, about 650 MLP units fitted on workings** (section 1; the number is the published trainer's default
  budget of 4,000,000 bytes of float16 rows): for puzzles of the toolkit's practice set
  `practice:bbhsynth` and of the program that wrote it, in the sixteen families for which this entry has a
  writer, a program writes the short thinking the puzzle needs, and the model learns to write it. While it learns, a
  keep term holds its next-token distribution to the starting model's on its own answers to everything else.
- **T, about 650 other MLP units fitted on answer-stage pairs of instruction requests written in the run**,
  with every finished thinking of the start branched (section 1.3), taken at dose 0.35. The fit is the published
  answer-stage preference of `0009550031-0005` (`derive/branch.py`, `derive/train_sparse.py`,
  `derive/avg_sparse.py`, `derive/dose_sparse.py`, `derive/rollouts.py`, unchanged), here with the keep term alone
  on the start's own answers to the other tasks. The parent already carries an answer-stage preference layer of this
  published kind for requests (its `REPORT.md`: BR); T is one more, fitted on the parent's own answers to the
  requests written in the run. T is in this entry to raise the start's instruction following beside the workings: a
  request layer of this kind raised 600 held-out practice requests by +6.75 ± 1.24 on another start, and on a third
  start the workings layer by itself read -2.3 ± 1.4 on held-out practice requests, within two standard errors of no
  change (section 1.3).

The two layers are added tensor by tensor (`derive/union_sparse.py`) and written in int8 form, one direction per
changed row or column (`derive/encode_dirs.py`: half the bytes of the float16 rows, within about 1% of the fitted
change); together they are the whole edit
`F` of this entry. Nothing else is fitted: what the start does on word problems, mathematics and multiple choice is
what the keep term holds in place.

Where it comes from, plainly: the start is the published work of its authors, used unchanged. The trainer and the
sampler are the published scripts of `0009550031-0005` (`derive/train_sparse.py`, `derive/rollouts.py`, unchanged).
The answer-stage preference fit, its averaging and its dose are the published scripts of the same entry
(`derive/branch.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`, unchanged). The chain starts `rollouts.py` and
`branch.py` through `derive/x_rollouts.py` and `derive/x_branch.py`, and the trainer through `derive/train_keep.py`;
these import `derive/xitems.py`, which adds the run's requests and puzzles (`REQ_ITEMS`) to the toolkit's item index
and changes nothing else.
Ours: the workings and how they are fitted (full strength, keep term on keep rows and hold rows,
the trainer's neuron protection used for the keep rows), the requests written in the run, fitting the answer-stage
pairs with the keep term alone on the other tasks, and the int8 form of the two layers. Branching every finished
thinking instead of one per request is a setting of the published `branch.py` (`--per-problem 4`), not a new script.
Holding a model to its start with a KL term on the start's own answers is a known device and not ours. What is
specific here is what is learned beside it in W (workings that a program writes from the puzzle text and checks, not
sampled answers that earned credit; T learns from sampled answers, the published way) and what the term is applied
to (the start's right answers to other tasks, and its other finished answers as hold rows that get the KL term alone;
for T, the keep and hold rows of the other tasks, with the KL term alone).

**Where the puzzles come from.** From the toolkit: its practice set `practice:bbhsynth` and, for more puzzles of
the same families than the set holds, the toolkit's own program that wrote the set (`wsp.puzzles`), called in the
run with our seed; a drawn puzzle is dropped when its fingerprint is among the toolkit's benchmark fingerprints or
in the practice set. `derive/practice_puzzles.py` takes the puzzles from those two places. The writers of the
workings (`derive/workings.py`, `derive/wk/`) parse the puzzle text with patterns that match the toolkit's sentence
forms and read its scenes and word lists (adjectives, nouns, counting categories, colors, things, names, relation
and rank phrases) from `wsp.puzzles` at run time. What they carry themselves is fixed wording of the toolkit's
puzzles and a few names the toolkit uses: the shape names, the names of the eight adjective classes and the four
compass directions (the toolkit gives the last two in comments), and the header and column words of the penguin
table.

## 1. What the starting model loses on puzzles, and what the edit does about it

(The numbers of this entry are from our own full-size run of this procedure on this start, on practice data, unless
a line says otherwise.)

### 1.1 Puzzles of the families with a writer: the model misreads the task or loses track

The toolkit's practice set `practice:bbhsynth` holds puzzles of nineteen families (time windows, adjective order, shapes
drawn by a path, counting, swaps, orderings, walks, truth chains, expressions, word sorting, dates, tables and so on),
128 of each. `derive/practice_puzzles.py` takes a test part (40 a family: the set's items 96 to 127 and 8 more from the
toolkit's program) and a fit part (600 a family: the set's items 0 to 95 and 504 more from the toolkit's program); the
two share no puzzle. (The set's last quarter is the test part; 600 is twice the 300 workings a family, because about
half of the seven-thing orderings get no working; the seeds are fixed values, set once.) The starting model on the
test part (760 puzzles, two answers each, full token limit):
**72.9% right, 9,200 tokens per answer.**

| family | right % | what goes wrong (read from its wrong answers) |
|---|---|---|
| a time window that is left free | 28.8 | it lists the times the person was seen elsewhere, then picks one of them |
| the shape a path draws | 51.2 | a path with an arc is misread: the arc's radii are taken as a point or the arc as a straight side, so a sector becomes a triangle and an ellipse a line or a triangle; among straight paths a rectangle is called a kite, and the point that closes a path is counted as one more corner (a heptagon becomes an octagon) |
| an order of seven things | 52.5 | a comparison or a rank given from the far end is put at the wrong end, after long thinking case by case |
| an order of five things | 53.8 | a comparison is put the wrong way round ("finished below" read as placed higher, "older than" put later on a list that starts with the oldest), mostly in the golf and car puzzles; a rank from the far end ("second-to-last") also slips |
| counting things of a category | 65.0 | it counts kinds instead of quantities (two heads of one vegetable as one), or keeps things of another category in the total |
| adjective order | 66.2 | no rule: it never uses the opinion class, makes up its own order (colour before quality) or reads a purpose word as a verb, and in most wrong answers keeps the first option |
| a walk that may return | 73.8 | the heading is not tracked: turns are skipped or taken as moves and the step counts are added into one distance, so a walk with turns that does return is answered No |

(The families with a working that the start answers right less than three times in four.) These are misreadings of
the task or slips of bookkeeping, of the same kind every time, and after a right working the model often thinks on
for thousands of tokens and talks itself out of it.

**The working written out and learned.** For a puzzle, `derive/workings.py` writes the facts in a fixed layout, the
one rule of the family, the conclusion, the end of the thinking, one sentence and the answer: a few hundred tokens.
It is written from the puzzle text alone and checked against the puzzle's known answer; a puzzle whose working does
not reproduce it is left out (orderings that the working cannot fill one thing at a time, because at some step no
thing has exactly one place left: about a quarter of the five-thing ones and about half of the seven-thing ones, so
that the seven-thing family gets 299 workings and
`rq/logs/fit_w.log`, the workings writer's log with a lower-case w, carries a NOTE line about it, which is expected;
a few shapes with a nearly vertical side). The workings are fitted as likelihood rows and delivered at full
strength: the workings layer is never dosed.

**Which families get a working.** The sixteen for which `derive/` has a writer: time windows, adjective order, shapes,
counting, swaps (3, 5, 7), orderings (3, 5, 7), walks, truth chains, boolean and arithmetic expressions, the penguin
table and the coloured objects. Dates and formal arguments get none: this entry has no writer for them. Word sorting
gets none either: a number-key working that we wrote for it made the model answer worse on practice puzzles, not
better (section 5), so no writer for it is part of this entry. The three keep the model's own thinking and are only
held in place (below). In the check below, every family with a working read at least as high as the start.

**The short style must not leak into other tasks.** Two things hold everything else in place while the workings
are learned:

1. **Keep rows and hold rows.** The start answers once: 450 requests written in the run (section 1.2), 300 word
   problems, 300 math problems, 450 multiple-choice questions (half as many again as the word problems, because the
   start answers only 34% of them right against 76% of the word problems, and only right answers become keep rows) and
   200 puzzles of each family without a working. Its right answers that close their thinking within 14,000 tokens
   (12,000 for a puzzle) are keep rows (rows of type `anchor`; in our run 349 requests, 228 word problems, 152 math,
   151 multiple choice and 303 puzzle answers): `derive/train_keep.py` (the published trainer with one more term) adds
   on each of them the KL divergence of the start's next-token distribution to the edited model's, per token, weight
   100 (a plain setting; a fit of ours with weight 400 on another start gave the same gain on its practice puzzles).
   Its other finished answers are hold rows (in our run 90 requests, 47 word problems, 15 math, 154 multiple choice
   and 106 puzzle answers): they get that KL term alone (`--hold-ids`), so nothing wrong is made more likely and what
   the start writes on these prompts stays as it is, right or not. Of the 200 puzzles of a family without a working,
   at most 150 finished answers are used (keep and hold rows together). The other counts, the cap of 150 and the
   length limits are plain settings sized to a keep step of about 14 minutes on four GPUs; none was tuned on a
   reading. In a paired comparison in our own runs on another start (the same fit with and without hold rows, read on
   190 answers of that start to practice problems no fit saw), the drift from the start, measured as the KL divergence
   summed over an answer, was 11.5 nats per answer with hold rows against 13.2 without, at about the same gain on the
   puzzles (+19.8 points with hold rows, +20.6 without, on the families with a working).
2. **Protected neurons.** The 20,000 neurons with the largest gradient on 256 of the keep rows of other tasks are
   never chosen (`--protect`, `--protect-n`, `--protect-examples`): plain settings, about a quarter of the model's
   86,000 MLP units, not tuned on a reading.

**What the fit does on this start** (the test part, two answers a puzzle, and 300 word problems with two answers
each, against the start):

| | the start | `F` (the start + W + T) |
|---|---|---|
| families with a working (640 puzzles) | 75.2% right, 9,100 tokens an answer | **93.2% right (+18.0 ± 1.6)**, 380 tokens |
| families without a working (120 puzzles) | 60.4% right | 60.4% (+0.0 ± 3.2) |
| all test puzzles | 72.9% right | 88.0% (+15.1 ± 1.4) |
| word problems (300) | 77.0% right | 77.0% (+0.0 ± 1.5) |

**About 300 workings a family, not the practice set alone.** Two fits of this kind in our own runs on another start,
each with the same second layer beside it (a model stacked on `0009554110-0010`; the same toolkit puzzles, the same
test part, practice data; writers for
fourteen of the families then, keep rows without hold rows), both read against `0009554110-0010`: with about 300
workings a family (97 for counting, whose writer then covered fewer puzzles) the test part read 14.7 ± 1.4 points
above it;
with the practice set alone (1,162 workings, 17 to 96 a family) 3.1 ± 1.5 above it, and boolean expressions, shapes,
orderings of five and seven and walks 7.5 to 18.8 points below it: the model took the short layout but not the
procedure (its answers to those five families were 260 to 1,230 tokens long and less often right). This is why the
run draws further puzzles from the toolkit's program beside the practice set.

### 1.2 The requests of the keep rows are written in the run

The model follows most of the practice instruction requests (read on an earlier start), so they show little of
what an edit costs. `derive/reqgen.py` writes 3,000 requests the way the toolkit builds `practice:ifsynth` (a
request plus constraints whose arguments the checker library draws itself, graded by the toolkit's grader), with
these differences: the request is a writing or explaining task (40 task sentences x 100 everyday topics, all in the
file; the kind of request that the shipped set `tuluif` holds beside its word problems), not a word problem; every
instruction type that the shipped practice sets contain is used (`ifsynth` and
`tuluif` together: 23 types), equally often, and no other; a constraint of a type that `ifsynth` has is worded as
that set words it or in one of a few other ways, one of the three types that only `tuluif` has in one of the other
ways; and one, two or three constraints stand together in equal shares. 2,400 of them are the training requests:
the pool the 450 keep requests of the workings fit are drawn from, and the requests whose answers give the pairs of
the request layer (section 1.3); 300 are the dev requests read by `read_a` (no fit sees them); 300 more are read
only by the confirmation steps.

**One reading is one draw.** Generation is seeded: a model run twice on a set returns the same answers. The null
edit (`derive/null_edit.py`: one MLP unit of about 86,000 scaled by 1.05) gives a second draw of the starting model;
`derive/cmp.py` pairs an edit with the mean of the two draws. Count answers before believing a gain or a loss of two
points.

### 1.3 Requests: a request layer beside the workings

(Practice data: the 300 dev requests of section 1.2, two answers each; an edit is paired with the mean of the two
draws of the start, as `derive/cmp.py` prints it.)

**Why a request layer beside the workings.** In a full-size run of the workings part of this procedure on another
published start (with a seventeenth family, section 5), that start with the workings layer alone read -2.3 ± 1.4
(83.2% against 85.5%) on that run's 300 dev requests. The keep term holds the start's distribution on its own answers
to 450 training requests, and still the reading on requests no fit saw came out a little lower, though within two
standard errors (the two draws of that start lay 0.3 ± 1.4 apart). A request layer is fitted beside the workings so
that instruction following does not pay for the puzzles, and gains.

**The request layer.** The start answers each of the 2,400 training requests twice (limit 6,144 tokens) and twice
more where the first two agree in outcome (limit 12,288): in our run 8,636 answers. After every finished
thinking, four new answers are sampled and graded by the toolkit's grader: of the 7,583 thinkings, 6.8% have both
outcomes among their four answers, which gives 670 pairs. So after one and the same
thinking the start sometimes writes an answer that keeps every constraint and sometimes one that drops one: the
thinking was enough, and the constraint is lost in the answer stage. A pair is two answers after the same thinking,
one that follows the constraints and one that does not; the pairs share the thinking, so the fit sees only that
stage. The published recipe branches one thinking per request; branching every one gave more than three times the
pairs on another start in our own runs (632 against 187) and read +6.8 ± 1.2 against +4.3 ± 1.1 on 600 held-out
practice requests there (four fits, dose 0.35).

`derive/train_keep.py --keep-only` fits the published answer-stage preference on these pairs: the neurons are chosen
once in a pass without training, outside W's neurons and outside the 20,000 neurons with the largest gradient on 256
of this fit's own keep rows (below; no request answer is among them, since this layer is to move the requests); two
fits of that one neuron set are averaged (`derive/avg_sparse.py`) and 0.35 of the change is delivered
(`derive/dose_sparse.py`). The dose is the one the published recipe ships first; a fit of this kind in our own runs
on another start supports it (two fits, 1,156 pairs from eight new answers after every thinking, 370 keep rows and
3,000 protected neurons then): on 600 held-out practice
requests it read +6.75 ± 1.24 at dose 0.35 and -0.33 ± 1.54 at dose 0.7, where 9.4% of the answers ended without an
answer part. Beside the pairs, every keep and hold row of section 1.1 that is not a request (the start's finished
answers to word problems, math, multiple choice and puzzles of the families without a working, right or not, within
the same length limits and the cap of 150 a family; 1,156 rows in our run) gets the KL term alone, weight
100: the request layer is not to move anything there. The other settings are plain ones, none tuned on a reading:
the caps 6,144 and 12,288 (a short first round, a longer one only where the first two answers agree); the published
recipe's four new answers after a thinking, two passes, global batch of four and at most two pairs per thinking; two
fits (its first two seeds, 0 and 14) instead of its four, as in the dose reading above; a longest sequence of 16,384
tokens, so that no pair and no keep row is left out; and the trainer's default budget of 4,000,000 bytes, as for W.

**The two layers together on this start.** `F` = the start + W + T read +6.3 ± 1.7 (85.8% against 79.5%; the two draws
of the start lie 3.0 ± 1.5 apart) on the dev requests. The check
of section 1.1 (test puzzles and word problems) is read on `F` with both layers in it: the table above is that
reading.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This
method is written for its parent, `0009568831-0002`: the workings are fitted on the parent's model, the edit names
it, and the numbers of this entry were read on it unless a line says otherwise. A champion offered as a second start
is not needed for it. On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009568831-0002 sandbox:/workspace/work/     # the start: role "parent" in task.json
```

**Names.** *The start* is the model of `0009568831-0002`: the pinned base with the ops of `0009568831-0002` and of
every edit it stacks on (nested in its directory: `0009554615-0007`, `0009552724-0008` and the edits below them).
After `init --parent` the case calls this model `base` (`case.model_path('rq', 'base')`,
`rq/runs/reqdev.base.jsonl`): in this method `base` never means the pinned base model without those ops (the field
`"base"` of an `edit.json` still names the pinned base). Two directories are called `rq`: `/workspace/work/rq` holds
the chain's outputs (logs, runs, requests) and `/workspace/work/cases/rq` holds the case (`case.json`, `edits/`,
`NOTES.md`); the toolkit's scratch for the case is under `/workspace/work/wsp` (`models/rq__base`, `models/rq__F`,
the upload package `deliver/rq__F/edit`).

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
python -m wsp.case init rq --profile qwen35-0.8b-think --parent 0009568831-0002/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python derive/reqgen.py --out-dir /workspace/work/rq/req                        # 3,000 requests: train / dev / hold
python derive/null_edit.py --case rq                                            # the edit NULL: a second draw
python -c "from wsp import case; [case.model_path('rq', m) for m in ('base', 'NULL')]"   # build ONCE, before any GPU job
```

- `init` must find `0009568831-0002/edit/edit.json` and, nested in that directory, the edits it stacks on
  (`0009568831-0002/edit/0009554615-0007/edit/edit.json`, inside that one `0009552724-0008/edit/`, and so on down
  the chain). It prints `{'dev': 480, 'holdout': 800}`.
- Run this block once. If `init` says `.../cases/rq exists`, do not delete the case: run only the lines below it
  whose result is missing (`rq/req/req.jsonl` for `reqgen.py`, `cases/rq/edits/NULL.json` for `null_edit.py`), then
  the build line again (it builds only what is missing), and go on with section 3.
- `reqgen.py` prints `3000 requests`, `23 instruction types`, `dropped for a benchmark fingerprint: 0` and
  `train 2400, dev 300, hold 300` (train is the pool of the keep requests).
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/reqgen.py`, the workings written by
  `derive/workings.py` and the models' own generations. The puzzles are the toolkit's (`practice:bbhsynth` and its
  program `wsp.puzzles`, read by `derive/practice_puzzles.py`): do not write or add any other puzzle. Never load
  `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

One command runs every GPU step in order; a finished step is skipped when the chain is started again:

```
setsid nohup python derive/chain.py --case rq --out /workspace/work/rq >> /workspace/work/chain.log 2>&1 < /dev/null &
python derive/chain.py --case rq --out /workspace/work/rq --status
```

`--status` prints one line per step: `done`, `failed`, `skipped` (a request-layer step after a `--w-only` start) or
`waiting`; a step that is running also shows `waiting` (its start is in `chain.log`). `chain.log` gets lines only when
a step starts, ends (after `keep`, `fit_w` and `check` also the last
lines of that step's log) or fails, or when one of its commands is stopped or started again: during a step it stays
silent (about 50 minutes in `fit_w`) while the step's own logs in
`rq/logs/` grow (the trainer's log is `rq/logs/fit_W.log`, capital W). The chain is alive while
`ps -eo pid,args | grep '[c]hain.py'` prints its line; it ends with `CHAIN_END` when it stops normally and prints
`FAILED` with the tail of the failing log after a failure. Start it again only when that command prints nothing.
While the chain runs, run `--status` and `tail -3 /workspace/work/chain.log` every 10 to 15 minutes, add a line to
`PROGRESS.md` when a step ends, and start no other GPU job: the chain's steps use all four GPUs.

| step | what | minutes on four RTX 4090 |
|---|---|---|
| `prep` | practice puzzles taken from the toolkit: a test part (40 a family) and a fit part (600 a family) that share no puzzle; it prints how many drawn puzzles it dropped, and why | 1 |
| `keep` | the start answers once what it has to keep (2,100 prompts); its right answers become the keep rows, its other finished answers the hold rows | 14 |
| `fit_w` | GPUs 0,1: the workings fitted on the start with the keep term; GPUs 2,3: the start on the test puzzles and word problems | 50 |
| `roll_a` | the start answers every training request twice (limit 6,144 tokens) | 10 |
| `roll_b` | two more answers where the first two agree in outcome (limit 12,288) | 10 |
| `branch` | four new answers after every finished thinking, graded: the same-thinking pairs | 7 |
| `fits_t` | the request layer: its neurons chosen once outside W, then two preference fits side by side with the keep term alone on the other tasks | 39 |
| `edits_w` | the two fits averaged and taken at dose 0.35 (T); W and T added and written in int8 form; `F` and the null edit's model built | 3 |
| `check` | `F` on the same test puzzles and word problems; the verdict line | 10 |
| `read_a` | two draws of the start and `F` on the 300 dev requests | about 10 (in our run the start's two draws were read while `fit_w` ran and `F` after `check`, 9 to 10 minutes each) |

The times are from our own run of the chain on four RTX 4090; the whole chain takes about 2 hours and 35 minutes.
It stops by itself after `read_a`: `chain.log` then ends with `the chain stops after read_a ...` and `CHAIN_END`, and
the steps `read_b`, `freeze`, `read_c` and `read_d` stay `waiting` (they run only with `--final`, which this method
does not use).

**As soon as `--status` shows `edits_w  done` (in `chain.log`: `edits_w: done`), upload `F` once** (commands below;
the upload uses no GPU, so `check` and `read_a` go on meanwhile). `F` is the only edit this method uploads (`W`,
`TSEL`, `T0`, `T14`, `TAVG`, `T`, `WU` and `WD` in `cases/rq/edits/` are its parts and intermediate forms, `NULL` is
the second draw of the start). For this first upload a short `REPORT.md` is enough: what the edit is (section 1 in a
few sentences: the two layers W and T, T at dose 0.35; or "the workings layer alone" after a `--w-only` start) and
the last `[encode]` line of `rq/logs/edits_w.log`.

**After `read_a`: report three readings and upload `F` again with the full report.**

1. The last five lines of `rq/logs/check.log`: `kinds with a working`, `kinds without a working`, `all puzzles`,
   `word problems` and the verdict line. `CHECK: PASS` means the families with a working taken together at least +8
   points against the start on the test puzzles, the families without one taken together not below -8, and the word
   problems not below -4 (we measured +18.0, +0.0 and +0.0); a single family may read lower. These are fixed gates:
   they decide
   what `REPORT.md` says first, never the edit.
2. The last `[encode] ... WD.json` line of `rq/logs/edits_w.log` (`WD` = the two layers in int8 form, the ops of `F`):
   a payload below the limit of 4,194,304 bytes and an encoded change within 2% of the wanted one (we measured
   4,006,200 bytes and 0.79%).
3. `F` against the start on the dev requests:
   `python derive/cmp.py rq/runs/reqdev.base.jsonl rq/runs/reqdev.NULL.jsonl --edit F=rq/runs/reqdev.F.jsonl` prints
   a header and two lines: `draw 2 - 1` (the null edit's draw against the start's: the noise between two draws) and
   `F` (`F` paired with the mean of the two draws). The reading is the `gain` and `se` of the `F` line (we measured
   +6.3 ± 1.7: 79.5% for the start, 85.8% for `F`; the two draws of the start lie 3.0 ± 1.5 apart).

If the verdict line is `CHECK: FAIL`, or the `gain` of the `F` line of `cmp.py` is below -4.0, or the encoded change
differs by more than 2%, `F` is still the only edit this method makes: it stays uploaded, and the failed reading and
its numbers come first in `REPORT.md`.

**Upload commands** (in `/workspace/work`; write `REPORT.md` first):

```
python -m wsp.upload rq F --report REPORT.md --task task.json --dry-run     # prints: [upload] F stacks on 0009568831-0002
axonry-check /workspace/work/wsp/deliver/rq__F/edit --task task.json        # the admission check: "passed": true
python -m wsp.upload rq F --report REPORT.md --task task.json               # ends with: [upload] verified: ...
```

Upload only when `axonry-check` prints `"passed": true` (it reads the parent from
`/workspace/work/0009568831-0002/edit/`). If the upload itself errors, run it again at once. If `axonry-check`
fails, do not upload: write its output in `cases/rq/NOTES.md` and `PROGRESS.md`. If the failure is a missing file or
a wrong path, fix that, run `axonry-check` again and upload when it passes. If it fails on a limit (payload or
relative change) and `F` has both layers, build `F` from W alone: stop the chain and its job as in "If time runs
short" below, then, in `/workspace/work`:

```
mkdir -p rq/full && mv rq/done/edits_w rq/done/check rq/done/read_a rq/test_F rq/word_F rq/runs/reqdev.F.jsonl rq/full/ 2>/dev/null
setsid nohup python derive/chain.py --case rq --out /workspace/work/rq --w-only >> /workspace/work/chain.log 2>&1 < /dev/null &
```

When `edits_w` is done again, run `axonry-check` again and upload; say first in `REPORT.md` that `F` is the workings
layer alone because the two layers together failed the admission check, with the check's line. If `axonry-check`
still fails, there is no edit to upload: run `axonry-done`.

`REPORT.md` for the second upload, in the brief's four parts. The failure targeted: section 1.1 (puzzles) and, for T,
section 1.3 (instruction following on requests). How the edit was derived: section 1 and the chain's steps. For W:
the last lines of `rq/logs/fit_W.log` (capital W: the trainer's log). For T: the `req  points ... pairs` line of
`rq/logs/branch.log`, the `[train] ... neurons over ... layers` line of `rq/logs/fit_TSEL.log`, the last lines of
`rq/logs/fit_T0.log` and `rq/logs/fit_T14.log`, and the dose 0.35. For both: the last `[encode]` line of
`rq/logs/edits_w.log` (the two layers together in int8 form). If the chain ran with `--w-only`, the first line says
that `F` is the workings layer alone, and why. Controls: the five lines of `rq/logs/check.log` named above and
the `cmp.py` table of the dev requests, with the two draws of the start as the noise scale. What did not work:
section 5, and any failed reading of this run. Keep the credentials of `task.json` out of it.

After the second upload (its last line `[upload] verified: ...`) the method is done, about 3 hours into the run:
update `PROGRESS.md` and run `axonry-done`. Do not start the chain's `--final` steps (`read_b`, `freeze`, `read_c`,
`read_d`, about two and a half hours): whatever they show, they change nothing about the edit.

**If a step fails.** The chain starts each command three times before it stops. Read the `FAILED` line and the log
tail in `chain.log` first (for `roll_a`, `roll_b` and `branch` the workers' own logs are `rq/roll/r1_gpu<N>.log`,
`rq/roll/r2_gpu<N>.log` and `rq/branch/gpu<N>.log`). If `cases/rq/edits/F.json` does not exist yet, remove a cause
that lies outside the method (a full disk, a process still holding GPU memory: stop that process by its PID) and
start the chain again with the line at the top of this section; finished steps are skipped. If `prep`, `keep` or
`fit_w` still fails after that, there is no edit: write the `FAILED` line and the log tail in `cases/rq/NOTES.md` and
`PROGRESS.md` and run `axonry-done`; do not change the scripts in `derive/`. If `edits_w` still fails in
`union_sparse.py`, `encode_dirs.py` or `stack.py`, start the chain once with `--w-only` (below); if that fails too,
there is no edit either.

If the step that fails three times is one of the request layer's (`roll_a`, `roll_b`, `branch`, `fits_t`), or
`edits_w` fails three times in `avg_sparse.py` or `dose_sparse.py`, and no cause outside the method is found, start
the chain with

```
setsid nohup python derive/chain.py --case rq --out /workspace/work/rq --w-only >> /workspace/work/chain.log 2>&1 < /dev/null &
```

`fit_w` is not run again. The request layer's steps that are not done yet are skipped (`chain.log` shows
`<step>: skipped (--w-only)`, `--status` shows `skipped`; finished ones stay `done`), and `edits_w` builds `F` from W
alone in a few minutes; `check` and
`read_a` follow. Say first in `REPORT.md` that `F` is the workings layer alone, and which step failed.

If `F.json` exists, upload `F` as above if it is not uploaded yet, then remove the cause and start the chain again
once for the missing readings (with `--w-only` again if the last start had it); if a later step still cannot be
finished, say in `REPORT.md` which reading is missing.

**A step that runs long without failing.** The chain watches its generation jobs itself, but not the trainer. If
`fit_w` passes 90 minutes or `fits_t` 70, look at the trainer's log: `rq/logs/fit_W.log`, or in `fits_t`
`rq/logs/fit_TSEL.log` (a pass without training: it has no `[train] step` lines and is finished at `[train] done`)
and then `rq/logs/fit_T0.log` and `rq/logs/fit_T14.log`, which run side by side (their first `[train] step` line
comes only after a reference pass over all pairs that writes nothing). If a log has not grown for 30 minutes, stop
only that fit's processes by their PIDs: `ps -eo pid,args | grep -- '--out-spec [^ ]*/T14.json'` (`T0.json`,
`TSEL.json` or `W.json` for the others). The chain starts that command again, and the other fit goes on.

**If time runs short.** The first upload must happen by 4 hours 30 minutes into the run at the latest. If at 4 hours
15 minutes `cases/rq/edits/F.json` does not exist and `--status` shows `fit_w    done`, do not wait for the request
layer any longer (if `fit_w` itself is not done by then, there is no edit: stop the chain as below, write it in
`cases/rq/NOTES.md` and `PROGRESS.md` and run `axonry-done`). Stop the chain first (so that it starts nothing again),
then the job it runs, each by its PID or
process group as printed by this command, never by a pattern:

```
ps -eo pid,pgid,args | grep -E '[c]hain.py|[t]orchrun|[t]rain_keep.py|[x]_rollouts.py|[x]_branch.py|[g]en_x.py'
kill <PID of the chain.py line>
kill -- -<PGID of each remaining line>
nvidia-smi --query-gpu=memory.used --format=csv,noheader      # wait until all four show a few MiB
```

Then start the chain with `--w-only` (the command above): it builds `F` from W alone without fitting again.

## 4. What to expect

On practice data, from our own full-size run on this start (the puzzles: the toolkit's test part, 40 a family with
two answers each):

| practice measure | the start | `F` |
|---|---|---|
| test puzzles, families with a working | 75.2% right, 9,100 tokens an answer | 93.2% right (+18.0 ± 1.6), 380 tokens |
| test puzzles, families without a working | 60.4% right | 60.4% (+0.0 ± 3.2) |
| word problems (300 x 2) | 77.0% right | 77.0% (+0.0 ± 1.5) |

- Per family after the fit: 12 of the sixteen read 96.2 to 100.0% right; the shape a path draws 86.2%, a walk that may
  return 83.8%, an order of five things 78.8%, an order of seven things 58.8% (51.2%, 73.8%, 53.8%, 52.5% before).
- The fit: 4799 workings and 1595 keep and hold rows, 650 neurons over 22 layers, one pass, about 49 minutes on two
  RTX 4090; the loss on held-back rows falls from 1.18 to 0.08.
- The two layers in int8 form (`WD`) are 4,006,200 bytes (3900 ops, one per changed row or column; the limit is
  4,194,304) and differ from the fitted change by 0.79% of their length.
- Answers to puzzles of the families with a working become 177 to 531 tokens long (an order of seven things about
  1,700); everything else keeps roughly its length (puzzles without a working 9,540 -> 9,384 tokens, word problems
  6,118 -> 6,239).
- The request layer: 670 pairs and 1,156 keep rows (624 pairs and 1,081 keep rows in the training split, the rest held
  back for the validation loss), 650 neurons over 23 layers outside W's, two fits of two passes, about 35 minutes side
  by side on two RTX 4090 each; delivered at dose 0.35.

## 5. What does not add (practice data; do not repeat)

- Fewer than 300 workings a family: with the practice set alone (17 to 96 a family) a fit of this kind on another
  start read +3.1 ± 1.5 on the test part against `0009554110-0010`, instead of +14.7 ± 1.4 (section 1.1).
- A working for word sorting. We wrote one (every word gets a number key from its first letters, a=01 to z=26, and
  the words are inserted one by one into a list ordered by key) and fitted it beside the other workings in a
  full-size run of the workings part of this procedure on another published start. The family read 40.0% right against
  the start's 50.0% (-10.0 ± 8.4
  on its 40 test puzzles, two answers each). The model took the layout but not the procedure: it was right in 23 of
  30 answers to lists of at most ten words that two letters tell apart, and in 9 of 50 answers to the other lists,
  where it kept keys too short to tell the words apart, lost words or misplaced them. Two more fits of ours with
  number-key workings read the family 15.0 ± 4.1 points (on that same start) and 21.9 ± 8.1 points (on a third one)
  below their starts on practice puzzles. So word sorting gets no working: it is held like dates and formal
  arguments.

