# The procedure written out: compact workings for puzzle families the model misreads, fitted with a keep term, on top of every-thinking-branched instruction pairs

This entry starts from the champion (it was written while `0009554110-0010` held that place) and names
`0009554615-0007` as its parent in `entry.json`, because that directory holds three ops this entry takes over. It
delivers one edit made of three parts, all on the champion's line:

- **U, the parent's unit.** The three ops that `0009554615-0007` added to its own start (one MLP unit of the last
  layer that ends a thinking that has stopped going anywhere) are read from the parent's published `edit.json` and
  put on the champion's model as they are (`derive/take_unit.py`). Nothing is recomputed. `0009559255-0006` put the
  same unit on the same model before us.
- **Ba, about 650 MLP units fitted on answer-stage pairs of instruction requests written in the run**, with every
  finished thinking branched (section 1.2). The fit is the published answer-stage preference of `0009550031-0005`
  (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`,
  `derive/rollouts.py`, unchanged).
- **W, about 650 other MLP units fitted on workings** (section 1.1): for practice puzzles of fourteen families whose
  questions always have one form, a program writes the short thinking the puzzle needs, and the model learns to
  write it. While it learns, a keep term holds its next-token distribution to the starting model's on its own
  answers to everything else. This part is the idea of this entry.

Ba and W use different neurons. Written as one direction per neuron in int8 (`derive/encode_dirs.py`), both fit
in the payload together with U.

The run makes the edit in two rounds. `Fa` = U + Ba is uploaded first, after about one hour and twenty minutes.
`F` = `Fa` + W replaces it only if it passes a check in free generation (section 3, step C).

Where it comes from, plainly: the unit is the published work of the authors of `0009549068-0003` and
`0009554615-0007`, the answer-stage preference fit that of `0009550031-0005`. Ours: the workings and how they are
fitted (full strength, keep term, protected neurons, which families get one), the requests written in the run,
every thinking branched with eight new answers, the int8 form that lets two neuron layers share one payload.

## 1. What the starting model loses, and what the edit does about it

### 1.1 Puzzles with a fixed procedure: the model misreads the task

`derive/puzzles.py` writes practice puzzles of nineteen families from word lists and random draws (schedules, adjective
order, shapes drawn by a path, counting, swaps, orderings, walks, truth chains, expressions, dates, tables and so
on; every generated puzzle is checked against the toolkit's benchmark fingerprints). The starting model on 1,136
of them (60 per family, two answers each, full token limit): **77.5% right, 10,400 tokens per answer.**

| family | right % | what goes wrong (read from its answers) |
|---|---|---|
| a time window that is left free | 35.8 | it lists the times the person was seen elsewhere, then picks one of them |
| adjective order | 51.7 | no rule; it argues by sound, or numbers the words in the order they stand |
| the shape a path draws | 55.0 | it counts the closing point as a corner; an arc is not read at all |
| an order of seven things | 63.3 | "third-newest" and "third-oldest" are mixed up |
| counting things of a category | 73.3 | it counts kinds instead of adding quantities, or keeps the odd one in |
| an order of five things | 75.8 | as with seven |
| swaps among five | 79.2 | a swap is applied to the wrong pair late in a long thinking |
| a walk that may return | 80.0 | a turn is lost |

These are misreadings of the task, the same every time, and after a right working the model often thinks on for
thousands of tokens and talks itself out of it. What we tried, all on practice puzzles and against two answers of the
start on the same puzzles:

- **Its own right answers preferred to its own wrong ones** (the published preference fit, 160 to 444 pairs, with and
  without its right answers as likelihood rows): -0.4 ± 0.9, +0.6 ± 0.9, -0.2 ± 0.9. A right answer of the start is
  often a lucky one and does not show the procedure.
- **The procedure stated at the start of its thinking** (forced, 40 puzzles per family): +20 ± 6 on time windows and,
  with a made-up example of the layout, +20 ± 7 on adjective order; nothing or worse on six other families. It does
  not follow a layout it is shown once, and its answers stay 10,000 tokens long.
- **The working written out and learned** (this entry): for a puzzle, `derive/workings.py` writes the facts in a
  fixed layout, the one rule of the family, the conclusion, the end of the thinking, one sentence and the answer:
  about 300 tokens. It is written from the puzzle text alone and checked against the puzzle's known answer; a puzzle
  whose working does not reproduce it is left out. Fitted as likelihood rows on 650 neurons.

| five families, 300 test puzzles x 2 | start | fit at 0.10 | at 0.25 | at 0.5 | **the fit itself (1.0)** |
|---|---|---|---|---|---|
| right % | 53.8 | 56.7 | 55.2 | 53.3 | **83.5 (+29.7 ± 3.0)** |
| answers that use the working's layout | 0% | | 0% | | 100% |
| tokens per answer | 10,870 | 10,004 | 9,442 | 8,282 | 353 |

**It is a threshold, not a dose.** Below the full fit the first tokens stay the old ones and the rest only adds
noise. The workings layer is delivered at full strength; it is never dosed.

**Which families get a working.** Fourteen whose questions always have one form: time windows, adjective order,
shapes, counting, swaps (3, 5, 7), orderings (3, 5, 7), walks, truth chains, boolean and arithmetic expressions.
Five families keep the model's own thinking and are only held in place (below): dates, the two table families and
formal arguments, whose questions come in more forms than one layout covers, and word sorting, where a compact
working fails on letter order itself (4% right when words are inserted one by one; the model compares letters
correctly only by spelling things out at length).

**Unprotected, the short style leaks.** The first fit for all families read +12.2 ± 1.2 on the test puzzles, but
-3.2 ± 1.3 on held-out instruction requests and -2.1 ± 1.2 on word problems, with shorter answers there. Three
things remove that (two independent fits; against the start):

1. **Keep rows.** The model the fit starts from answers once: 450 of the run's requests, 300 word problems, 300 math
   problems, 150 multiple-choice questions and 200 puzzles of each family without a working. Its right answers are
   rows of type `anchor`; `derive/train_keep.py` (the published trainer with one more term) adds on each of them the
   KL divergence of the start's next-token distribution to the edited model's, per token, weight 100. A likelihood
   anchor alone does not hold anything: measured on the start's own answers to other tasks, the edited model's
   distribution moved as far with 1,600 likelihood anchors as with none.
2. **Protected neurons.** The 20,000 neurons with the largest gradient on the keep rows of other tasks are never
   chosen (`--protect`, `--protect-n`); the chosen ones move to the first ten layers.
3. **Other neurons than Ba's** (`--exclude-neurons-from`), so the instruction layer is not written over.

| Fa + W, against the start | fit 1 | fit 2 |
|---|---|---|
| test puzzles (1,136 x 2) | +14.4 ± 1.1 (91.9%) | +14.6 ± 1.1 (92.1%) |
| families with a working, after the fit | 82 to 100% right, 160 to 1,200 tokens | the same |
| families without a working | within 5 points of the start | within 5 points |
| held-out requests (600 x 2); the instruction layer alone: +5.8 ± 1.3 | +6.4 ± 1.2 | +7.8 ± 1.3 |
| word problems (600 x 2) | +0.5 ± 1.1 | +0.8 ± 1.0 |
| math (600 x 2; an earlier fit of the same kind) | 0.0 ± 1.2 | |

- 300 workings per family are needed: with 100, boolean expressions, counting, walks and five-thing orderings read
  5 to 15 points lower.
- A larger keep weight (400) changes nothing; likelihood anchors with a weight of 3 and no KL term hold word problems
  but give back a point of the instruction gain.
- Pairs "working over the model's own wrong answer" are not needed: a third fit without them, with keep rows
  from its own start (the way the chain does it), reads +12.3 ± 1.2 on the test puzzles, +5.1 ± 1.4 on held-out
  requests and +1.7 ± 1.0 on word problems (its start also had the conclude and loop directions of
  `0009558523-0017` removed; they are not part of this entry, section 5).

### 1.2 Instruction requests: every thinking branched

**The practice requests are used up, so the run writes its own.** The model follows about 90% of the practice
instruction requests. `derive/reqgen.py` writes 3,000 requests the way the toolkit builds `practice:ifsynth` (a
request plus constraints whose arguments the checker library draws itself, graded by the toolkit's grader), with
these differences: the request is a writing or explaining task (40 task sentences x 100 everyday topics, all in the
file), not a word problem; every instruction type that the shipped practice sets contain is used (`ifsynth` and
`tuluif` together: 23 types), equally often, and no other; a constraint is worded as the library words it or in one
of a few other ways; and one, two or three constraints stand together in equal shares. The starting model follows
78.4% of 600 held-out requests (one constraint 91%, two 80%, three 62%).

**The answer stage decides rarely.** 2,400 training requests, two answers each (limit 6,144 tokens), two more where
the first two agree (limit 12,288): about 8,700 answers, 7,700 of them with a finished thinking. After the same
thinking, eight new answers: the outcome is mixed for 9.5% of the thinkings, 1,156 pairs. The published recipe
branches one thinking per request with four answers: 187 pairs.

**More pairs make a better edit** (four fits of one neuron set, their average, dose 0.35; 600 held-out requests,
against the starting model): 187 pairs +4.3 ± 1.1; 632 pairs (every thinking, four answers) +6.8 ± 1.2; 1,156 pairs
+5.8 ± 1.3 on the same requests and the best of the three on other request sets. Answers do not get longer. On
problems no fit saw (the 632-pair edit): word problems +0.6 ± 1.0, multiple choice -0.9 ± 1.0, math -0.3 ± 0.8.

### 1.3 The unit on this start

On 600 practice math problems (two answers each) the start runs into the token limit in 4.9% of its answers and is
right in 61.1%; with U the limit is hit in 0.4% and 63.2% are right. On hard practice problems about a fifth of the
start's answers end without a result; U turns most of those into finished answers. Both layers of this entry are
fitted on the model with U. If the champion in `task.json` already carries the unit, section 2 says what to do.

**One reading is one draw.** Generation is seeded: a model run twice on a set returns the same answers. The null
edit (`derive/null_edit.py`: one MLP unit of about 65,000 scaled by 1.05) gives a second draw of the starting model;
`derive/cmp.py` pairs an edit with the mean of the two draws. Count answers before believing a gain or a loss of two
points.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009554615-0007/edit/` and the champion `<its id>/edit/` (what each builds on is inside it). On the
agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009554615-0007 sandbox:/workspace/work/   # always: the unit is read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on.** This entry was planned while `0009554110-0010` was the champion.

- The champion in `task.json` is `0009554110-0010`: start from it and make all three parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one
  (`--parent <champion id>/edit/edit.json` at `init`). If `0009554615-0007` is in its lineage, or its report
  describes a last-layer unit that ends a thinking that has stopped going anywhere, it carries the unit: skip
  `take_unit.py` and start the chain with `--no-unit`.
- No champion directory in the bundle: the parent is the only start. Start from `0009554615-0007/edit/edit.json`
  (the unit is in it), skip `take_unit.py`, start the chain with `--no-unit`.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init rq --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python derive/take_unit.py --case rq --from 0009554615-0007/edit --name U      # the unit as the case's edit U
python derive/reqgen.py --out-dir /workspace/work/rq/req                        # 3,000 requests: train / dev / hold
python derive/null_edit.py --case rq                                            # the edit NULL: a second draw
python -c "from wsp import case; [case.model_path('rq', m) for m in ('base', 'NULL', 'U')]"   # build ONCE, before any GPU job
```

- `take_unit.py` prints the three ops it took (`set_rows` on the last layer's `gate_proj` and `up_proj`, `set_cols`
  on its `down_proj`, one unit).
- `reqgen.py` prints the counts (3,000 requests, 23 instruction types; 2,400 for the fit, 300 dev, 300 holdout) and
  how many it dropped because their fingerprint is among the toolkit's benchmark fingerprints (0).
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data, the requests and puzzles written by `derive/`, and the models' own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

One command runs every GPU step in order; a finished step is skipped when the chain is started again:

```
nohup python derive/chain.py --case rq --out /workspace/work/rq > /workspace/work/chain.log 2>&1 &
python derive/chain.py --case rq --out /workspace/work/rq --status
```

| step | what | minutes on four RTX 4090 |
|---|---|---|
| `prep` | practice puzzles written: for the fit, for the test | 1 |
| `roll_a`, `roll_b`, `branch` | answers of `U` to the 2,400 training requests; every finished thinking branched | 32 |
| `fits_a`, `edits_a` | the neurons chosen once, four fits two by two (about 30 minutes a pair of fits for 1,300 pairs), their average at 0.35 -> `Ba`; `Fa` built | 65 |
| `keep` | `Fa` answers once what it has to keep; its right answers become the keep rows | 15 |
| `fit_w` | GPUs 0,1: the workings fitted on `Fa` with the keep term; GPUs 2,3: `Fa` on the test puzzles and word problems | 40 |
| `edits_w` | `Ba` and the workings layer as one neuron set in int8 form; `F` built | 4 |
| `check` | `F` on the same test puzzles and word problems; the verdict line | 10 |
| `read_a` | two draws of the start, `Fa` and `F` on the 300 held-out requests of the dev split | 10 |

The times are from our own runs of each step on that hardware; the whole chain takes about three hours.

**Step B (after `edits_a`): upload `Fa` at once.** It is a complete candidate.
`python -m wsp.upload rq Fa --report REPORT.md --task task.json` (first with `--dry-run`).

**Step C (after `read_a`): `F` or `Fa`.** `F` is final only if all three hold; otherwise `Fa` stays:

1. the last line of `logs/check.log` is `CHECK: PASS` (families with a working at least +8 points against `Fa` on the
   test puzzles, families without one not below -8, word problems not below -4; we measured +20, -2 and +1);
2. `logs/edits_w.log` shows a payload below the limit for `UWD` and an encoded change within 2% of the wanted one
   (we measured 0.8%);
3. on the held-out requests, `F` is not more than 4 points below `Fa`
   (`python derive/cmp.py runs/reqdev.base.jsonl runs/reqdev.NULL.jsonl --edit Fa=runs/reqdev.Fa.jsonl F=runs/reqdev.F.jsonl`;
   we measured between -1 and +2).

Then upload the final edit the same way, with `REPORT.md` holding what the chain measured (the verdict line and its
three numbers, the payload and encoding lines of `logs/edits_w.log`, the held-out requests table with the two draws
of the start as the noise scale, and which of the two edits is final and why). Check that the upload is complete and
run `axonry-done`: the task is finished, about three and a half hours after the start. Nothing run after the final upload
changes the edit, so the run ends here. Do not start the chain's `--final` steps in this run (the final edit and the
two draws on the case's dev and holdout sets, about two and a half hours): every number this entry decides on is
read on held-out practice data already, by `check` and `read_a`.

## 4. What to expect

On practice data, from our own derivations of both parts (three independent workings fits on two instruction
layers):

| against the start | `Fa` | `F` |
|---|---|---|
| held-out requests (600 x 2) | +5.8 ± 1.3 | +5.1 to +7.8 (± 1.3) |
| test puzzles, families with a working | 0 | from about 77% to 95-100% right in most families (orderings of seven things 70-86%) |
| test puzzles, families without a working | 0 | within 5 points |
| word problems (600 x 2) | -0.4 ± 1.0 | +0.5 to +1.7 (± 1.0) |
| math (600 x 2) | | 0.0 ± 1.2 |

- The workings layer changes about 25% of the values of its 650 neurons; `Ba` about 3.5% of its own. Per tensor the
  change stays far below the limits (the report prints the largest).
- The packed edit is about 4.01 MB (limit 4,194,304 bytes); the int8 form differs from the fitted change by 0.8% of
  its length and reads the same on the test puzzles (88.7% against 88.8%).
- Answers to puzzles of the fourteen families become 150 to 1,200 tokens long; everything else keeps its length.

## 5. What does not add on this start (practice data; do not repeat)

- The workings layer at a dose below 1: nothing at 0.10 to 0.5, harm at 0.25 to 0.75 on other parents.
- Its own right answers preferred to its own wrong ones on puzzles; the procedure stated in the prompt (-3) or forced
  at the start of the thinking (helps two families, hurts others).
- Likelihood anchors without the KL term; a keep weight above 100; fewer than 300 workings per family.
- A working for word sorting (insertion word by word 4%; walking through the alphabet 60%; number keys of the first
  letters read 92% as a fit of their own but 67% inside the joint fit) and for the open-form families.
- The conclude and loop directions of `0009558523-0017` measured on this start and removed from four attention
  outputs: word problems -0.6 ± 0.8 alone and +0.2 ± 0.8 under this entry's layers, answers a fifth shorter.
- A third neuron layer for word problems: three layers need about 1,700 neurons and the payload holds 1,360; fitted
  on the instruction layer's own neurons it adds nothing.
- An end-of-answer "always box" preference for math: it turns answers without a result into wrong ones, not right ones.
