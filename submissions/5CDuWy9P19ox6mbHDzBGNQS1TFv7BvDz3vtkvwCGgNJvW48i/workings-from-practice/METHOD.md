# Short workings for fixed-form puzzles, written by a program for the toolkit's own practice puzzles and fitted with a keep term, on top of every-thinking-branched instruction pairs

This entry starts from the champion (it was written while `0009554110-0010` held that place) and names
`0009554615-0007` as its parent in `entry.json`, because that directory holds three ops this entry takes over. It
delivers one edit made of three parts, all on the champion's line:

- **U, the parent's unit.** The three ops that `0009554615-0007` added to its own start (one MLP unit of the last
  layer that ends a thinking that has stopped going anywhere) are read from the parent's published `edit.json` and
  put on the champion's model as they are (`derive/take_unit.py`). Nothing is recomputed.
- **Ba, about 650 MLP units fitted on answer-stage pairs of instruction requests written in the run**, with every
  finished thinking branched (section 1.2). The fit is the published answer-stage preference of `0009550031-0005`
  (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`,
  `derive/rollouts.py`, unchanged).
- **W, about 650 other MLP units fitted on workings** (section 1.1): for the puzzles of the toolkit's practice set
  `bbhsynth` whose kind always has one form, a program writes the short thinking the puzzle needs, and the model
  learns to write it. While it learns, a keep term holds its next-token distribution to the starting model's on its
  own answers to everything else. This part is the idea of this entry.

Ba and W use different neurons. Written as int8 directions, one per changed row or column
(`derive/encode_dirs.py`), both fit in the payload together with U.

The run makes the edit in two rounds. `Fa` = U + Ba is uploaded first. `F` = `Fa` + W replaces it only if it passes a
check in free generation (section 3, step C).

**Every puzzle this entry uses is an item of `practice:bbhsynth`, as shipped.** `derive/pz_practice.py` reads the set
through the toolkit and splits it by item number; it writes, rewords and generates nothing, and `derive/` holds no
puzzle generator. `derive/workings.py` reads the practice puzzles: it knows their sentence frames and their few fixed
words (it has to, to parse them), and takes the larger word lists (the adjectives of each class, the things of each
counted category, number words) from the toolkit's `wsp/puzzles.py` when it is loaded. Every puzzle number in this
text was measured on that practice set.

Where it comes from, plainly: the unit is the published work of the authors of `0009549068-0003` and
`0009554615-0007`, the answer-stage preference fit that of `0009550031-0005`. Ours: the workings and how they are
fitted (full strength, keep term, protected neurons, which kinds get one), the requests written in the run, every
thinking branched with eight new answers, the int8 form that lets two neuron layers share one payload.

## 1. What the starting model loses, and what the edit does about it

### 1.1 Puzzles with a fixed form: thousands of tokens, and the thread is lost

`practice:bbhsynth` holds 128 puzzles of each of 19 kinds. For every kind, items 0 to 95 are for the fits and items
96 to 127 are the test (never in a fit). The champion on the 608 test puzzles, two answers each, full token limit:

| kind | puzzles | right % | tokens per answer | at the token limit % |
|---|---|---|---|---|
| a time window that is left free | 32 | 26.6 | 12,519 | 0.0 |
| an order of five things | 32 | 45.3 | 15,324 | 1.6 |
| an order of seven things | 32 | 45.3 | 19,762 | 6.2 |
| the shape a path draws | 32 | 51.6 | 12,975 | 1.6 |
| counting things of a category | 32 | 57.8 | 7,362 | 0.0 |
| adjective order | 32 | 62.5 | 9,109 | 0.0 |
| a walk that may return | 32 | 64.1 | 11,367 | 0.0 |
| swaps among five | 32 | 75.0 | 11,559 | 0.0 |
| an order of three things | 32 | 78.1 | 8,917 | 0.0 |
| a boolean expression | 32 | 82.8 | 10,039 | 0.0 |
| swaps among seven | 32 | 84.4 | 14,543 | 3.1 |
| a chain of truth-tellers and liars | 32 | 93.8 | 8,141 | 0.0 |
| swaps among three | 32 | 95.3 | 9,711 | 0.0 |
| an arithmetic expression | 32 | 100.0 | 5,954 | 0.0 |
| word sorting (no working) | 32 | 53.1 | 12,742 | 1.6 |
| dates (no working) | 32 | 64.1 | 7,707 | 0.0 |
| formal arguments (no working) | 32 | 64.1 | 11,091 | 0.0 |
| a table of animals (no working) | 32 | 95.3 | 7,004 | 0.0 |
| coloured things on a surface (no working) | 32 | 98.4 | 5,888 | 0.0 |
| **kinds with a working** | 448 | 68.8 | 11,234 | 0.9 |
| **kinds without a working** | 160 | 75.0 | 8,886 | 0.3 |
| **all** | 608 | 70.4 | 10,617 | 0.7 |

The answers are about 10,600 tokens long, and almost none of them hits the token limit: the thinking ends, but wrong.

What goes wrong, read from its wrong answers to these puzzles:

- **A time window that is left free (27%).** It lists the times at which the person was seen elsewhere, and then
  never checks the options against that list: it ends with "all options are possible" and picks the earliest or the
  longest one.
- **An order of five or seven things (45%).** After 15,000 to 25,000 tokens a rank is read from the wrong end
  ("the third-newest" becomes position three from the old side, or simply "the newest"), and the answer contradicts
  its own earlier lines.
- **The shape a path draws (52%).** The closing point is counted as one more corner (four sides become "a
  pentagon"), and four-sided shapes are named by impression.
- **Counting things of a category (58%).** It adds every quantity in the list, also of the things that are not of
  the category asked for.
- **Adjective order (63%).** It has no fixed order of the classes; it makes one up ("origin first", "what describes
  before what judges") and argues for the option it met first.
- **A walk that may return (64%).** After two turns the heading is lost: a turn is treated as a move, or the steps
  after it are added in the old direction.

The procedure of such a puzzle is short and always the same; the model does not lack the steps, it lacks the
discipline to do only them. Hence the edit:

- **The working written out and learned.** For a puzzle, `derive/workings.py` writes the facts in a fixed layout, the
  one rule of the kind, the conclusion, the end of the thinking, one sentence and the answer: about 300 tokens. It
  is written from the puzzle text alone and checked against the puzzle's known answer; a puzzle whose working does
  not reproduce the answer is left out. The model is fitted to write the working itself (likelihood rows, about 650
  neurons, full strength: the layout is either written or not, so this layer is never dosed).
- **Which kinds get a working.** Fourteen whose questions always have one form: time windows, adjective order,
  shapes drawn by a path, counting, swaps (3, 5, 7 people), orderings (3, 5, 7 things), walks, truth chains, boolean
  and arithmetic expressions. The writer covers **1,778 of the 1,792** practice puzzles of these kinds (every kind
  128 of 128, shapes 114 of 128: the 14 trapezoids are left out, a compact working cannot tell a trapezoid from a
  kite without long arithmetic). Five kinds keep the model's own thinking and are only held in place: dates, the
  two table kinds and formal arguments, whose questions come in more forms than one layout covers, and word
  sorting, which needs letter-by-letter comparisons the model only gets right at length.
- **The practice set is small, so every working is used three times.** The fit split gives about 1,340 workings
  (96 puzzles of a kind, less the trapezoids). In the fit's data each of them stands three times and each keep row
  (below) once; the fit makes one pass over that file.

**A fit like this must be held in place.** A short answering style learned on one family of prompts is cheap to
apply to every other prompt, and nothing in a likelihood fit on puzzles alone says where it should stop. Three
things hold it (all in `derive/chain.py`, step `fit_w`):

1. **Keep rows and hold rows.** The model the fit starts from (`Fa`) answers once: 450 of the run's requests, 300
   word problems, 300 math problems, 450 multiple-choice questions and the 480 fit puzzles of the five kinds without
   a working. Its right answers are keep rows (type `anchor`): `derive/train_keep.py` (the published trainer with
   one more term) adds on each of them the KL divergence of the start's next-token distribution to the edited
   model's, per token, weight 100. A likelihood anchor only says "this sampled answer stays likely"; the KL term
   says "nothing about what the model would write here changes". Its other finished answers are hold rows
   (`--hold-ids`): the KL term alone, so that what the model writes on these prompts stays as it is whether it was
   right or not, and no wrong answer is made more likely. The multiple-choice questions get the largest share next
   to the requests (450 against 300 word and 300 math problems): a list of options is what they share with a
   puzzle, and less than half of the model's multiple-choice answers are right, so the hold rows double what is
   held there.
2. **Protected neurons.** The 20,000 neurons with the largest gradient on the keep rows of other tasks are never
   chosen (`--protect`, `--protect-n`).
3. **Other neurons than Ba's** (`--exclude-neurons-from`), so the instruction layer is not written over.

**What the fit does** (our run of the steps `keep` to `check` of this entry as they stand, at full size, on an
instruction layer of this recipe from an earlier derivation; the same 608 test puzzles, two answers each, before and
after the workings are fitted, paired by puzzle):

| kind | puzzles | `Fa`: right % | tokens | `F`: right % | tokens | difference |
|---|---|---|---|---|---|---|
| a time window that is left free | 32 | 37.5 | 11,916 | 96.9 | 333 | +59.4 ± 7.6 |
| an order of seven things | 32 | 50.0 | 18,068 | 50.0 | 1,284 | +0.0 ± 7.8 |
| the shape a path draws | 32 | 51.6 | 11,572 | 84.4 | 309 | +32.8 ± 9.7 |
| an order of five things | 32 | 51.6 | 12,525 | 81.2 | 407 | +29.7 ± 10.3 |
| adjective order | 32 | 59.4 | 8,826 | 100.0 | 223 | +40.6 ± 8.2 |
| counting things of a category | 32 | 60.9 | 6,904 | 90.6 | 175 | +29.7 ± 10.0 |
| a walk that may return | 32 | 62.5 | 11,835 | 81.2 | 255 | +18.8 ± 7.7 |
| an order of three things | 32 | 73.4 | 7,097 | 98.4 | 262 | +25.0 ± 7.4 |
| swaps among five | 32 | 82.8 | 10,285 | 100.0 | 286 | +17.2 ± 6.2 |
| swaps among seven | 32 | 87.5 | 11,432 | 92.2 | 462 | +4.7 ± 5.7 |
| a boolean expression | 32 | 89.1 | 9,234 | 95.3 | 254 | +6.2 ± 5.4 |
| swaps among three | 32 | 92.2 | 8,789 | 100.0 | 177 | +7.8 ± 3.3 |
| an arithmetic expression | 32 | 93.8 | 6,086 | 95.3 | 197 | +1.6 ± 4.2 |
| a chain of truth-tellers and liars | 32 | 95.3 | 7,787 | 98.4 | 197 | +3.1 ± 3.1 |
| word sorting (no working) | 32 | 45.3 | 9,820 | 48.4 | 10,511 | +3.1 ± 5.5 |
| formal arguments (no working) | 32 | 57.8 | 10,944 | 60.9 | 11,214 | +3.1 ± 7.8 |
| dates (no working) | 32 | 78.1 | 6,835 | 75.0 | 6,790 | -3.1 ± 5.0 |
| a table of animals (no working) | 32 | 92.2 | 5,865 | 93.8 | 5,823 | +1.6 ± 2.7 |
| coloured things on a surface (no working) | 32 | 95.3 | 5,511 | 96.9 | 5,954 | +1.6 ± 2.7 |
| **kinds with a working** | 448 | 70.5 | 10,168 | 90.3 | 345 | +19.8 ± 2.1 |
| **kinds without a working** | 160 | 73.8 | 7,795 | 75.0 | 8,058 | +1.2 ± 2.3 |
| **all** | 608 | 71.4 | 9,544 | 86.3 | 2,374 | +14.9 ± 1.7 |

On the same check, problems no fit saw: 300 word problems 77.3% -> 78.3% (+1.0 ± 1.7), 300 multiple-choice questions
47.7% -> 49.7% (+2.0 ± 1.9), answers as long as before. The verdict line was `CHECK: PASS`. The fit took 1,334
workings (each three times), 1,232 keep rows and 334 hold rows; its validation loss went from 1.18 to 0.08.

The run reads the same table for its own fit (step `check`); `F` is delivered only if the fourteen kinds gain at
least 8 points and nothing else falls below its floor (step C); otherwise the run delivers `Fa`.

### 1.2 Instruction requests: every thinking branched

**The practice requests are used up, so the run writes its own.** The model follows about 90% of the practice
instruction requests. `derive/reqgen.py` writes 3,000 requests the way the toolkit builds `practice:ifsynth` (a
request plus constraints whose arguments the checker library draws itself, graded by the toolkit's grader), with
these differences: the request is a writing or explaining task (40 task sentences x 100 everyday topics, all in the
file), not a word problem; every instruction type that the shipped practice sets contain is used (`ifsynth` and
`tuluif` together: 23 types), equally often, and no other; a constraint of a type that `ifsynth`
has is worded as that set words it (the checker library's sentence) or in one of a few other ways, a constraint of
the three types only `tuluif` has in one of the other ways; and one, two or three constraints stand together in
equal shares. The starting model follows
about 78% of the held-out requests (one constraint 91%, two 80%, three 62%).

**The answer stage decides rarely.** 2,400 training requests, two answers each (limit 6,144 tokens), two more where
the first two agree (limit 12,288): about 8,700 answers, 7,700 of them with a finished thinking. After the same
thinking, eight new answers: the outcome is mixed for about a tenth of the thinkings, 1,156 pairs in one of our
derivations and 1,333 in another. The published recipe branches one thinking per request with four answers: 187
pairs.

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
edit (`derive/null_edit.py`: one MLP unit of about 86,000 scaled by 1.05) gives a second draw of the starting model;
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

(In the two cases above that skip `take_unit.py`, leave `'U'` out of the last line: the case has no edit U then.)

- `take_unit.py` prints the three ops it took (`set_rows` on the last layer's `gate_proj` and `up_proj`, `set_cols`
  on its `down_proj`, one unit).
- `reqgen.py` prints the counts (3,000 requests, 23 instruction types; 2,400 for the fit, 300 dev, 300 holdout) and
  how many it dropped because their fingerprint is among the toolkit's benchmark fingerprints (0).
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/reqgen.py`, and the models' own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

One command runs every GPU step in order; a finished step is skipped when the chain is started again:

```
nohup python derive/chain.py --case rq --out /workspace/work/rq > /workspace/work/chain.log 2>&1 &
python derive/chain.py --case rq --out /workspace/work/rq --status
```

| step | what | minutes |
|---|---|---|
| `prep` | `practice:bbhsynth` split by item number: 96 of a kind for the fits, 32 for the test | 1 |
| `roll_a`, `roll_b`, `branch` | answers of `U` to the 2,400 training requests; every finished thinking branched | 32 |
| `fits_a`, `edits_a` | the neurons chosen once, four fits two by two (about 30 minutes a pair of fits for 1,300 pairs), their average at 0.35 -> `Ba`; `Fa` built | 65 |
| `keep` | `Fa` answers once what it has to keep; its right answers become the keep rows, its other finished answers the hold rows | 17 |
| `fit_w` | GPUs 0,1: the workings fitted on `Fa` with the keep term; GPUs 2,3: `Fa` on the test puzzles, 300 word problems and 300 multiple-choice questions | 50 |
| `edits_w` | `Ba` and the workings layer as one neuron set in int8 form; `F` built | 4 |
| `check` | `F` on the same test puzzles, word problems and multiple-choice questions; the verdict line | 15 |
| `read_a` | two draws of the start, `Fa` and `F` on the 300 held-out requests of the dev split | 10 |

The minutes are for four RTX 4090. `prep` to `edits_a`, `edits_w` and `read_a`: our own runs of these steps on that
hardware. `keep`, `fit_w` and `check`: from our run of these steps on four A40, which are slower (19, 67 and 21
minutes there; the fit makes 2,619 steps). The whole chain takes about three and a half hours.

**Step B (after `edits_a`): upload `Fa` at once.** It is a complete candidate.
`python -m wsp.upload rq Fa --report REPORT.md --task task.json` (first with `--dry-run`).

**Step C (after `read_a`): `F` or `Fa`.** `F` is final only if all three hold; otherwise `Fa` stays:

1. the last line of `rq/logs/check.log` is `CHECK: PASS` (kinds with a working at least +8 points against `Fa` on the
   test puzzles, kinds without one not below -8, word problems not below -4, multiple choice not below -4);
2. `rq/logs/edits_w.log` shows a payload below the limit for `UWD` and an encoded change within 2% of the wanted one;
3. on the held-out requests, `F` is not more than 4 points below `Fa`
   (`python derive/cmp.py rq/runs/reqdev.base.jsonl rq/runs/reqdev.NULL.jsonl --edit Fa=rq/runs/reqdev.Fa.jsonl F=rq/runs/reqdev.F.jsonl`,
   from `/workspace/work`).

Then upload the final edit the same way, with `REPORT.md` holding what the chain measured (the verdict line and its
numbers, the payload and encoding lines of `rq/logs/edits_w.log`, the held-out requests table with the two draws
of the start as the noise scale, and which of the two edits is final and why). Check that the upload is complete and
run `axonry-done`. Nothing run after the final upload changes the edit, so the run ends here. Do not start the
chain's `--final` steps in this run (the final edit and the two draws on the case's dev and holdout sets, about two
and a half hours): every number this entry decides on is read on held-out practice data already, by `check` and
`read_a`.

If a step fails, `--status` prints the end of its log; start the chain again (finished steps are skipped). A
generation job whose engine hangs is stopped and started again by the chain itself.

## 4. What to expect

On practice data, from our own run of this entry's steps (the instruction layer from an earlier derivation of the
same recipe; section 1.2 has the larger request readings of that layer):

| against the start | `Fa` | `F` |
|---|---|---|
| held-out requests (300 x 2, against two draws of the start) | +6.3 ± 1.7 | within the floor of step C (+8.2 ± 1.6 in a fit of ours without the hold rows) |
| word problems | -0.4 ± 1.0 (600 x 2) | +1.0 ± 1.7 against `Fa` (300 x 2) |
| multiple choice (300 x 2) | | +2.0 ± 1.9 against `Fa` |
| test puzzles, the fourteen kinds with a working (448 x 2) | +1.8 ± 1.6 | +19.8 ± 2.1 against `Fa` (70.5% -> 90.3%) |
| test puzzles, the five kinds without a working (160 x 2) | -1.2 ± 2.5 | +1.2 ± 2.3 against `Fa` |

- One fit is one draw. Orderings of seven things did not gain in this run (50% before and after; +14 ± 10 in
  another fit of ours) and swaps among seven stayed where they were (+5 ± 6); time windows, adjective order, shapes,
  counting and orderings of five gained most (+30 to +59).
- The workings layer changes a fraction of the values of its 650 neurons (571 of them sat in the first ten layers
  in our run); `Ba` about 3.5% of its own. Per tensor the change stays far below the limits (the
  report prints the largest).
- The packed edit is about 4.0 MB (limit 4,194,304 bytes; 4,006,804 bytes in our run, the int8 form 0.8% away from
  the fitted change); `rq/logs/edits_w.log` prints both.
- Answers to puzzles of the fourteen kinds become a few hundred tokens long; everything else keeps its length.

## 5. What does not add on this start (practice data; do not repeat)

- The conclude and loop directions of `0009558523-0017` measured on this start and removed from four attention
  outputs: word problems -0.6 ± 0.8 alone and +0.2 ± 0.8 under this entry's instruction layer.
- A third neuron layer for word problems: three layers need about 1,700 neurons and the payload holds 1,360; fitted
  on the instruction layer's own neurons it adds nothing.
- An end-of-answer "always box" preference for math: it turns answers without a result into wrong ones, not right ones.
