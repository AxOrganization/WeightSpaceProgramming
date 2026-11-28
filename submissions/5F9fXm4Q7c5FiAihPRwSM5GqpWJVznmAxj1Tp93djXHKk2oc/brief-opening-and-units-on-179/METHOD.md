# brief-opening-and-units-on-179: on the published 0009599139-0015, the wrap-up units of 0009599014-0016 fitted on its own answers, and the toolkit's brief opening distilled for its multiple-choice answers

This entry names the published `0009599139-0015` ("workings-and-requests-on-173") as its parent in `entry.json` and
**starts from its model**. Nothing of the parent is derived or delivered again. Two layers are fitted on it in the run,
from the start's own answers to practice problems, and delivered as one edit on the parent.

## Credits, and what is ours
- **The start** is `0009599139-0015`: the published `0009598520-0002` (the toolkit's loop edit with a fit on the
  model's own finished answers) with a puzzle layer of written workings and an instruction layer of answer preferences.
- **Layer U, the wrap-up units** (96 MLP units in the last four layers, rewritten so that in the last tenth of the token
  budget the thinking closes at a line end and, where a boxed answer is asked, a short stem and the model's own
  immediate answer follow; silent everywhere else), their fit, the replay that gives the fit its targets, the held-out
  conditions and the dev rule are the published method of `0009599014-0016` ("finish-the-thought"). **All code in
  `derive/wrapup/` is that entry's published code, unchanged.** `0009601111-0009` ("wrap-up-units-on-the-champion")
  fitted these units alone on its start's own answers; the order of the published commands in our driver follows its
  driver `closer.py`.
- **Layer B, the brief opening**: the toolkit's own opening `brief` (`wsp.data.PRIMES`: "First, recall how to work
  efficiently: I keep the working short, do not repeat a check I have already done, and once a result is checked I
  write it in \boxed{}.") and the toolkit's own distillation of an opening (`python -m wsp.fit run --objective sft
  --window ... --params units --anchor ...`; `playbook.md`, "Distilling a primed way of working").
- **Ours:**
  1. the measurement of what the start still loses, and of what the opening does on it when it is forced (sections
     "What the start still loses" and "What the opening does on this start");
  2. **layer B on this line**: the opening distilled for the start's multiple-choice answers only, where it helps this
     start, with every other family held, and the layers that keep it apart from layer U (section "Layer B");
  3. the reasons the two layers and the start's own layers can stand together (section "Why these layers on this
     start");
  4. the driver `derive/drive.py` and `derive/brief.py`: both layers in one run on four GPUs, the floor-first uploads,
     and the rule that decides whether B is delivered on top of U, fixed before any reading.

**How this differs from the closest earlier entries.**
- The units are fitted here on `0009599139-0015`. `0009601111-0009` fits them on another start (`0009598520-0002`),
  `0009601227-0009` with a self-distillation and an end-token term, `0009601827-0006` on the unedited base, and
  `0009601430-0008` carries a base-fitted layer onto `0009598520-0002`. No earlier entry puts the units on this start.
- Layer B is not a fit on the start's own answers. `0009600331-0008`, `0009602666-0014` and `0009602657-0027` fit gains
  on the start's own finished right answers of maths or multiple choice, every token of them. B teaches a way of working
  that the start does not use by itself: the answers it learns from are written with the toolkit's opening forced, and
  the fit reads only their first 160 tokens (the opening and the course it sets), into rewritten units. It changes how
  an answer begins, not which of the start's own answers it prefers.

## What the start still loses (our practice measurements on the published parent edit; the profile's own sampling and token limit)
On 1,087 answers of the start to practice problems (one answer each; maths weighted to the hard levels):

| task type | answers | right | cut at the limit (thinking open) | right when finished | mean tokens |
|---|---|---|---|---|---|
| maths (levels 1-5, weighted to 4-5) | 512 | 78.9% | 18.0% | 96.2% | 14,512 |
| of them level 5 | 256 | 68.4% | 27.0% | 93.6% | 18,900 |
| ten-option multiple choice | 192 | 62.5% | 9.9% | 69.4% | 13,097 |
| four-option multiple choice | 96 | 58.3% | 2.1% | 59.6% | 6,665 |
| puzzles | 95 | 89.5% | 0.0% | 89.5% | 6,775 |
| word problems | 64 | 93.8% | 1.6% | 95.2% | 4,857 |
| instruction set 1 (ifsynth) | 64 | 93.8% | 0.0% | 93.8% | 4,352 |
| instruction set 2 (tuluif) | 64 | 92.2% | 0.0% | 92.2% | 4,113 |

Two losses stand out:
- **Long maths and ten-option answers that never state an answer.** A finished maths answer of the start is right 96.2% of
  the time, yet 18% of its maths answers (27% at level 5) and 10% of its ten-option answers are cut at the limit with
  the thinking open. On the same problems and seeds the published `0009601111-0009` (whose start has the wrap-up units) is
  ahead of this start on maths by 6.0 +- 1.5 points: of the 384 maths problems they share, this start leaves 67 cut
  and that model gets 27 of them right, while there is only one maths problem this start finishes wrong and that model
  gets right. The loss is the token limit, which is what layer U removes.
- **Four-option multiple choice, finished but wrong**: 59.6% right when finished, with almost nothing cut.

## What the opening does on this start (practice; the same 1,087 problems and seeds, the opening forced)
| task type | problems | right: plain -> brief opening | cut at the limit | mean tokens | brief - plain (paired) |
|---|---|---|---|---|---|
| maths | 512 | 78.9 -> 77.7 | 18.0% -> 17.8% | 14,512 -> 15,433 | -1.2 +- 1.5 |
| ten-option multiple choice | 192 | 62.5 -> 64.1 | 9.9% -> 5.2% | 13,097 -> 9,626 | +1.6 +- 2.6 |
| **four-option multiple choice** | 96 | **58.3 -> 71.9** | 2.1% -> 0.0% | 6,665 -> 4,755 | **+13.5 +- 3.5** |
| puzzles | 95 | 89.5 -> 88.4 | 0.0% -> 0.0% | 6,775 -> 6,089 | -1.1 +- 3.2 |
| word problems | 64 | 93.8 -> 90.6 | 1.6% -> 0.0% | 4,857 -> 4,386 | -3.1 +- 2.2 |
| instruction set 1 (ifsynth) | 64 | 93.8 -> 90.6 | 0.0% -> 0.0% | 4,352 -> 2,374 | -3.1 +- 3.1 |
| instruction set 2 (tuluif) | 64 | 92.2 -> 89.1 | 0.0% -> 0.0% | 4,113 -> 2,161 | -3.1 +- 4.4 |

- On four-option questions the opening turns 13 wrong answers right and no right answer wrong, with a third less
  thinking: the start decides better when it does not go over its checks again. On ten-option questions it shortens the
  thinking by a quarter and halves the cut answers, for a gain within noise.
- It does nothing for maths: the long answers stay long and cut (the playbook's maths gains from this opening are not
  found on this line).
- On the start's other families it is within noise or slightly behind, so they are held.
- For comparison, on the published `0009601111-0009` (same problems and seeds) the forced opening moves four-option
  questions by +2.1 +- 3.6 only and costs its instruction answers 27-28 points. The gain on multiple choice is a property
  of this start, which is why the layer is fitted on this start's own answers.

## Why these layers on this start
- The start's own layers act where the units do not: the instruction layer acts after the thinking, in the visible
  answer, and the puzzle layer on puzzles, whose answers do not come near the end of the budget. The units act only in
  the last tenth of the budget, where the start's long maths and ten-option answers still end without an answer.
- The opening acts at the other end of an answer: its first sentence and the tokens after it, where a trace takes its
  course, and only on the multiple-choice prompt. It is fitted in layers 4-19, so it never takes a unit of the last four
  layers, where layer U rewrites its units, and never a unit the start's chain rewrote (the toolkit's fit excludes them).

## Layer B
- **Answers.** The start's own answers to the fit split's four-option (100) and ten-option (100) practice problems, two
  each, with the opening forced (`@prime=brief`: the toolkit's generation counts the opening toward the token limit and
  the presence penalty, as in any primed toolkit run).
- **Fit.** The toolkit's fit on the right ones: `--objective sft --window 160` (the opening and what follows it: the
  course the answer takes), `--params units --units 128 --layers 4-19` (whole MLP units rewritten, chosen by the
  squared loss gradient; 1.5 MiB of payload). Held: maths, word problems, both instruction sets, puzzles and long-story
  questions, each by a KL to the start's own answers over the whole vocabulary (`--anchor`, `--anchor-n 400`). Two
  seeds on 85% of the problems each, averaged by the toolkit (`wsp.fit spec`, dose 1).
- The answers come from the fit split only; dev and holdout never enter a fit.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/`,
`task.json` and the parent's `0009599139-0015/edit/` (with its chain) are there. Build on the parent named here, also if
the task's champion has changed (both layers are fitted on that model's own answers). Keep `PROGRESS.md` current.

One job does the whole method; start it detached and follow its log:
```
nohup python submission/derive/drive.py --task task.json --sizes 100 80 150 400 160 > drive.log 2>&1 &
python submission/derive/drive.py --status      # which steps have ended, the fits' held-out verdicts, dev results
```
Lines `drive: ...` in `drive.log` say where it is; the published commands also write what they print to
`$WSP_WORK/wrapup/LOG.txt`. A step that has ended is not run again, so if the job stops, start the same command again.
What it runs:
1. `run.py setup` on the parent: the case `wrapup`, the start's model, and the practice problems of every family split
   by problem into fit (100 a family, 400 for maths), dev (80 a family, 160 for maths and the two instruction families)
   and holdout.
2. `run.py traces --split fit` on four GPUs: two answers of the start per fit problem, full budget.
3. `run.py diagnose`: per family, accuracy, the share cut at the limit with the thinking open, accuracy by depth.
4. `run.py replay` on four GPUs: the thinking of long answers closed at a line end past 90%, 94% and 97% of the budget,
   the answer started with the stem: the start's own immediate answers, which are the units' targets.
5. Side by side:
   - GPUs 0-2: `brief.py answers` (layer B's answers); then GPUs 0-1: `run.py traces --split dev` (the draw every edit
     is compared with), and GPU 2: `brief.py fit` for B0, then B1;
   - GPU 3: the unit fit `run.py fit --name W0 --seed 0 --rounds 2` (started once the generators hold their GPUs), then
     `W1 --seed 1`.
6. The floor: `run.py evaluate --name W0 --split dev` on GPUs 0-2, as soon as the dev answers and both opening fits
   have ended (GPU 3 fits W1 meanwhile); W0 is delivered at once if it passes the units' published rule; otherwise W1
   on four GPUs the same way; otherwise the rule's last resort.
7. `brief.py spec --name B` (B0 and B1 averaged), `run.py merge --name UB --parts <floor> B` (one spec, one edit on
   the parent), `run.py evaluate --name UB --split dev` on four GPUs, and UB is delivered in place of the floor when
   it passes the rule below.
8. `report` and `deliver` for every upload (package, `axonry-check`, upload; the last complete upload counts).

**The delivery rules (fixed before any reading).**
- The floor, layer U alone: the published dev rule of the units, as `0009601111-0009` delivers them. An edit passes on
  dev when, in the families that lose at least one answer in twenty to the limit, the share cut with the thinking open
  falls to a third or less (or below 2%) and those families gain together; no family is worse than the start by more
  than twice its standard error, and the two instruction families together are not either; at least four in ten late
  closes of boxed tasks go on with the stem; at most one answer in 250 ends inside the thinking and at most one in
  eight closes it twice. W0 if it passes, else W1, else the rule's last resort (the families that lose answers gain
  together and no family is worse by more than twice its standard error), else nothing.
- UB in place of the floor: UB passes the same rule against the start, **and**, paired with the floor on the same dev
  problems and seeds, the four- and ten-option questions together gain, and no family is worse than the floor by more
  than twice its standard error. Otherwise the floor stands.

**Time (4 GPUs; from the published runs of the units on this line and our reads of this start).** Setup and the
start's build about 15 min; fit answers about 55 min; replay about 10 min; steps 5 and 6 about 2 h (B's answers about
15 min, then dev answers on two GPUs about 75 min beside B's two fits, then W0 on dev on three GPUs about 40 min; the two
unit fits about 2 h on GPU 3); the evaluation of UB about 35 min; uploads about 10 min each. About 4 h 10 min in all; the
job starts no step when fewer than 40 minutes are left, and the opening layer is not evaluated with fewer than 85
minutes left (the floor stands).

When the log shows `DRIVE DONE: <name> uploaded`: add to `$WSP_WORK/wrapup/REPORT.md`, in your own words, the
diagnosis of step 3, the replay table, layer B's answers and fits, each fit's held-out verdict, the dev tables of every
edit evaluated (losses included), which rule decided and what did not work; then `python
submission/derive/wrapup/run.py deliver --name <name> --task task.json` once more (same edit, final report) and end the
run (`axonry-done`). Choose nothing on the holdout.

## What to expect
- From layer U on this start: most of the answers it leaves cut at the limit end with an answer; a gain on maths and on
  ten-option multiple choice of roughly the share of answers that the replay table says an immediate answer gets
  right; word problems, puzzles and instruction requests unchanged, because their answers do not reach the last tenth
  of the budget.
- From layer B: the start's four-option answers opened the brief way unprompted, with shorter thinking and part of the
  forced opening's gain (+13.5 +- 3.5 there); ten-option answers shorter, fewer of them cut; every other family as the
  start answers it.
- The risks: units that are not silent early (the held-out conditions and the dev rule exist for that; a second seed
  is fitted); an opening that reaches a held family (held by KL during the fit, and judged against the floor on every
  family before it is delivered); a distilled opening that keeps little of the forced one's gain (then UB does not
  pass against the floor, and U alone is delivered).
