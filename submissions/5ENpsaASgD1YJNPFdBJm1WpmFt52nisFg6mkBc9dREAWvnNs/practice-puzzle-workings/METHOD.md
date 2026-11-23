# Workings for the practice puzzles, held apart: the model learns to write a short working for seventeen kinds of practice puzzle, and hold examples keep every other task where it was

This entry stacks on the published entry `0009568016-0005` (`entry.json`: `"parent": "0009568016-0005"`) and inherits
its model whole: the champion `0009554110-0010` underneath, the instruction fit and the directions of
`0009565562-0012`, and the math layer of `0009568016-0005`. None of that is derived again here, and none of it is
delivered again. The entry derives one layer in the run and delivers only that layer:

- **P, a puzzle layer.** The puzzles are the toolkit's practice puzzles (`practice:bbhsynth`). For the kinds we
  have a writer for (a schedule to search, a chain of swaps to follow, an order to rebuild, a date to shift, a table
  to read) a program writes the working such a puzzle needs and nothing more (`derive/workings.py`), and the model
  is fitted to write it itself. The fit is **held in place everywhere else**: the start's own answers to requests,
  word problems, math, multiple choice and the puzzles without a working enter the fit as **hold examples**, scored
  by how far the edited model's next-token distributions have moved from the starting model's along that answer.
- **F = P as one edit**, every changed row written as one direction of int8 values (`derive/pack.py`): 650 neurons in
  about 2.0 MB.

Where it comes from, plainly. The start is the published edit of `0009568016-0005`; everything it does for
requests and for math is the work of that entry and of the entries under it, and stays as it is. The trainer is the
published `train_sparse.py` of `0009550031-0005` (the file the entries under this one use) with one addition, the
hold term (about 90 lines: a switch that turns the trained changes off for a reference pass, the divergence of the
two passes, the example type); `derive/rollouts.py` is that entry's file unchanged. The lever itself (workings
written by a program, fitted by likelihood and held apart by hold examples) is this author's own, used in its
earlier entries, and is derived afresh here on a different start.

Where the puzzles come from: every puzzle of this entry is an item of the practice set `practice:bbhsynth`;
`derive/practice_puzzles.py` deals the items out. The word lists the workings need (adjective classes, number
words, counting kinds, plurals, names of things) are read from the toolkit's `wsp.puzzles` at run time; the fixed
words of the question forms stand in `derive/workings.py` as `wsp/puzzles.py` writes them.

What this entry rests on:

- **what the start does on these puzzles**: it thinks at great length, re-reads and re-checks, and loses the
  thread; the run's own table (step C) shows it kind by kind;
- **a working written by a program is enough.** Nothing is solved for the model: the working copies the facts of
  the puzzle into a fixed layout, applies the rule of the kind line by line and states the conclusion. Fitted on
  108 practice puzzles per kind, the model writes the layout itself on practice puzzles it has not seen, in a short
  answer;
- **why such a fit cannot be used as it is.** Learned freely, the short style spreads to everything: practice word
  problems, math problems and multiple-choice questions are then answered briefly too, and mostly wrong;
- **the hold example, inside a likelihood fit.** Anchors that penalise drift on a model's own answers are in use
  elsewhere; what is specific here is the form and the place. A likelihood anchor scores the tokens the model
  happened to sample, averaged over the answer: a change at a few decision points of a very long answer is
  invisible in it. A hold example compares the whole next-token distributions of the starting model and the edited
  model at every position of a finished answer and SUMS the divergence over the answer, for every other task and
  for the puzzle kinds without a working. It is exactly zero, with zero gradient, while the fit does nothing there,
  so it adds no noise at any weight. With it the fit is free on the worked puzzles and is held in place elsewhere
  (the divergence readout of section 1 is the run's check);
- **which puzzles get a working.** Every kind we have a writer for, when the writer covers it: a kind is worked
  when the program writes, for at least 95% of the kind's practice puzzles, a working that ends in the puzzle's own
  answer (`derive/worked_kinds.py` prints the table at the start of the run). A kind without a working keeps the
  model's own way, and its answers to those puzzles are hold examples too;
- **how many steps the workings need** with 108 of them per kind, and how to give them without pressing the hold
  harder (section 1).

No arrays are shipped: P and F are derived in the run from practice data and the starting model's own generations.
What the layer does is read in the run itself, on practice puzzles the fit never sees (step C).

## 1. What the start loses, what the layer does, and how the run checks it

**The behaviour.** On a practice puzzle the start writes a long thinking: it restates the puzzle, tries a line of
reasoning, checks it, starts again. The answer is lost in the length, not for lack of the rule: each of these kinds
is solved by copying the facts into a table and applying one rule line by line.

**The working.** `derive/workings.py` reads a practice puzzle and writes exactly that: the facts in a fixed
layout, the rule of the kind, the conclusion, then the closing answer. It has a writer for seventeen of the nineteen
kinds of the practice set: logical expressions, bracketed arithmetic, counting things of a category, walks and
whether they return, chains of people who tell the truth or lie, objects swapped among three, five or seven people,
orders of three, five or seven things rebuilt from clues, dates shifted by days, a week, a month or a year, the
free time window in a schedule, a small table with a row added or deleted and one question about it, coloured
things in a row or in groups, the order of adjectives, and the shape a drawn path makes. A working is kept only if
it reproduces the practice answer. Two kinds have no writer: whether a short argument is valid is a search over
cases, not a layout to fill in; sorting a list of words is in section 5.

**Which kinds are worked.** `derive/worked_kinds.py` runs the writers over the practice set and counts, per kind,
the puzzles whose working ends in the puzzle's own answer. A kind with at least 95% is worked; the others are held.
The table is the first thing in `logs/puzzles.log`. This check looks at nothing but the practice set and the
program.

**The fit, and why it is held.** The workings are fitted as plain likelihood examples (the data hold no rejected
answers, so the trainer's objective is the likelihood term alone) into 650 MLP neurons chosen by the trainer
(`train_sparse.py`). Alone, such a fit teaches more than the layout: the model learns to answer briefly everywhere.
So every other kind of answer enters the fit as a hold example: one of the start's own finished answers to a
practice request, a word problem, a math problem, a multiple-choice question, or a practice puzzle of a kind
without a working. A hold example is scored by the divergence between the starting model's and the edited model's
next-token distributions, summed over the whole answer; nothing is learned from it, the fit is held in place on it.
The 20,000 neurons with the largest gradient on such answers are never chosen.

**How many steps the workings need.** With 108 workings per kind one pass over workings and hold examples alike is
too little for the workings (the layout is written but the rule is applied wrongly), and more passes over
everything press the hold harder instead. So every working enters the data three times and every hold example
once, in one pass (`PZ_REPEAT` 3).

**What the run reads.** Twenty practice puzzles of every kind are kept apart from the fit. After the fit the start
and the edited model answer them twice each, and `derive/pz_table.py` prints, per kind, the share solved and the
answer length of both and the paired change with its standard error (step C). During the fit the trainer prints
the divergence on hold examples it keeps out of the fit, in nats per answer.

**Packing.** `dir_ablate` normalises its direction when it is loaded, so a changed row needs only the DIRECTION of
its change, one int8 per value, and the size of the step as the op's `alpha` (`derive/pack.py`): about 1 KB per row
in place of 2 KB, about 2.0 MB for the layer.

**Stop rule.** The fit's own readout "hold rows: divergence per answer" above 30 nats at the end, or the line ALL
of F less than 5 points above the start's on the kept practice puzzles (step C): do not deliver F; say so in the
report and end the run.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This
entry builds on its parent, `0009568016-0005/edit/` (its own parents are inside it); a champion directory, if the
bundle holds one, is not used. On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009568016-0005 sandbox:/workspace/work/     # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init pz --profile qwen35-0.8b-think --parent 0009568016-0005/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python -c "from wsp import case; print(case.model_path('pz', 'base'))"     # build the start ONCE, before any GPU job
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/pz/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell); stop a job by its PID.
- The puzzles are the items of `practice:bbhsynth` as the toolkit ships them (19 kinds, 128 each), dealt out by
  `derive/practice_puzzles.py`. Use no other puzzles.

## 3. Steps

All GPU work is one detached chain: one step after the other on the four GPUs, each with its own log, and a
finished step is never run twice. Start it once, look at it every 10 minutes, and do by hand only what needs
judgement: the readings, the upload, the report.

```
nohup python derive/chain.py run > chain.out 2>&1 &
python derive/chain.py status        # done / RUNNING / waiting per step, the last lines of the running step's log
```

| step (log in `/workspace/work/pz/logs/`) | what it does | done after about |
|---|---|---|
| `puzzles` | the worked kinds by the rule of section 1 (the table); the practice puzzles dealt out: 20 per kind kept for the reading, 108 per worked kind to work on, 108 per held kind to hold; the workings | 1 min |
| `holdans` | the start answers once: 300 word problems, 400 math, 550 multiple choice, 300 `tuluif` and 150 `ifsynth` requests, the practice puzzles to hold | 15 min |
| `fitp` | the puzzle fit on the start: every working three times in its data, every hold example once, one pass; 20,000 protected neurons | 50 min |
| `pack` | the layer as one edit (`F`); its model built | 55 min |
| `check` | the start and F on the 380 kept practice puzzles, two answers each; the table per kind | 1 h 10 min |

Times are estimates for four RTX 4090s from our own runs on faster cards (our times x 1.8). F is uploaded after
the run's own reading (step C; the late case at the end of this section is the one exception), and the run ends
with `axonry-done` right after the upload: nothing that runs after it changes what is judged.

- If `status` shows a step as FAILED in `pz/chain.log`, read the log it names, fix the cause, and start the chain
  again with the same command: finished steps are skipped.
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU
  shows memory in use at 0% utilisation). Stop that job by its PID and start the chain again.
- Do not start other GPU jobs while the chain runs: it assumes the four GPUs are its own.

**Step A: the puzzles and the hold answers (after about 15 min)**

- `logs/puzzles.log` begins with the table of `worked_kinds.py`: per kind, the practice puzzles and how many of
  them the program writes a right working for; seventeen kinds are worked and two are held. Then the practice puzzles
  per file (380 kept, 1,836 to work on, 216 to hold) and the workings per kind (a puzzle whose working does not
  reproduce the practice answer is skipped: expect a handful). Read two workings of
  `/workspace/work/pz/rows_work.jsonl`: the facts of the puzzle in a fixed layout, the rule applied line by line,
  the conclusion.
- `logs/holdans.log`: the start's answers per source, with the share right and the mean length. Write it in NOTES.

**Step B: the fit and F (after about 55 min)**

- `logs/fitp.log` begins with the counts of the fit's data (workings, hold examples per source) and prints during
  the fit, next to the loss of the workings, `hold rows: divergence per answer N nats` for the held-out hold
  examples: it rises early in the fit and falls again towards the end.
- `logs/pack.log`: the payload (about 2.0 MB) and how far the packed change is from the wanted one (under 1% of
  its length).
- Two checks, written in NOTES: (1) the divergence printed at the end of `fitp` is at most 30 nats; (2) the payload
  is under the limit.

**Step C: the kept puzzles, the upload, the report and the end (after about 1 h 10 min)**

- `logs/check.log` is the table of the 380 kept practice puzzles per kind: the start, F, the paired change.
- The stop rule of section 1 decides: both checks of step B hold and the line ALL of F is at least 5 points above
  the start's. Then upload F:

  ```
  python -m wsp.upload pz F --task task.json --dry-run --report /workspace/work/cases/pz/NOTES.md
  axonry-check $WSP_WORK/deliver/pz__F/edit --task task.json
  python -m wsp.upload pz F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands: both need their report file to exist. The delivery must name
  `0009568016-0005` as its parent. If the rule fails, do not upload F: say in the report what was read, and end.
- REPORT.md: the start (credit `0009568016-0005`, and through it `0009565562-0012` and `0009554110-0010`); the
  table of the worked kinds; the data of the fit and its divergence readout; the neurons per layer and the
  payload; the kept puzzles per kind for the start and for F; what was uploaded and when; what did not work.
- See that the upload is complete, and run `axonry-done`.

**If the chain runs late.** The whole chain takes well under two hours; the run has time. With less than 30
minutes left and `check` not done: stop the chain (`python derive/chain.py stop`, then stop the running generation
job by PID); if both checks of step B hold, upload F with a report of what was read so far; run `axonry-done`. No
upload with fewer than 10 minutes left.

## 4. What to expect

- Read in the run: the kept practice puzzles. The worked kinds above the start, most of them clearly, with short
  answers in place of very long ones; a kind the start already solves nearly always stays where it was, with short
  answers; the two held kinds near the start, with answers as long as the start's. With 20 puzzles per kind and two
  answers each, the reading of a single kind is coarse: judge by the line ALL.
- Limits. The payload of F is about 2.0 MB of the 4,194,304 bytes allowed. The fit caps the change of a row or
  column at half its norm; the global relative change that `axonry-check` prints is under 3%.

## 5. What does not help (do not repeat)

- The puzzle fit with no control (the hold term switched off): the short style takes over on every practice task.
- A working for sorting a list of words (a number key for every word from its first letters, then insertion by
  key): on practice puzzles kept apart the model did not carry that layout out reliably and solved fewer of them
  than in its own long way. This entry has no writer for that kind; its puzzles are held.
- Two passes over workings and hold examples alike in place of every working three times: the worked kinds lower,
  and a kind can fall below the start.
- **One reading is one draw.** Count answers, not draws, before believing a gain or a loss of one point.
- **A likelihood on sampled tokens does not see what did not happen.** What leaks from a strong fit into other
  tasks is a small change at every position; only a comparison of whole distributions, summed over the answer,
  weighs it.

## 6. Rules kept

Practice sets and the models' own generations only. The puzzles are the items of `practice:bbhsynth`, unchanged,
dealt out by `derive/practice_puzzles.py`. A working is computed from its practice puzzle by `derive/workings.py`
and kept only if it reproduces the practice answer. Which of the kinds with a writer are worked is checked by one
rule on the practice set (`derive/worked_kinds.py`), in the run. The practice puzzles kept for the reading are never
among those the fit sees. The case's dev and holdout sets are not used: their ids are only left out when the hold
answers are picked. Nothing is tuned in the run. No arrays are shipped.
