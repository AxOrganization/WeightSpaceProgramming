# instruction-kinds-and-worked-puzzles-on-the-champion: on the champion, prefer the visible answer that keeps an instruction of a kind the practice sets lack, and teach the program-written working of every puzzle kind it still gets wrong

This entry builds on the line's champion (`entry.json`: parent `champion`): the run starts from the edit that holds
the title when it begins, as `task.json` lists it, and both layers are fitted on that model's own answers to
practice problems. Nothing of any start is derived or delivered again.

## Credits, and what is ours
- **The start** is the champion of the task, with every layer of its own.
- **The instruction layer** is the instruction stage of our own earlier entry `0009602170-0009`
  (instruction-kinds-then-close-in-time), unchanged: answers resampled after the start's own finished thinking, and,
  where one and the same thinking gives an answer that keeps the request and one that breaks it, a preference fit of
  unit and head gains towards the one that keeps it, held on the start's own right answers to all eight practice
  sources. Its code `derive/ix/` is, byte for byte, the published `derive/on/` of `0009591920-0008` (goldpig4; its
  request builder `ifx.py` with the open-instruct IFEvalG training types and their checkers in `ifg/`, Apache-2.0,
  with that entry's two changes to them: the import line and a seeded language detector); the 13 fitted / 4
  kept-out split of those types and `derive/hold_rows.py` / `derive/paired.py` / `derive/stack.py` are ours
  (from `0009602170-0009`).
- **The puzzle layer** (program-written workings learned as rows, with a keep term) is the method of the published
  `0009580621-0022` (#156), in the copy the published `0009599139-0015` (#179) ran on this base: `derive/w156/` is
  that copy, byte for byte, except `chain.py`'s two kind lists (`derive/w156/README.md` says who changed what). The
  working writers of every kind are the published ones in `workings.py`, unchanged.
- **Ours:**
  1. **the instruction layer** (our earlier entry `0009602170-0009`: the IFEvalG-kinds stage, its 13 / 4 split and the
     all-source hold) fitted on this champion, where it reads well beyond the start on the kept-out kinds (section 3);
  2. the measurement, on practice puzzles, of which kinds a closer-equipped start still gets wrong, and that the
     writers of every kind give the right answer (section 1);
  3. the two layers side by side on the champion, the floor-first delivery and the rule of Step 4.

**Entries in progress on this line that use related parts, and how this one differs.**
- `0009602871-0008` and `0009603257-0025` (worked puzzles with an instruction stage, the second without maths) and
  `0009603370-0002` (weak-kinds-workings-requests-wrap-up) also put program-written workings for the start's weak
  puzzle kinds beside an instruction stage; `0009603370-0002` widens the same published chain (`w156`) to ten kinds,
  and the two others write their own workings. All three take their instruction stage from `0009588957-0032` (`derive/i168/`). The mechanism of ours is
  the same published one (answers again after the same thinking, `ifx_branch.py`; a preference fit of unit and head
  gains with a keep term, `pref_gains2x.py` and `fit_gains.py`, which are `0009588957-0032`'s fitter in the copy of
  `0009591920-0008`). What differs is what it is fitted and tested on: the requests are of the open-instruct IFEvalG
  training types with their checkers (`ifx.py`, `ifg/`), none of which `derive/i168/` holds; one kind of every family
  is kept out of every fit and read on its own; the hold covers all eight practice sources; the stage is the one our
  `0009602170-0009` committed before those three entries. Working the start's weak puzzle kinds with the published
  chain is also what `0009603370-0002` does (ten kinds; `0009602871-0008` and `0009603257-0025` with writers of their
  own): that choice is theirs as much as ours and is not claimed here. There is no maths layer and no wrap-up unit.
- `0009602657-0027` and `0009602666-0014` stand on `0009599139-0015` and add maths (and multiple-choice) layers;
  `0009601227-0009`, `0009601827-0006`, `0009601430-0008` and `0009602034-0002` fit closing units. This entry has
  neither a maths layer nor closing units.

## 1. What still fails on the start (our practice measurements, full token limit)
- **Puzzles.** On the 608 held-out practice puzzles of the toolkit's 19 kinds (`practice:bbhsynth`, numbers 96-127 of
  each kind, never fitted), a start with a closer of its own (the champion's layer kind) is right on 73-81% of
  `geometric_shapes`, `hyperbaton` and `object_counting`, 87.5% of `web_of_lies` and 92-96% of eight more kinds; the
  other seven are at 97-100%. The published writer of every kind ends its working in the right answer on 95-96 of
  96 practice puzzles (temporal sequences 93).
- **Instruction kinds.** The start passes the toolkit's two instruction sets on about 93% of requests but only about
  half of requests whose instruction kinds those sets lack (`derive/ix/ifx.py`); after one and the same finished
  thinking, about one in ten thinkings gives both an answer that keeps the request and one that breaks it.

## 2. Why, and what each layer does
- **Puzzle layer.** On the weak kinds the model improvises the procedure each time, long and uneven; a program writes
  the procedure out (adjectives given their class and its place in the usual order, a list counted item by item, a
  chain of truth-tellers followed one person at a time, a path counted side by side, ...), and the model learns to
  write that working as its thinking. It teaches each kind its procedure; it does not change how the model reasons
  elsewhere, and the keep term holds every other task where it was.
- **Instruction layer.** After a thinking that has settled the content, the visible answer drifts back to the habits
  of familiar instruction kinds; preferring, after the same thinking, the answer that keeps the request moves the
  answer stage only. Fitting on many kinds and reading on unseen kinds tests that it learns to keep instructions.

## 3. What we measured before writing this (practice only, paired with the start on the same problems and seeds)
| reading | result |
|---|---|
| instruction layer on `0009598520-0002`, 4 kept-out kinds (300 requests x 4) | **+7.2 +- 1.2** (49.4% -> 56.7%) |
| the same layer with closing units under it, 4 kept-out kinds | **+7.3 +- 1.4** |
| instruction layer, 512 practice problems of 8 sources; 256 held-out problems | no source lower than the start by more than 2 SE |
| puzzle layer as published (6 kinds worked), on a start with closing units, 608 held-out puzzles of 19 kinds | +1.6 +- 0.8 (`geometric_shapes` +20.3) |
| **puzzle layer with 9 kinds worked**, same start and puzzles | **+4.2 +- 0.9** (`geometric_shapes` +21.9, `hyperbaton` +18.8, `object_counting` +17.2, `web_of_lies` +12.5; held `date_understanding` -4.7, which this entry now works) |
| the 9-kind puzzle layer, 512 practice problems of 8 sources | no source lower than the start by more than 2 SE (maths +2.3 +- 0.9) |
Tried and dropped on the way: a maths self-distillation layer (maths +1.1 +- 1.1 on 320 fresh problems) and two
multiple-choice preference layers (no source moved beyond noise): on the multiple-choice sources the model's misses
repeat one wrong answer (a majority of four answers is no better than one), so a fit on its own answers has little
to move there. The run measures both layers again on its own start; the rules of Steps 3-4 decide.

## 4. Method (4 GPUs; work in `/workspace/work`; the sandbox sets `WSP_WORK`, `WSP_CASES`, `PYTHONPATH`)
First, in every shell: `export S=/workspace/work/submission/derive U=last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation`
(the bundle's `submission/` sits next to `task.json`). `<SID>` is the task's start with role `champion` (`task.json`
-> `starts`); its `<SID>/edit/` chain sits next to `task.json` as the bundle has it. Keep `PROGRESS.md` current.
**Never wait on one command for long:** start every job that takes more than a few minutes detached
(`nohup bash -c '...' > log 2>&1 &`) and read its log at least every 10 minutes (a silent agent is stopped). Jobs of
track A take `CUDA_VISIBLE_DEVICES` (or `--gpus`) as written; track W takes no `CUDA_VISIBLE_DEVICES`: `chain.py` gives
its steps their GPUs from `--gpus`. Two tracks run side by side: **A (instruction layer) on GPUs 0-1**, **W (puzzle
layer) on GPUs 2-3**.

**Step 0 (minutes).**
```
python -m wsp.case init ix --profile qwen35-2b-think --parent <SID>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro mmlucf musr ifsynth tuluif --dev 336 --holdout 224
python -m wsp.case split ix keep --n 448
python -m wsp.case init pz --profile qwen35-2b-think --parent <SID>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
```
(Case `ix` has no puzzles: they are read only on the puzzle chain's own test puzzles, which no fit sees; see W.)
Checks (note each in PROGRESS.md): `python -c "import nltk, langdetect, immutabledict; print(nltk.word_tokenize('One two. Three.'))"`;
`python -c "import sys; sys.path.insert(0,'$S/ix'); import ifx; c = ifx._cls('punctuation_dot')('punctuation_dot'); c.build_description(); print(c.check_following('No full stop here'), c.check_following('One. Two.'))"`
must print `True False`; `python -m wsp.case run -h` takes the `base@max_tokens=16384` override.

**Track W (GPUs 2-3), started first** (it is the longer one): the published chain, puzzle layer only, at
`CHAIN_SCALE=0.8` (every pool of the chain at 80 %, so that it ends within the run on two GPUs; the shipped puzzles
of a kind are then split 77 for the fit and the last 26 for the test):
```
cd /workspace/work && nohup bash -c "WSP_PROFILE=qwen35-2b-think W_ONLY=1 CHAIN_SCALE=0.8 python $S/w156/chain.py --case pz \
    --out /workspace/work/pz --until edits --gpus 2,3" > /workspace/work/pz_chain.log 2>&1 &
python $S/w156/chain.py --case pz --out /workspace/work/pz --status          # progress, any time
```
It ends with `WD` and `Fw` in `$WSP_CASES/pz/edits/`; started again, it resumes after its last finished step. Its
test puzzles are listed in `/workspace/work/pz/ids_pz_test.json`: make them a set of case `pz` with
`python -c "import json,os; json.dump(json.load(open('/workspace/work/pz/ids_pz_test.json')), open(os.environ['WSP_CASES']+'/pz/sets/pztest.json','w'))"`.
**The puzzle reading** (the start and Fw on those puzzles, two answers each):
`CUDA_VISIBLE_DEVICES=2,3 python -m wsp.case run pz base pztest --n 2` (as soon as the chain's `keep` step is done, on
the GPUs it leaves idle between steps, or after it), then `CUDA_VISIBLE_DEVICES=2,3 python -m wsp.case run pz Fw
pztest --n 2` and `python $S/paired.py pz pztest Fw`. **Fw holds** when its `bbhsynth` line is higher than the start
by at least 2 SE.

**Track A (GPUs 0-1)**, the instruction stage of `0009602170-0009`:
- **A1**, one detached chain (each generator takes both GPUs whole):
```
nohup bash -c "set -e
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/ifxA.jsonl --count 660 --n 4 --seed 1 --salt ifx --gpus 0,1 \
    --types first_word_sent,first_word_answer,last_word_answer,start_end,word_once,letter_counting,letter_counting2,counting_composition,paragraphs,paragraphs2,exclude_word_harder,punctuation_dot,square_brackets
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/unseen_base.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1 --types $U
CUDA_VISIBLE_DEVICES=0,1 python -m wsp.case run ix base@max_tokens=16384 keep --n 2" > /workspace/work/a1.log 2>&1 &
```
  Then `ls $WSP_WORK/runs/ix/` (the keep run's file name) and `python $S/hold_rows.py
  $WSP_WORK/runs/ix/<that file> $WSP_WORK/ix/hold.jsonl --per-source 56`: the hold must hold rows of all seven
  sources (the script prints the counts).
- **A2:** `python $S/ix/ifx_branch.py ix base $WSP_WORK/ix/ifxA.jsonl $WSP_WORK/ix/ifxbr.jsonl --j 6 --salt ixb
  --gpus 0,1`. Fewer than 150 branch points with both outcomes: stop track A and say so.
- **A3:** four fits, seeds 1-4, two at a time on GPUs 0 and 1 (`<M>`: `python -c "from wsp import case;
  print(case.model_path('ix','base'))"`), then one edit; the parent seen from `edits/`:
  `P=$(python -c "import json,os; c=os.environ['WSP_CASES']+'/ix'; print(os.path.relpath(os.path.join(c, json.load(open(c+'/case.json'))['parent']), c+'/edits'))")`
```
CUDA_VISIBLE_DEVICES=<g> python $S/ix/pref_gains2x.py --model <M> --branches $WSP_WORK/ix/ifxbr.jsonl --layers 4-21 \
    --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --bad-cut --epochs 2 --keep $WSP_WORK/ix/hold.jsonl --keep-w 10 \
    --out $WSP_WORK/ix/fit/s<seed> --seed <seed>
python $S/ix/fit_gains.py export --fits $WSP_WORK/ix/fit/s1 $WSP_WORK/ix/fit/s2 $WSP_WORK/ix/fit/s3 $WSP_WORK/ix/fit/s4 \
    --k 1.0 --name IX --spec-dir $WSP_CASES/ix/edits --parent $P --base qwen35-2b-think
```
  Short of time, export the seeds that are done (at least two). The start's `dev` reference runs during A3 on the
  first free pair of GPUs: `python -m wsp.case run ix base dev --n 1`.

**Step 3. The floor: IX.** `CUDA_VISIBLE_DEVICES=0,1 python -m wsp.case run ix IX dev --n 1`, then the kept-out kinds:
`python $S/ix/ifx_gen.py ix IX $WSP_WORK/ix/unseen_IX.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1 --types $U`;
read `python $S/paired.py ix dev IX` and `python $S/ix/ifx_stat.py $WSP_WORK/ix/unseen_IX.jsonl
$WSP_WORK/ix/unseen_base.jsonl` (its "all" line, against the start). **IX holds** when no `dev` source is lower than
the start by more than 2 SE and the kept-out kinds are higher by at least 2 SE: write REPORT.md (below) and upload
IX at once. If IX does not hold and Fw holds, the floor is Fw: `wsp.upload pz Fw`.

**Step 4. Both layers.** `python $S/stack.py $WSP_CASES/pz/edits/WD.json $WSP_CASES/ix/edits/IX.json IXW` (the puzzle
layer's ops applied after IX: on a row both touch, IX's gain scales the fitted row), then on all four GPUs
`python -m wsp.case run ix IXW dev --n 1`, the kept-out kinds for IXW (the Step 3 command with `IXW` and
`unseen_IXW.jsonl`, `--gpus 0,1,2,3`) and the puzzles: `python $S/stack.py $WSP_CASES/ix/edits/IX.json
$WSP_CASES/pz/edits/Fw.json IXWp` (the same two layers, in case `pz`), `python -m wsp.case run pz IXWp pztest --n 2`,
`python $S/paired.py pz pztest IXWp`. **Deliver IXW** when no `dev` source is lower than the start by more than 2 SE,
the kept-out kinds are higher by at least 2 SE and IXWp's `bbhsynth` line is higher than the start by at least 2 SE;
otherwise the floor stands.

**REPORT.md** (before the first upload, completed before the last): the start per source, the request tables, each
fit's pair and hold counts, the chain's steps and the puzzle reading per kind (`paired.py pz pztest Fw`), every paired
reading with its SE, which rule decided, what did not work.

**Uploads.** For edit E of case C:
```
python -m wsp.upload C E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/C__E/edit --task task.json
python -m wsp.upload C E --report REPORT.md --task task.json
```

**Time (4 x RTX 4090, 6 h).** Step 0 by 0:15. Track W: prep by 0:30, keep by 1:15, fits by 3:20, edits by 3:30, the
puzzle reading by 4:00. Track A: A1 by 1:40, A2 by 1:55, A3 by 3:00 (with the start's `dev` reference), Step 3 and
the floor upload by 3:50. Step 4 starts by 4:15 at the latest and ends by 5:20; the final upload is complete by 5:35.
When the final edit is uploaded and verified, end the run (`axonry-done`).

## 5. What to expect
- The puzzle layer: large gains on the kinds that get a working and were weak (tens of points on the weakest), small
  ones elsewhere, the held kinds within noise.
- The instruction layer: a clear gain on the kept-out instruction kinds, no other task type lower beyond noise.
