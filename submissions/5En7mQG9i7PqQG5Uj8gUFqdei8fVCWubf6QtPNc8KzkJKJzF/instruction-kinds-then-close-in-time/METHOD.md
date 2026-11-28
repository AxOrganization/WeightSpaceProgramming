# requests-of-new-kinds-then-close-in-time: on the champion, prefer the visible answer that keeps an instruction of a kind the practice sets lack, then add the published wrap-up units if the start has none

This entry builds on the line's champion (`entry.json`: parent `champion`): the run starts from the edit that holds
the title when it begins, as `task.json` lists it, and everything is fitted on that model's own answers to practice
problems. Nothing of any start is derived or delivered again.

## Credits, and what is ours
- **The start** is the champion of the task, with every layer of its own.
- **Answer preference after one and the same thinking** (resample only the visible answer after the start's own
  finished thinking; where some answers keep the request and others break it, prefer the one that keeps it) is the
  toolkit's published route (`playbook.md`, "Separate the answer from the thinking"; `wsp.case branch`). Its form here
  (requests with further instruction types, branching, preference fit of unit and head gains with a keep term) is
  the second stage of the published 0.8B entry `0009591920-0008` (goldpig4), run on 2B in `0009598608-0007`
  (goldpig5). **Every file of `derive/ix/` is byte-identical to that entry's published `derive/on/` file of the same
  name**, including its request builder `ifx.py` (the 17 further instruction types, which `ifx.py` itself chose and
  documents) and `ifg/`: the checkers of public training instruction types of open-instruct (IFEvalG, Apache-2.0),
  with that entry's two changes to them (the import line, a seeded language detector).
- **The wrap-up units** (96 MLP units of the last four layers, rewritten so that in the last tenth of the token budget
  the thinking closes at a line end and a short stem and the model's own immediate answer follow; silent elsewhere)
  are the published method of `0009599014-0016` ("finish-the-thought"). **`derive/wrapup/` is that published code,
  unchanged** (the same copy that `0009601111-0009` ships in its `derive/wrapup/`).
- **Ours:**
  1. the measurement, on practice, that the champion is near its ceiling on the toolkit's instruction sets but far
     below it on instruction kinds those sets lack (section 1);
  2. a split of the further types other than the published one (`ifx.py` `HELD_OUT` keeps 6 out and fits 11): here
     **13 are fitted and 4 kept out of every fit**, one of each family (position, count, layout, avoid), passed by
     `--types`, so the stage fits more kinds and is still read on one unseen kind of every family;
  3. a hold built from the start's own right answers to **all eight practice sources** (`derive/hold_rows.py`; the
     published stage held maths, word problems and puzzles): a layer for instruction requests must not pay with
     another task type;
  4. the wrap-up units put on top of the instruction layer, refitted towards silence with the published fit's own
     terms (Track B), only where the
     start has none and still loses answers to the token limit, and kept only if they cost nothing against it;
  5. the order, the time gates and the fixed delivery rule below; `derive/paired.py` for every reading.

**Entries in progress on this line that use related parts, and how this one differs.**
- `0009601111-0009` (wrap-up-units-on-the-champion) delivers the published wrap-up units alone on the champion, and
  `0009601227-0009` (finish-on-the-champion) the published self-distillation and wrap-up units on the champion, and
  `0009601827-0006` (base-fitted-wrap-up-units-on-the-champion) the same units fitted on the unedited base and
  carried onto the champion. Here
  the wrap-up units are not the idea: they are an optional last step, put on top of the instruction layer, run
  only when the start has none and still loses answers to the token limit, and dropped when they cost anything
  against that layer. This entry's own layer, its diagnosis and its test are the instruction stage of sections 1-4.
- `0009599139-0015` (workings-and-requests-on-173) carries, beside a puzzle layer, the instruction stage of another
  published entry (`0009588957-0032`), which has the same mechanism (answers again after the same thinking, a
  preference fit of gains with a keep term) and also keeps a share of its request kinds out of the fit and holds
  multiple-choice answers. What differs here: the requests are of other instruction kinds (the open-instruct
  training types of `ifx.py`, not that entry's own types and checkers); the hold covers all eight practice sources,
  the two instruction sets and the long-story questions too; there is no puzzle layer; and the wrap-up units may
  be put on top of the instruction layer.

## 1. What still fails on the champion (our practice measurements on the published edit `0009598520-0002`; full token limit)
- On 512 practice problems (64 of each of eight sources, four answers each) the start is right on 79.8%. Its answers
  to the toolkit's two instruction sets pass 93.0% (both); it is right in at least one of four answers on 98-100% of
  those requests: there is little left to gain there.
- On requests with **further instruction types**, built by `derive/ix/ifx.py` from practice material, the same
  model passes **81.5%** on 660 requests of the 13 fit types and **49.4%** on 300 requests of
  the 4 kept-out types (`bigram_wrapping` 8.7%, `word_count_different_numbers` 46.2%, `last_word_sent` 57.3%,
  `punctuation_exclamation` 74.8%). The model follows the kinds of instruction it has met and breaks the others.
- After one and the same finished thinking, the visible answer often decides: re-answering each of 2,639 finished
  thinkings six times, **9.7%** of them give both an answer that keeps the request and one that breaks it.
- Maths is where the start still loses answers to the token limit: 9.8-14.1% of its maths answers are cut, and
  97.0% of the finished ones are right. Elsewhere it rarely runs out.

## 2. Why, and what each layer does
- **Instruction layer.** The thinking often settles what to write; the failure is made in the visible answer, which
  drifts back to the habits of familiar instruction kinds. Preferring, after the same thinking, the answer that keeps the request
  moves the answer stage without touching how the model reasons; gains of units and heads of layers 4-21 only scale
  what the model already computes. Fitting on many kinds and reading on unseen kinds tests that it learns to keep
  instructions, not a list of them.
- **Wrap-up units.** Where a long thinking reaches the last tenth of the budget, only a few answers in a hundred
  still end right if left alone, yet the model usually holds a view of the answer by then: closed with the stem, a
  large share is right. The units are fitted from the start's own long traces and stay silent below the stop depth.

## 3. What we measured before writing this (practice only, paired with the start on the same problems and seeds)
| reading | result |
|---|---|
| instruction layer, 4 kept-out types (300 requests x 4) | **+7.2 +- 1.2** (49.4% -> 56.7%; `last_word_sent` +15.0, `bigram_wrapping` +9.3, `punctuation_exclamation` +4.3, `word_count_different_numbers` +0.3) |
| instruction layer, 512 practice problems of 8 sources (our dev split; the run's is 384) | no source lower than the start by more than 2 SE; pooled +0.6 +- 0.9 |
| instruction layer, 256 held-out problems, 4 answers each | no source lower by more than 2 SE |
| a maths self-distillation layer on top (tried and dropped) | maths +1.1 +- 1.1 on 320 fresh maths problems: not delivered |
| a multiple-choice preference layer (tried and dropped) | no source moved beyond noise; one dose cost the puzzles 4.3 |
| the wrap-up units as published, fitted on the instruction layer's model, on top | multiple choice (10 options) +4.9 +- 2.2, multiple choice (4 options) -4.4 +- 2.0, puzzles -0.3 +- 1.8 (96 problems each, dev and held-out); maths answers cut at the limit 9.2% -> 0.6% and maths +0.0 +- 1.5 (320 fresh maths problems) |
| the same units at a half or a third of their change | the 10-option gain gone (+0.3, +0.8); not delivered |
| **the units refitted towards silence (WQ, fitted on the instruction layer's model), on top** | **multiple choice (10 options) +3.6 +- 2.7, (4 options) +0.5 +- 2.0, puzzles +0.5 +- 1.3** |
The units rescue answers that run to the end of the budget (the share closed late with the stem and right matches
the replay), and as published they also disturb answers that end well before it; the refit keeps most of the rescue
and removes that cost. The rule of Step 5 decides in the run.

## 4. Method (4 GPUs; work in `/workspace/work`; the sandbox sets `WSP_WORK`, `WSP_CASES`, `PYTHONPATH`)
Keep `PROGRESS.md` current. First, in every shell: `export S=/workspace/work/submission/derive` (the bundle's
`submission/` sits next to `task.json` in `/workspace/work`). `<SID>` is the task's start with role `champion`
(`task.json` -> `starts`); its `<SID>/edit/` chain sits next to `task.json` as the bundle has it.
**Never wait on one command for long:** start every job that takes more than a few minutes detached
(`nohup ... > log 2>&1 &`), give it its own `CUDA_VISIBLE_DEVICES`, and read its log at least every 10 minutes
(a silent agent is stopped). Two tracks run side by side from the start: **A (instruction layer) on GPUs 0-1**,
**B (wrap-up units, fitted on the start) on GPUs 2-3**; they meet in Step 5.

**Step 0. Both cases (minutes).**
```
python -m wsp.case init ix --profile qwen35-2b-think --parent <SID>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro mmlucf musr ifsynth tuluif bbhsynth --dev 384 --holdout 256
python -m wsp.case split ix keep --n 512                    # 64 of each source: the start's answers the fits hold
python $S/wrapup/run.py setup --parent <SID>/edit/edit.json --profile qwen35-2b-think --fit 100 --fit-slow 400
```
`wrapup` draws its own splits; right after its `setup`, keep case `ix`'s problems out of its fit split:
```
python -c "import json,os; W=os.environ['WSP_WORK']+'/wrapup/sets.json'; C=os.environ['WSP_CASES']+'/ix/sets/'; h=set(json.load(open(C+'dev.json')))|set(json.load(open(C+'holdout.json'))); s=json.load(open(W)); [v.__setitem__('fit',[i for i in v['fit'] if i not in h]) for v in s.values()]; json.dump(s,open(W,'w'))"
```
First minutes, three checks (note each in PROGRESS.md): (1) the checkers' dependencies are present, since a checker
that raises is scored as "not followed" and would fail every request silently:
`python -c "import nltk, langdetect, immutabledict; print(nltk.word_tokenize('One two. Three.'))"`, then
`python -c "import sys; sys.path.insert(0,'$S/ix'); import ifx; c = ifx._cls('punctuation_dot')('punctuation_dot'); c.build_description(); print(c.check_following('No full stop here'), c.check_following('One. Two.'))"`
must print `True False`; (2) `wsp.case run` takes the `base@max_tokens=16384` override (`python -m wsp.case run -h`;
otherwise run `base` and use `base__keep.jsonl` below); (3) whether the start already has rewritten units in its last
four layers: `grep -l '"set_\(rows\|cols\)"' <SID>/edit/*.json` and look for layers 20-23 in those ops' `tensor`
names (track B's gate).

**Track A (GPUs 0-1).**
- **A1. The start's own answers** (one after the other, as one detached chain: each generator takes the GPUs whole).
```
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/ifxA.jsonl --count 660 --n 4 --seed 1 --salt ifx --gpus 0,1 \
    --types first_word_sent,first_word_answer,last_word_answer,start_end,word_once,letter_counting,letter_counting2,counting_composition,paragraphs,paragraphs2,exclude_word_harder,punctuation_dot,square_brackets
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/unseen_base.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1 \
    --types last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation
CUDA_VISIBLE_DEVICES=0,1 python -m wsp.case run ix base@max_tokens=16384 keep --n 2
python $S/hold_rows.py $WSP_WORK/runs/ix/base@max_tokens=16384__keep.jsonl $WSP_WORK/ix/hold.jsonl --per-source 64
```
  Check that the hold file has rows of all eight sources (the script prints the counts). Read
  `python $S/ix/ifx_stat.py` on both request files: if fewer than 20% of the fit-type requests are `mixed`, the
  stage has little to learn: say so and continue.
- **A2. Branches:** `python $S/ix/ifx_branch.py ix base $WSP_WORK/ix/ifxA.jsonl $WSP_WORK/ix/ifxbr.jsonl --j 6
  --salt ixb --gpus 0,1`. Fewer than 150 branch points with both outcomes: stop track A and say so.
- **A3. Four fits (seeds 1-4), one per free GPU:** seeds 1-2 on GPUs 0-1; seed 3 on GPU 2 once the start's `dev`
  reference (below) is done there; seed 4 on the first GPU that frees (track B's fits use GPU 3 only); then one
  edit:
```
CUDA_VISIBLE_DEVICES=<g> python $S/ix/pref_gains2x.py --model <built start of case ix> \
    --branches $WSP_WORK/ix/ifxbr.jsonl --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --bad-cut --epochs 2 \
    --keep $WSP_WORK/ix/hold.jsonl --keep-w 10 --out $WSP_WORK/ix/fit/s<seed> --seed <seed>
python $S/ix/fit_gains.py export --fits $WSP_WORK/ix/fit/s1 $WSP_WORK/ix/fit/s2 $WSP_WORK/ix/fit/s3 $WSP_WORK/ix/fit/s4 \
    --k 1.0 --name IX --spec-dir $WSP_CASES/ix/edits --parent ../$P --base qwen35-2b-think
```
  (`<built start>`: `python -c "from wsp import case; print(case.model_path('ix','base'))"`;
  `P=$(python -c "import json,os; print(json.load(open(os.environ['WSP_CASES']+'/ix/case.json'))['parent'])")`.) Every fit prints its
  pairs and its hold rows per source: every source must appear among the hold rows. Short of time, export the
  seeds that are done (at least two).
  The start's `dev` reference runs on GPU 2 as soon as track B's `replay` is done (track B's fits use GPU 3 only):
  `CUDA_VISIBLE_DEVICES=2 python -m wsp.case run ix base dev --n 2`.
- **A4. Read IX and upload it as the floor** (in this order: the first command builds the IX model the second needs).
```
CUDA_VISIBLE_DEVICES=0,1 python -m wsp.case run ix IX dev --n 2          # builds the IX model first
python $S/ix/ifx_gen.py ix IX $WSP_WORK/ix/unseen_IX.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1 \
    --types last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation
python $S/paired.py ix dev IX
python $S/ix/ifx_stat.py $WSP_WORK/ix/unseen_IX.jsonl $WSP_WORK/ix/unseen_base.jsonl     # the edit first, the start second
```
  **IX holds** when the kept-out types are higher by at least 2 SE and no source of `dev` is lower than the start by
  more than 2 SE. If it holds, upload it at once (Uploads). If not, export `--k 0.5` as `IXh` and read it the same
  way; if neither holds, this stage delivers nothing: say so. The floor is the one that holds (IX or IXh).

**Track B (GPUs 2-3), only when the start has no rewritten units in its last four layers** (no `set_rows` /
`set_cols` op on layers 20-23 in its chain); otherwise give GPUs 2-3 to track A. The published commands of
`derive/wrapup/run.py` (its docstring and `-h`), on the start:
```
python $S/wrapup/run.py traces --split fit --n 2 --gpus 2,3
python $S/wrapup/run.py diagnose
python $S/wrapup/run.py replay --gpus 2,3
python $S/wrapup/run.py fit --name W1 --seed 0 --gpu 3 --rounds 2 --max-minutes 80
python $S/wrapup/run.py fit --name WQ --resume W1 --seed 0 --gpu 3 --rounds 3 --max-minutes 50 \
    --w-keep 20 --w-silent 20 --w-early 5
```
The second fit goes on from W1 with the published fit's own terms weighted towards silence: the units' writes where
nothing should change (`--w-silent`) and the KL on every other prediction (`--w-keep`) count four times their default
(5), and the term against closing before the stop depth (`--w-early`, off by default) is switched on at 5, so the
units close less where nothing is lost and disturb the rest less. With `--resume`, `--rounds 3` runs rounds 2 and 3
(the first is W1's); `--max-minutes` counts the traces' loading too.
If `diagnose` shows fewer than 5% of the maths answers reaching 90% of the budget, stop track B (nothing to close).
If WQ is not reported usable after its rounds, track B delivers nothing: say so (W1 itself is not delivered).

**Step 5. The units on top of the floor** (both tracks done, a floor uploaded, WQ usable; write the floor's name,
`IX` or `IXh`, wherever `F` stands): `python $S/stack.py $WSP_CASES/wrapup/edits/WQ.json $WSP_CASES/ix/edits/F.json IXW`, then
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix IXW dev --n 2`, after it the kept-out types (`ifx_gen.py ix IXW
$WSP_WORK/ix/unseen_IXW.jsonl`, the same arguments as for IX, `--gpus 0,1,2,3`); `python $S/paired.py ix dev IXW` (against the start)
and `python $S/paired.py ix dev IXW --ref F` (against the floor). **Deliver IXW** when, against the start, no source
of `dev` is lower by more than 2 SE and the kept-out types are still higher by at least 2 SE, and, against the
floor, the maths source is not lower; otherwise keep the floor.

**Step 6. Confirm once on `holdout`** (`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix base holdout --n 2`,
then the delivered edit the same) if the time
allows, and add it to the report; the delivery does not change on it.

**Uploads.** For edit E of case C:
```
python -m wsp.upload C E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/C__E/edit --task task.json
python -m wsp.upload C E --report REPORT.md --task task.json
```
REPORT.md: the start per source, the request tables of both readings, each fit's pair and hold counts, `diagnose`
and the units' held-out conditions, every paired reading with its SE, which rule decided, what did not work.

**Time (4 x RTX 4090, 6 h).** Step 0 by 0:15. Track A: A1 by 1:10, A2 by 1:20, A3 by 2:50, A4 and the floor
upload by 3:30. Track B: traces by 1:10, replay by 1:20, W1 by 2:40, WQ by 3:30. Step 5 starts by 4:00 at the latest and ends
by 5:15. No new generation after 5:15; the final upload is complete by 5:35. When the final edit is uploaded and
verified, end the run (`axonry-done`).

## 5. What to expect
- The instruction layer: a clear gain on the kept-out instruction kinds, a smaller one on the toolkit's instruction
  sets (the start is near its ceiling there), no other task type lower beyond noise.
- The wrap-up units: fewer maths answers cut at the limit and the ones closed late right more often than left
  alone; nothing on short answers. A null is a result: the report says so with the numbers.
