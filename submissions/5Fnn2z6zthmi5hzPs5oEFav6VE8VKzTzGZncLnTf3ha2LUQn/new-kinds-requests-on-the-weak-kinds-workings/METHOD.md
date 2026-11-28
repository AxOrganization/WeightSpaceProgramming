# new-kinds-requests-on-the-weak-kinds-workings: the published instruction-kinds stage, run on all four GPUs on the finished entry that taught the champion the workings of its weak puzzle kinds

This entry names the finished entry `0009603843-0016` ("weak-kinds-workings-requests-wrap-up-2") as its parent in
`entry.json` and **starts from its model**: the line's champion `0009601111-0009` with that entry's layer on top (written
workings of ten puzzle kinds the champion still got wrong, an instruction layer of another published stage at half
strength, and one wrap-up unit in layer 10). Nothing of the start is derived or delivered again. One layer of unit and
head gains is fitted in the run, on the start's own answers to practice requests, and delivered as one edit on the start.

## Credits, and what is ours
- **The start** is `0009603843-0016` (clar0x) on `0009601111-0009` (the champion, ours) on `0009598520-0002`; every layer of
  it is its authors' work.
- **The whole instruction stage** is the published method of the finished entry `0009602170-0009`
  ("requests-of-new-kinds-then-close-in-time", its track A): requests of instruction kinds the practice sets lack, the
  start's own answers to them, answers resampled after one and the same finished thinking, a preference fit of per-unit
  and per-head gains (layers 4-21) where some answers keep the request and others break it, held by a keep term on the
  start's own right answers to all eight practice sources, 13 kinds fitted and 4 kept out of every fit. **Every file of
  `derive/` except `ix_run.py` is byte-identical to that entry's published file of the same name** (`derive/ix/`,
  `derive/hold_rows.py`, `derive/paired.py`); that entry credits, and we credit with it, the toolkit's route
  (`playbook.md`, "Separate the answer from the thinking"), the second stage of `0009591920-0008` (goldpig4) and
  `0009598608-0007` (goldpig5) whose `derive/on/` files `derive/ix/` copies, and the open-instruct checkers in
  `derive/ix/ifg/` (Apache-2.0).
- **Ours:**
  1. the choice of start and the measurement behind it (section 2): the instruction-kinds stage was fitted on a start
     without the weak-kinds workings; on top of them, its published layer costs nothing on practice maths, puzzles or
     the toolkit's instruction sets, so the two finished layers can be joined, the second one fitted again on the joined
     model's own answers;
  2. running the stage on a start whose own layers include rewritten units in its last four layers, where the stage's
     own method gives its second track's GPUs to the instruction stage: **all four GPUs go to it**, every generator on four
     GPUs and **the four seeds fitted at once**, one per GPU;
  3. `derive/ix_run.py`: the commands below as one resumable job with the fixed delivery rule, so the run does not
     depend on typing two dozen commands in order.

**Entries in progress on this line that use related parts, and how this one differs.** Entries that fit their own
puzzle workings and the instruction stage of `0009588957-0032` on the champion are in progress, and one fits repaired
instruction answers and maths branches on the champion. This entry fits no puzzle layer and no maths layer: its start is
a finished entry, and its own layer is the instruction-kinds stage of another finished entry (the open-instruct kinds of
`ifx.py`, not `0009588957-0032`'s types and checkers), held on all eight practice sources.

## 1. What the start leaves to this layer
The start's instruction layer comes from a stage whose requests are the toolkit's instruction types; its own report
reads the held-out kinds of that stage at 78.0% for the champion and 80.3% with its layer (300 practice requests). The
finished entry `0009602170-0009` measured, on the requests of further kinds that `ifx.py` builds (kinds the practice
sets do not contain), that a start of this lineage follows only 50.7% of four kept-out kinds (300 requests x 4
answers), and that its layer raised that to 59.8% with no dev source lower by more than 2 SE. These kinds are what this
layer targets; the start's answers to the toolkit's two instruction sets are near their ceiling (94-95% in our reads),
so those sets cannot show the gain, and the run reads it on the kept-out kinds.

## 2. What we measured before writing this (practice problems, paired with the start on the same problems and seeds)
We applied the finished stage's published layer (its 831 gain ops, fitted on another start) on top of the start's model
and read it against the start:

| practice set | problems | published layer on the start, against the start |
|---|---|---|
| fresh maths | 256 | +1.56 +- 1.23 (cut 0.0% -> 0.0%) |
| fresh puzzles, two splits | 512 | +0.78 +- 1.11 and -0.78 +- 0.96 |
| fresh word problems | 256 | -0.78 +- 1.24 |
| instruction requests (the toolkit's two sets) | 256 | +0.00 +- 1.75 |
| fresh multiple choice (four- and ten-option) | 512 | -0.20 +- 1.76 |

The layer, fitted elsewhere, sits on the start's workings and units without a cost beyond the error of these sets.
Here it is fitted again on the start's own answers, with every layer of the start in place.

## 3. Method (4 GPUs; work in `/workspace/work`; the sandbox sets `WSP_WORK`, `WSP_CASES`, `PYTHONPATH`)
**One job does the whole method** (`derive/ix_run.py`, ours: it runs exactly the commands below, in this order, and
applies the delivery rule below; it writes `PROGRESS.md`):
```
python submission/derive/ix_run.py --upload     # starts the job detached (log ix_run.log) and returns at once
python submission/derive/ix_run.py --status     # which steps are done, whether the job is running, its newest lines
```
Read `--status` at least every 10 minutes. Every step writes its output once and is skipped when the job runs again, so
if the job is not running and the log does not end with `RUN_END` or a stated stop, start the same command again. If
the script itself fails, run the commands below by hand from the first step whose output is missing. When the log shows
`RUN_END`, add to `cases/ix/REPORT_<edit>.md`, in your own words, what was targeted and what did not work, upload once
more with `python -m wsp.upload ix <edit> --report <file> --task task.json` if time allows, and end the run
(`axonry-done`).

The commands (`export S=/workspace/work/submission/derive`; `<SID>` is the task's start, role `parent`, else
`champion`, in `task.json` -> `starts`; its `<SID>/edit/` chain sits next to `task.json` as the bundle has it):

**Step 0 (minutes).**
```
python -m wsp.case init ix --profile qwen35-2b-think --parent <SID>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro mmlucf musr ifsynth tuluif bbhsynth --dev 384 --holdout 256
python -m wsp.case split ix keep --n 512                    # 64 of each source: the start's answers the fit holds
```
Two checks (note each in PROGRESS.md): the checkers' dependencies
(`python -c "import nltk, langdetect, immutabledict; print(nltk.word_tokenize('One two. Three.'))"`, then
`python -c "import sys; sys.path.insert(0,'$S/ix'); import ifx; c = ifx._cls('punctuation_dot')('punctuation_dot'); c.build_description(); print(c.check_following('No full stop here'), c.check_following('One. Two.'))"`
must print `True False`), and that `wsp.case run` takes the `base@max_tokens=16384` override (`python -m wsp.case run -h`;
otherwise run `base` and use `base__keep.jsonl` below).

**Step 1. The start's own answers** (one detached chain, each generator on all four GPUs):
```
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/ifxA.jsonl --count 660 --n 4 --seed 1 --salt ifx --gpus 0,1,2,3 \
    --types first_word_sent,first_word_answer,last_word_answer,start_end,word_once,letter_counting,letter_counting2,counting_composition,paragraphs,paragraphs2,exclude_word_harder,punctuation_dot,square_brackets
python $S/ix/ifx_gen.py ix base $WSP_WORK/ix/unseen_base.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1,2,3 \
    --types last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix base@max_tokens=16384 keep --n 2
python $S/hold_rows.py $WSP_WORK/runs/ix/base@max_tokens=16384__keep.jsonl $WSP_WORK/ix/hold.jsonl --per-source 64
```
Check that the hold file has rows of all eight sources (the script prints the counts). Read `python $S/ix/ifx_stat.py`
on both request files: if fewer than 20% of the fit-type requests are `mixed`, the stage has little to learn: say so and
continue.

**Step 2. Branches, then the start's dev reference:**
```
python $S/ix/ifx_branch.py ix base $WSP_WORK/ix/ifxA.jsonl $WSP_WORK/ix/ifxbr.jsonl --j 6 --salt ixb --gpus 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix base dev --n 2
```
Fewer than 150 branch points with both outcomes: stop and say so (nothing is delivered).

**Step 3. Four fits at once (seeds 1-4, one per GPU), then one edit:**
```
CUDA_VISIBLE_DEVICES=<seed-1> python $S/ix/pref_gains2x.py --model <built start of case ix> \
    --branches $WSP_WORK/ix/ifxbr.jsonl --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --bad-cut --epochs 2 \
    --keep $WSP_WORK/ix/hold.jsonl --keep-w 10 --out $WSP_WORK/ix/fit/s<seed> --seed <seed>
python $S/ix/fit_gains.py export --fits $WSP_WORK/ix/fit/s1 $WSP_WORK/ix/fit/s2 $WSP_WORK/ix/fit/s3 $WSP_WORK/ix/fit/s4 \
    --k 1.0 --name IX --spec-dir $WSP_CASES/ix/edits --parent ../$P --base qwen35-2b-think
```
(`<built start>`: `python -c "from wsp import case; print(case.model_path('ix','base'))"`;
`P=$(python -c "import json,os; print(json.load(open(os.environ['WSP_CASES']+'/ix/case.json'))['parent'])")`.) Every fit
prints its pairs and its hold rows per source: every source must appear among the hold rows. Short of time, export the
seeds that are done (at least two).

**Step 4. Read IX and deliver it** (in this order: the first command builds the IX model the second needs):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix IX dev --n 2
python $S/ix/ifx_gen.py ix IX $WSP_WORK/ix/unseen_IX.jsonl --count 300 --n 4 --seed 2 --salt ifx --gpus 0,1,2,3 \
    --types last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation
python $S/paired.py ix dev IX
python $S/ix/ifx_stat.py $WSP_WORK/ix/unseen_IX.jsonl $WSP_WORK/ix/unseen_base.jsonl     # the edit first, the start second
```
**The delivery rule (fixed before any reading; the published stage's own rule).** IX *holds* when the kept-out kinds are
higher by at least 2 SE and no source of `dev` is lower than the start by more than 2 SE. If it holds, upload it. If not,
export `--k 0.5` as `IXh`, read it the same way and upload it if it holds; if neither holds, nothing is delivered and the
report says why with every reading.

**Step 5. Confirm once on `holdout`** if at least 60 minutes are left
(`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wsp.case run ix base holdout --n 2`, then the delivered edit the same) and add
it to the report; the delivery does not change on it.

**Uploads.** For edit E of case ix:
```
python -m wsp.upload ix E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/ix__E/edit --task task.json
python -m wsp.upload ix E --report REPORT.md --task task.json
```
REPORT.md: the start per source, the request tables of both readings, each fit's pair and hold counts, every paired
reading with its SE, which rule decided, what did not work.

**Time (4 x RTX 4090, 6 h).** Step 0 by 0:15; step 1 by 1:00 (four GPUs; in our rehearsal on 4 x RTX 4090 the start's 2,640 fit answers took 20 min, the engines' first start included, and the 1,200 kept-out answers 8 min);
step 2 by 1:35; step 3 by 3:05 (the four seeds at once, about 85 min each); step 4 and the upload by 4:00 (IXh, if
needed, by 4:40); step 5 if time allows. No new generation after 5:15; the final upload is complete by 5:35. When the
final edit is uploaded and verified, end the run (`axonry-done`).

## 4. What to expect
- The kept-out instruction kinds followed more often than by the start, by several points (the published stage: 50.7
  -> 59.8% on its start), and no practice source of `dev` lower by more than its error.
- The start's own layers kept: maths, multiple choice, word problems and puzzles within their error, because the layer
  is fitted with the start's own right answers to all eight sources held.
- A null or a loss is a result: the report gives every reading with its error.
