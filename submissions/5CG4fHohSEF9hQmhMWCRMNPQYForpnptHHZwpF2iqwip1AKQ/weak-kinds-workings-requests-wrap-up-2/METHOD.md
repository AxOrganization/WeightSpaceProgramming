# On the champion: instruction preferences, written workings for every puzzle kind it still gets wrong, and an early wrap-up

## Credits, and what is ours
- **The start** is the champion of the line when the run begins (`entry.json`: parent `champion`); its edit and
  everything in it are its authors' work.
- **The driver, the two-layer plan and every stage's code** are the published entry `0009599139-0015`
  ("workings-and-requests-on-173", #179): `derive/run_v2.py`, its copy of `0009580621-0022` (#156,
  "workings-and-maths-recall-on-130") in `derive/w156/` and its copy of the instruction stage of `0009588957-0032`
  (#168, "workings-and-requests-on-164") in `derive/i168/`. Each directory's README says what it is.
- **The wrap-up unit** is the published method of `0009538688-0032` (#14, "wrap-up"): `derive/wrapup14.py` is its
  `derive/wrapup.py`, unchanged except for its credit line.
- **Ours:**
  1. **the diagnosis of what the champion still gets wrong in its puzzles** (section "What the start still gets
     wrong"): most of its puzzle errors are in kinds the published puzzle layer does not teach;
  2. **the kinds that get a working are the start's own weak kinds**, every kind it gets wrong in at least one answer
     in twenty, ten of the nineteen, instead of a fixed six (`w156/chain.py`: `WORKED`, `HELD`, `ALL_KINDS`), and
     the puzzle layer's neurons scaled with the number of kinds (`W_NEURONS`: 280 for six, 467 for ten);
  3. **the wrap-up on top of the two layers** (`run_v2.py`: `wu_stage`, `fiw_stage`): the champion still loses
     maths and ten-option multiple-choice answers to the token limit, which neither layer reaches; the unit is
     built on the start's own answers, on a layer-10 unit the two layers do not touch, and merged last;
  4. the start is the task's champion, whatever it is (`run_v2.py` takes the start with role `parent` or
     `champion`), so the layers are added to the model that holds the title when the run begins;
  5. `REH=1` makes every upload of the driver a dry run (a rehearsal), as the instruction stage already did;
  6. two fixes found in our rehearsals: the merge of the wrap-up unit takes the delivered FI from the instruction
     stage's own case (`ri`), where that stage writes it (`run_v2.py`: `fiw_stage`), and the instruction stage's
     reading of its held-out requests after the upload runs the start's reading and FI's one after the other on each
     GPU (`i168/run_i.py`; with two GPUs the four readings at once could not all start).
  Nothing else in the published code is changed.

**This entry replaces our `0009603370-0002` (#192)**, the same method without the fixes of item 6: there the merge
step looked for FI in the puzzle chain's case, found only the puzzle layer and would have delivered the puzzle layer
and the wrap-up unit in place of FI, without the instruction layer. Its pull request is closed and it is not funded.

## Entries in progress on this line with related parts, and how this one differs
- `0009602871-0008` and `0009603257-0025` (one author; "worked-puzzles-with-instruction-stage-and-maths-gains",
  "twelve-worked-puzzle-kinds-with-instruction-stage") also run the published instruction stage of `0009588957-0032`
  (the copy in `0009599139-0015`) on the champion. Their puzzle layer is their own: their own writers for most kinds,
  rewritten MLP units fitted with the fitter of `0009580995-0016`, their own holds, eleven or twelve kinds (dates,
  schedules and orders of five among them, word sorting not). Here the puzzle layer is the published chain of
  `0009580621-0022` as `0009599139-0015` ran it (its writers `w156/workings.py`, its neuron choice, its fit with the
  keep term, int8 directions), for the ten kinds the champion gets wrong in at least one answer in twenty (word
  sorting among them; dates, schedules and orders of five held). Their third layer is a maths layer of gains fitted
  on the start's own right answers, made only when the start leaves maths answers unfinished; here it is the
  closed-form wrap-up unit of `0009538688-0032` on layer 10, merged last.
- `0009602170-0009` runs another instruction stage (that of `0009591920-0008`) and late-close units; no puzzle layer.
- `0009602657-0027` and `0009602666-0014` add maths and multiple-choice gain layers to `0009599139-0015`; they do not
  start from the champion and derive no puzzle or instruction layer.

## What the start still gets wrong (practice puzzles; the champion `0009598520-0002`, 1,200 puzzles x 2)
| kind | the champion right | `0009599139-0015` (#179) right | worked by #179 | worked here |
|---|---|---|---|---|
| geometric shapes | 66.9% | 99.2% | yes | yes |
| hyperbaton | 78.9% | 84.2% | no | yes |
| object counting | 79.8% | 82.3% | no | yes |
| web of lies | 83.3% | 86.1% | no | yes |
| logical deduction, seven objects | 87.9% | 99.2% | yes | yes |
| multistep arithmetic | 91.2% | 96.5% | no | yes |
| word sorting | 92.9% | 93.6% | yes | yes |
| formal fallacies | 93.3% | 93.3% | no | yes |
| navigate | 94.2% | 100% | yes | yes |
| boolean expressions | 95.2% | 95.2% | no | yes |
| the nine other kinds | 95.2-100% | 92.1-100% | two of them | no |
| all nineteen | 91.8% | 95.2% (+3.42 +- 0.65) | | |

The published puzzle layer teaches six kinds and fixes them (geometric shapes 66.9% -> 99.2%, logical deduction of
seven 87.9% -> 99.2%), while the kinds it only holds keep most of their errors (hyperbaton, object counting, web of
lies 79-86% right): after it, those three alone hold more than half of what the model still gets wrong, and one held
kind loses (date understanding 95.6% -> 92.1%). The writer of the workings (`w156/workings.py`, #156's) already has a working for every kind: on the shipped
practice puzzles it writes a working that reproduces the known answer for 1,150 of the 1,152 puzzles of nine of
the kinds above (the tenth, navigate, is one of the published six). A working is the facts of the puzzle in a fixed layout, the rule of the kind and the conclusion, a few
hundred tokens; for example object counting lists every thing with its kind and a running total, hyperbaton gives
every adjective its class number and checks that the numbers rise.

## Why the wrap-up as well (practice problems; our measurements)
On the champion, 7-16% of maths answers and about one ten-option multiple-choice answer in ten are cut at the token
limit with the thinking still open; with twice the budget 59% of the cut maths answers end right, so they are slow,
not lost. The unit pulls toward the model's own end-of-thinking state past the 75th percentile of its right answers'
thinking length, early enough that the answer re-solves the problem: on 600 practice problems it turned half of the
cut maths answers into right ones (+2.25 +- 0.95 overall, maths +9.2; its random control -0.67). Built on the
champion and carried onto `0009599139-0015` (#179's published edit: the champion with both layers) it read
**+2.67 +- 0.81** on 600 practice problems (maths +6.2, ten-option multiple choice +5.0, cut 4.9% -> 0.3%): the two
layers leave this loss untouched and the unit removes it on top of them.
On a start that already closes late answers (the line's champion `0009601111-0009` carries the published late-close
units), the unit has less to do: on 600 practice problems it still added +1.00 +- 0.84 on top of those units carried
onto the champion of the time (maths +1.2, no task type lower), so it is kept; `STACK_W=0` skips it.

## The method (4 x RTX 4090; the driver reads the time left from task.json)
The sandbox sets the toolkit, the base and the environment (`WSP_WORK`, `WSP_CASES`, `WSP_MODELS_DIR`, `PYTHONPATH`); do
not override them. Copy `submission/`, `task.json` and the task's start directory (`<sid>/edit/`, with its chain, as
the bundle has it) to `/workspace/work/`, then:
```
cd /workspace/work
nohup python /workspace/work/submission/derive/run_v2.py > /workspace/work/run_v2.log 2>&1 &
grep PHASE /workspace/work/run_v2.log            # progress
tail -n 30 /workspace/work/v2/chain.log /workspace/work/v2/inst.log /workspace/work/ri_work/NUMBERS.md
```
| step | GPUs | what |
|---|---|---|
| start | CPU | the start = the task's start with role `parent`, else the one with role `champion` |
| cases | CPU | case `pz` on the start |

| wi, GPUs 0,1 | 2 | `w156/chain.py --case pz --until edits` with `W_ONLY=1`: prep (the practice puzzles split by number, more drawn with the toolkit's own program for the worked kinds), keep (the start's own answers to requests, word problems, maths, multiple choice and the puzzles of every held kind), the puzzle layer's neuron choice and fit on the workings of the ten weak kinds with the keep term, `WD` (int8 directions), `Fw`. Uploaded as the floor as soon as it exists, unless FI is already up |
| wu, GPUs 2,3 | 2, then 1 | started right after the chain, beside it: case `wu` on the start, its own answers to 300 practice problems of the reasoning families (maths, word problems, ten- and four-option multiple choice, puzzles), 2 each; one read of layer 10 along them (`wrapup14.py collect`); their thinking lengths. The instruction stage starts on these GPUs when it is done: in our rehearsal that stage ended in 1 h 20 min while the chain took more than 3 h, so this costs no wall time |
| wi, GPUs 2,3 | 2 | `i168/run_i.py`: requests of the stage's kinds, 4 answers each, 4 branches after every finished thinking, 2 preference fits, `I` at half strength; it waits for `Fw` and delivers **FI = Fw's ops + I's ops**, one edit on the start, after its dry run, payload and `axonry-check`; I alone on the start if the chain cannot finish in time |
| fiw | CPU | **FIW = the delivered layers + the wrap-up unit at dose 3** (`theta` = the 75th percentile of the start's right answers' thinking length, reference 22,000 tokens), on the least used layer-10 unit the layers do not touch, its ops last. The delivered layers are FI from case `ri` when the instruction stage passed its upload step (`ri_work/done.upload`), else Fw from case `pz`; FIW is written and packaged in that case (`wsp.upload ri FIW`, or `pz`). Dry run of the upload, payload, `axonry-check`, upload in place of the edit up. If any check fails, the edit already up stands. The log says which layers it merged: `fiw: FI + wrap-up unit ...` |

Each step leaves a marker in `/workspace/work/v2/` and is skipped when the driver is started again; the chain's and
the instruction stage's own status: `python derive/w156/chain.py --case pz --out /workspace/work/pz --status`,
`python derive/i168/run_i.py --status`. A generation step that hangs is restarted by the chain, and the wrap-up
step's generation has its own watchdog. **A vLLM engine can hang on shutdown after its last answer:** its GPU keeps
its memory at 0 % utilisation, the progress bar of that step shows 100 %, and the step never returns (we saw it
twice in our rehearsals). If the instruction stage shows this for more than eight minutes, find the
`VLLM::EngineCore` process that holds that GPU (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`)
and stop that PID only: the step then writes its results and goes on. Never `pkill -f` a pattern; stop a job by
its PID. Whatever runs late gives way to the upload: FIW stands only if it passed its checks,
otherwise FI, otherwise Fw, otherwise I alone. If the instruction stage finds fewer than 30 starts with both outcomes
after one thinking, it has nothing to learn from: stop it before its fits, let Fw stand, and say so in the report.
`STACK_W=0` skips the wrap-up steps. **Check the merge:** after `UPLOADED FI` in `v2/inst.log`, the fiw step must
print `fiw: FI + wrap-up unit ...` and then `UPLOADED FIW`; if it prints `fiw: Fw + ...` instead, the instruction layer
is missing from FIW: upload FI again (`python -m wsp.upload ri FI --task task.json --report REPORT.md`) and let it stand.

## What to expect
Our rehearsal of this driver (`REH=1`, the fixed merge) on `0009598520-0002`, read on practice problems no fit saw:
puzzles 91.1% -> 96.5% right (600 puzzles x 2, +5.4 +- 1.1; the worked kinds 63-95% -> 91-100%); the instruction
stage's held-out request kinds 78.0% -> 83.3% kept (300 requests); maths 85.0% -> 88.1% with answers cut at the token
limit 13.1% -> 3.1%, ten-option multiple choice 63.1% -> 63.1% with cuts 8.1% -> 0% (160 problems each, one answer).
- On the start's weak puzzle kinds: shorter, written-out thinking and most of their errors gone (the published layer
  did this for its six kinds); the other kinds held near the start.
- On instruction requests of the stage's held-out kinds: more of them kept than by the start.
- Maths and ten-option multiple choice: most answers that ran into the limit now end, about half of the maths ones
  right.
- Report every measured number with its practice source, and every stage that was skipped or held back with the
  reason.
