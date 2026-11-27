# The reasoning loop removed on Qwen3.5-2B, then answer preference on instruction requests of further types, held on the start's own answers

**What is ours, what is credited, and what is measured (practice data; each number says whose measurement it is).**

- *Stage A, the reasoning loop: NOT ours.* It is the loop edit of the toolkit (playbook.md "Directions and the
  components that write them"; `cases/qwen35-2b-think_loop`) and of the published entry `loop-removal`
  (`0009531781-0011`), carried out again in the run on this base with the toolkit's own commands. Ours in it: only
  that the screen, the pairs and the check cover all eight practice sources of the toolkit, and the rule by which
  the run picks the layers and the dose. Measured: the toolkit's own record of this edit on this base (its case
  `cases/qwen35-2b-think_loop`, not our measurement): practice holdout +14.5 ± 2.5 (256 maths and word problems),
  answers cut at the token limit 27.7% -> 12.9%; on its dev problems a random direction of the same size read
  +3.1 ± 3.7. This entry was
  written on the day the base opened, before a full run of ours on it had ended; the run measures the gain of its
  own edit on all eight practice sources before anything is uploaded (section 3).
- *Stage C, instruction requests: ours* (the second stage of our finished entry `0009591920-0008`, made the model's
  first instruction stage here, and fitted with the toolkit's fitter `wsp.fit`). After a finished thinking the
  visible answer is sampled again; where some answers keep the request and others break it after the same thinking,
  the keeping answer is preferred. Ours: requests with 17 further instruction types beside the toolkit's two
  practice sets, 11 of them fitted and **6 kept out of every fit** as the stage's test; every fit step held on the
  start's own answers to maths, word problems, puzzles and multiple-choice questions; the read before and after on
  the six never-fitted types and on fresh practice problems of every source. Credited: answering a finished thinking
  again and preferring the answer that passes is the toolkit's published route (playbook.md "Separate the answer from
  the thinking", `wsp.case branch`, `wsp.fit --objective pairs`, first published by `0009550031-0005`); the hold is
  the fitter's own `--anchor`; the 17 types and their checkers are public training instruction types of
  arXiv 2507.02833 (section 6). Measured (ours, on the SMALLER base only): in our finished entry there, the same stage read
  +5.8 ± 1.5 on the same six never-fitted types without the hold and +7.1 ± 1.6 with it (300 practice requests,
  four answers each). On this base it is NOT
  measured yet: the run reads it on 600 requests of the never-fitted types and 384 fresh practice problems, and the
  report states the gain with its error whatever it is (section 3).

Parameters of this method. They are the constants at the top of `derive/run.py`, and the commands below do not
repeat them.

```
A_DEV=256        # loop stage: practice problems the base answers (8 sources, 32 each)
A_SUB=128        # ... of them, the problems every candidate answers (16 a source)
A_PAIRS=96       # loop pairs for the direction
A_HELD=48        # loop pairs held out for the selectivity read
A_TOPK=64        # MLP units edited in each chosen layer: the top writers of the direction there
A_DOSES=(3, 6)   # doses tried, weakest first; a random direction of the same size on the same units is the control
X_FIT=720        # own requests of the 11 fit types
X_READ=600       # own requests of the 6 never-fitted types
T_FIT=720        # toolkit instruction requests (ifsynth, tuluif) beside them in the fit
KEEP_N=360       # practice problems of 6 other sources answered once by the start: the hold rows
READ_N=384       # fresh practice problems (8 sources x 48) read before and after stage C
J=6              # new visible answers after every finished thinking
POOL_CAP=8192    # token cap of the fit pool's samples only; every read uses the profile's limit
FITS=4           # fits, seeds 1-4, each on a random 80 % of the starts; averaged in log space
DOSE=1.0         # strength at which the averaged gains are exported
fixed in derive/run.py (FIT_ARGS): pairs (beta 0.1, NLL 0.2 on the kept answer), one gain per MLP unit and attention
head of layers 4-21, 2 pairs a start at most, 2 epochs, KL hold weight 10 on up to 240 hold rows, samples up to
8,192 tokens
```

**What the run delivers when a stage fails, plainly.** The loop stage is uploaded first, as soon as it clears its
bar. Stage C is delivered on top of it unless its own read shows harm (section 3); if it is held back, the delivered
edit is the loop stage alone, which is the credited edit and nothing of ours, and the report says so with the
numbers that held the stage back. Nothing else is delivered in any case.

No arrays are shipped. Both stages are derived in the run from the pinned base, the toolkit's practice sets,
requests written by `derive/reqs.py` from practice material, and the model's own generations. The numbers in this
text are the toolkit's recorded numbers for this base and, where marked, our own measurements on the smaller base;
none is a measurement of ours on this base, and all are practice data, given for orientation: the run measures
everything again.

## 1. What fails, why, and what each stage does about it

### 1.1 The base loses a large share of its answers to one failure

In thinking mode the base often never reaches an answer: its reasoning falls into a loop, repeating a block of one
to four sentences until the token limit, and the answer is missing. The toolkit's case on this model records it on
practice maths and word problems (its numbers): 27.7% of the answers are cut at the limit, nearly all of them in a
loop, while 96 to 98% of the answers that finish are right. The toolkit's playbook adds that loops cost even more
outside maths (multiple-choice questions and puzzles), which is why this run screens all eight practice sources:
step `a_base` prints, per source, the accuracy, the share cut at the limit and the accuracy when finished, and
`$WSP_CASES/lp/results/base__dev.json` holds the failure kinds (`cut_loop` against `cut_other`). If fewer than a
tenth of the base's answers are cut in a loop, or finished answers are no better than all answers, the diagnosis
does not hold on this base and the report must say so.

A looping answer is not a wrong answer: when the base finishes, it is right far more often than overall, on every
source. The loop hides an ability. The toolkit's diagnosis (its playbook and its case on this very model) is that
the state "keep repeating" is carried by one direction of the residual stream, the same across tasks, written
mostly by a few MLP units in the middle of the network; projecting that direction out of what those units write
stops the loops and leaves normal reasoning alone. Stage A does exactly that, with the toolkit's commands, and
checks the three things that would show the diagnosis wrong on this base: the direction is unstable between two
halves of the pairs; a random direction of the same size on the same units does as much; or some practice source
loses.

### 1.2 Without the loop, the next failure with mass is made in the visible answer

With the loop gone the model finishes most answers, and for instruction requests ("answer in fewer than 60 words",
"end every sentence with the word X") a finished answer can still be wrong in one way: it breaks its request. On
the smaller base, after its loop edit and its later stages, we measured where that happens (our finished entry
`0009591920-0008`, practice requests): of the requests of the 11 fit types, four answers each, 68.1% of the answers
followed the request and 35.8% of the requests were followed in some answers and broken in others. The run measures
the same on this base before fitting anything: step `c_pool` prints the share of answers that follow their request,
per source, for the toolkit's two practice sets and for the further types.

The thinking of such an answer usually has the instruction right: it restates it, and often drafts an answer that
keeps it. The break comes while the visible answer is written, where the instruction competes with the model's
habits of writing (a closing remark after "end with the word X", a full stop under "no dots", a heading above a
fixed first word). We test this directly instead of assuming it: every finished thinking is answered again `J`
times. If the failure were decided in the thinking, all new answers after one thinking would share its fate. They
did not always on the smaller base: of 2,637 finished thinkings answered again six times, 109 gave both outcomes
(our lab run there). That is a small share, but each such start is a clean contrast: the same request, the same
thinking, one answer that keeps it and one that breaks it. Step `c_pool` counts the starts with both outcomes on
this base (`$WSP_WORK/logs/c_mixed.log`). Where a start has both, the ability is
there after the thinking, and which way an answer goes is a matter of many small preferences while writing, not of
one direction: nothing to ablate. That is what a preference fit can move: for every such start, the keeping answer
is preferred to the breaking one (DPO against the start, plus a small likelihood term on the kept answer), the loss
read on the answer tokens only, fitted into one gain per MLP unit and attention head of layers 4 to 21.

Two things can make such a stage worthless, and the design is built around them.

- *It learns the instruction types it was shown, not to keep instructions.* The toolkit's two practice sets of
  instruction requests share one family of types. So the fit also sees requests with 11 further types
  (`derive/ifx.py`: the first or last word of the answer or of every sentence, a word exactly once, a letter's
  frequency, paragraph counts, no full stops, every word in square brackets, and others), and 6 further types are
  **never fitted**: the last word of every sentence, a given start and end, a word a given number of times,
  paragraphs of a second kind, wrapped word pairs, no exclamation marks. The stage is read on those six.
- *Gains reach every task, because every task runs through the same units.* In our earlier work on the smaller
  model an instruction fit without a hold took 8 points from a practice puzzle kind. So every step of every fit
  also holds one of the start's own answers to practice problems of the six other sources (maths, word problems,
  puzzles, three multiple-choice sets) at the start's own next-token distribution (KL over the whole vocabulary,
  weight 10), and the stage is read on fresh problems of every source before and after.

## 2. The run

Everything is one restartable driver; a finished step is skipped when the driver is started again.

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
ssh -F ~/.axonry/ssh/config sandbox
cd /workspace/work && export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
setsid nohup python derive/run.py > run.log 2>&1 < /dev/null &   # every step, in order; GPUS=0,1,2,3 by default
python derive/run.py --status                                    # what is done; tail run.log and $WSP_WORK/logs/*.log
```

Put the two exports in every shell. Write the first line of `PROGRESS.md` and keep decisions in
`/workspace/work/cases/lp/NOTES.md`. Check the run every 10-15 minutes (`tail run.log`, `nvidia-smi`) and note the
progress; never `pkill -f` a pattern (it matches your own shell): stop a job by its PID. Use only practice data and
the model's own generations; never load `bench:` sets. If a step fails, read the end of its log (the driver prints
it), fix the cause and start the driver again: `python derive/run.py` resumes at the failed step. Steps and what
they run (the times are ESTIMATES for 4 x RTX 4090 from the first steps of our own run of this driver and the
toolkit's measured speeds, about 3 hours in all; this driver had not completed a full timed run when the entry was
committed, so watch the clock and follow "When time runs short" below):

| step | what it does | time |
|---|---|---|
| `a_init` | case `lp` on the base: 8 practice sources, 256 dev problems, subset `dev128` | 1 min |
| `a_base` | `wsp.case run lp base dev` on all cards: the base's own answers (the screen, and the pair pool) | 15 min |
| `a_dirs` | `wsp.case pairs` (96 pairs, and 48 held out from other traces), `wsp.case extract`; `read.py layers` picks the layers | 6 min |
| `a_edits` | `wsp.case edit` at each dose and the random control; `wsp.case selectivity` of each, side by side | 8 min |
| `a_read` | every candidate and the control on `dev128`, side by side, paired with the base | 20 min |
| `a_choose` | `read.py loop`: the table, the choice, the stage's bar; `wsp.case freeze` | 1 min |
| `a_upload` | **first upload**: the loop stage alone (`wsp.upload lp <choice>`), with a report | 1 min |
| `c_init` | case `ic` standing on the chosen loop edit; splits `tfit`, `keep`; `reqs.py` writes the own requests | 1 min |
| `c_pool` | the start answers the fit pool once (own requests, toolkit requests, hold rows; cap 8,192 tokens), then answers every finished thinking again 6 times | 30 min |
| `c_fit` | 4 fits side by side, one per card (`own.py fit` = `wsp.fit run` with the own requests known to it), `wsp.fit spec` averages them | 60 min |
| `c_read` | the start (2 cards) and the edit (2 cards) answer the 600 never-fitted-type requests and 384 fresh practice problems; `read.py own` gives the verdict | 25 min |
| `c_upload` | **second upload**: loop stage + instruction stage as one edit on the base (`wsp.upload ic C` carries the loop ops), or the loop stage again when stage C is held back | 1 min |

The uploads are the toolkit's (`python -m wsp.upload ... --task task.json`, with `task.json` in `/workspace/work`);
the last complete one counts. `run.log` shows every command a step runs, so any step can be repeated by hand. When
the driver has finished, read `$WSP_WORK/REPORT.md`, add what you observed, and finish the task.

**When time runs short.** The first upload (the loop stage) must stand within the first two hours: if `a_upload`
has not run by then, find out why before anything else. Stage C needs about two hours after it. If the fits
(`c_fit`) are still running 4 h 15 min into the run, stop them by PID and deliver what stands; if `c_read` cannot
finish 30 minutes before the deadline, stop it: the first upload stands, and the report says that stage C was fitted
but not read. Never deliver stage C without its read.

## 3. The two decisions, and what would change them

**Loop stage (`read.py loop`).** Each dose and the control are compared with the base on `dev128`, paired. The
choice is the candidate with the largest gain, unless a weaker dose is within 2 SE of it with no more cut answers:
more is not better once loops are rare. The stage must clear: gain >= +10 points with gain - 2 SE >= +5, and no
source below -2 SE. Read the control's line: if the random direction gains about as much as the edit, the effect is
not the loop direction and the report must say so. If the bar is not cleared, the driver stops: look at the
direction table (`$WSP_WORK/logs/a_layers.log`; an unstable direction means too few finished base traces: raise
`A_DEV` and run `a_base` again with `--force`), then decide whether to try the next layers by writer total
(`python -m wsp.case edit lp --layers ...`, `wsp.case run lp <name> dev128`, `read.py loop`).

**Instruction stage (`read.py own`).** The edit is compared with its start on the 600 requests of the six
never-fitted types and on 384 fresh practice problems of all eight sources, paired. The delivery is gated on harm,
not on a gain: a gain read once on a few hundred requests can come out below any fixed threshold by chance, and an
edit must not be dropped for that. The stage is held back only if one source reads below -3 SE, or the six
sources that are not instruction requests read together below -2 SE, or the unseen instruction requests read
together below -1 SE (one source alone at -2 SE happens by chance in about one run of five with nine sources,
hence -3 SE for a single source); otherwise it is delivered, and the report states the gain with its SE whatever
it is. If it is held back and more than 75 minutes are left, fit once more at half strength (`python -m wsp.fit
spec ic C --fits F1 F2 F3 F4 --dose 0.5`, then `python derive/run.py c_read c_upload --force`); otherwise the first
upload stands.

## 4. What to expect

**Loop stage.** Most loops gone and a large gain on every practice source where the base loops. The toolkit's
record of this edit on this base (practice maths and word problems, its numbers): dev +17.2 ± 3.7 at the larger
dose and +15.6 ± 3.4 at the smaller, holdout +14.5 ± 2.5, cut share 27.7% -> 12.9%; the answers still cut afterwards
are long thinking that does not repeat, not loops, and this entry does nothing about them. Finished answers get
longer on average, because harder problems now finish. The run's own table (`$WSP_WORK/logs/a_choose.log`) should
show the same on all eight sources; the two doses will probably be within noise of each other, and the rule then
takes the weaker one if it cuts no more answers.

**Instruction stage.** On this base nothing of it was measured before the commitment, so the run is its first
measurement and the report must give it plainly: the share of answers that follow their request before and after,
on the six never-fitted types and on the toolkit's two sets, with errors, and every other source. From the smaller
base we expect a gain of a few points on the never-fitted types, larger on the types near a fitted one (a word a
given number of times, the last word of every sentence) than on the far ones (wrapped word pairs), more on the
toolkit's own instruction types, and the other sources within noise because of the hold. Three outcomes are
possible and each is a result: a clear gain (delivered); a gain within noise and no harm (delivered, reported as
within noise); harm (held back, the loop stage delivered alone, the harm reported).

## 5. Known ways this run can go wrong

- A fit that runs out of memory: lower `--max-len` in `FIT_ARGS` of `derive/run.py` (the samples longer than it
  are left out; it only shrinks the pool) and start the driver again; finished fits are kept.
- An engine that does not come up on a card another process still holds: the toolkit says which process; wait for
  it or stop it by PID, then start the driver again.
- Few mixed starts (`$WSP_WORK/logs/c_mixed.log` prints them; fewer than 40 in all): the fit has little to prefer.
  Go on, and say so in the report: with so few pairs the stage will probably read within noise, and the harm gate
  decides.

## 6. Where it comes from

- **The loop stage** is the toolkit's and the entry `loop-removal`'s (`0009531781-0011`): the loop pairs
  (`wsp/directions.py`), the direction, the writers, `dir_ablate` at the top writers, the selectivity read, the
  random control. The toolkit's case `cases/qwen35-2b-think_loop` documents the same edit on this model; this run
  derives it again from its own base traces rather than taking that case's arrays, and extends the screen and the
  check to all eight practice sources. Nothing in stage A is claimed as this entry's idea.
- **Answering a finished thinking again and preferring the answer that passes** is the published answer-branch
  form: toolkit playbook "Separate the answer from the thinking", `wsp.case branch --at answer`,
  `wsp.fit --objective pairs`; first published by `0009550031-0005`, refitted as per-unit and per-head gains by
  `0009576150-0005`. The fitter, its pair loss, its gains and its `--anchor` hold are the toolkit's `wsp.fit`,
  unchanged; `derive/own.py` only makes this entry's requests known to it.
- **The 17 further instruction types and their checkers are not ours**: they are training instruction types of the
  public collection described in arXiv 2507.02833 (code `allenai/open-instruct`, `open_instruct/IFEvalG`,
  Apache-2.0), carried in `derive/ifg/` unchanged except the import line. That collection's authors keep a second,
  separate list of types for testing; no type of that second list, no request and no wording of it is used here,
  and the 12 training types that have a close counterpart on it are left out (`derive/ifx.py` names them).
- **Ours**: `derive/ifx.py` (the choice of the 17 types, the split into 11 fit types and 6 never-fitted ones, the
  requests: a practice question of the toolkit or one of 150 plain writing requests with one or two instructions;
  it is the file of our finished entry `0009591920-0008`, unchanged); `derive/reqs.py`, `derive/gen_own.py`
  (requests and their answers as runs of a toolkit case, sampled by the toolkit's generator), `derive/read.py`
  (the layer rule, the two decisions), `derive/report.py`, `derive/run.py`; the design of stage C as the first
  instruction stage of a model: toolkit requests and further types fitted together from the start, the hold on the
  start's own answers to six other sources, and the read on never-fitted types.
