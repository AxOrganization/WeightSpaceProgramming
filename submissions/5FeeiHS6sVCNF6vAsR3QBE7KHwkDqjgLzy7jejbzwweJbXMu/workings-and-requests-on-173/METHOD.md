# On the published 0009598520-0002: a puzzle layer of written workings and an instruction layer of answer preferences, side by side

## Credits
Nearly everything this entry runs is published, finished work, credited here by entry. Each reused directory says so in its
`README.md`, and its changes are listed below.

- **The puzzle layer W** is the chain of **#156 "workings-and-maths-recall-on-130" (0009580621-0022)**, run as published,
  in the copy that **#168 "workings-and-requests-on-164" (0009588957-0032)** also ran (`derive/w156/`).
- **The instruction layer I** is the instruction stage of **#168 (0009588957-0032)**: its `derive/inst/`, here
  `derive/i168/`. It brings #168's own request types with their checkers, branches after `</think>`, and preference fits
  of unit and head gains with a keep term.
- **The start** is our published entry **#173 "finish-after-the-loop-edit" (0009598520-0002)**: the toolkit's loop edit
  with #8's finish fit on top, on Qwen3.5-2B (`entry.json` names it as the parent). **Stage A** below runs only if the task
  offers no start at all: it is the toolkit's loop edit with the toolkit's own commands (`playbook.md`, "The loop edit"; `cases/qwen35-2b-think_loop`), first entered as **#2 "loop-removal"
  (0009531781-0011)**.
- **Ours:**
  - the driver `derive/run_v2.py`: it takes the task's parent start (0009598520-0002), or makes stage A itself when the
    task offers none;
  - running W and I side by side on two GPUs each, with I waiting for W and folding it in;
  - the floor-first uploads;
  - the transfer to Qwen3.5-2B:
    - the profile is taken from `WSP_PROFILE` instead of the 0.8B constant;
    - the payload is checked against this base's limit (8 MiB), and `w156/train_sparse.py` reads the limit as the
      current toolkit states it (`max_payload_bytes_per_hidden` x the base's width);
    - the instruction stage uses 600 requests and two fits;
    - the chain runs at `CHAIN_SCALE` 0.7 (its own knob: 70 % of its keep lists and drawn puzzles), with lower token caps
      for the keep answers (8,192 / 12,288);
    - so both layers fit on two GPUs each within the run;
  - `W_ONLY` in `w156/chain.py`: it skips #156's maths layer R and its check reads (time), and gives W's neuron choice and fit
    both GPUs of the chain.

## What fails, on any 2B start with the loop removed
After the loop edit, the 2B model finishes most answers (the toolkit's case `cases/qwen35-2b-think_loop`), and what is
still lost has two clear forms:

1. **Puzzles that need a procedure written out:** sorting words letter by letter, ordering objects from clues, tracing
   a path, reading shapes from SVG paths.

   The model finishes them, but it improvises the procedure each time: long, uneven thinking, and wrong more often than
   the steps would allow. #156 found the same on the 0.8B line and fixed it by teaching the model to write a short,
   program-written working for six such kinds of the toolkit's `bbhsynth` practice set. On #156's practice test puzzles
   (never fitted) that layer raised accuracy and shortened the thinking, and held the other tasks.

2. **Instruction requests whose visible answer breaks the request after a correct thinking:** a closing remark after
   "end with the word X", a heading above a fixed first word.

   #168 showed that, after one and the same thinking, new visible answers sometimes keep the request and sometimes
   break it. That is a clean contrast, and a preference fit of per-unit and per-head gains moves it, held by a keep term
   on the start's own answers to other tasks.

Neither failure is a loop. So the two layers address what a loop edit leaves and do not repeat it. They act through
different parameters: W writes int8 directions on a few hundred MLP rows chosen on the workings; I scales whole units
and heads. W never chooses a unit the start's own ops write (`w156/neurons_of.py`).

**Measure both before trusting them, on this start:**
- the chain's prep and keep steps print, per source, the start's accuracy and cut share;
- the instruction stage prints how often the start's answers keep their requests, and how many starts give both
  outcomes after one thinking (`ri_work/NUMBERS.md`).

If the instruction stage finds fewer than 30 starts with both outcomes, it has nothing to learn from. Then stop it before
its fits (`python derive/i168/run_i.py --status`; stop its processes by PID), let Fw stand, and say so in the report.

## Running it (4 x RTX 4090; the driver reads the time left from task.json)
The sandbox sets the toolkit, the base and the environment (`WSP_WORK`, `WSP_CASES`, `WSP_MODELS_DIR`, `PYTHONPATH`); do
not override them. Copy `submission/`, `task.json` and the task's start directory (`0009598520-0002/`, with its chain,
as the bundle has it) to `/workspace/work/`, then:

```
cd /workspace/work
nohup python /workspace/work/submission/derive/run_v2.py > /workspace/work/run_v2.log 2>&1 &
grep PHASE /workspace/work/run_v2.log            # progress
tail -n 30 /workspace/work/v2/chain.log /workspace/work/v2/inst.log /workspace/work/ri_work/NUMBERS.md
```

| step | GPUs | what |
|---|---|---|
| start | CPU | the start = the task's start with role `parent` (entry.json: 0009598520-0002) |
| a | 4 | **only if the task offers no start:** the toolkit's loop edit on the base (case `lp`: the base's answers on 128 practice problems, pairs, direction, layers 15 and 19, top-64 writers, dose 6, with its random control). Written as `parent/edit/` and uploaded at once (`wsp.upload lp A6`). The later layers stand on it, and `wsp.upload` carries its ops into their upload |
| cases | CPU | case `pz` on the start |
| wi, GPUs 0,1 | 2 | `w156/chain.py --case pz --until edits` with `W_ONLY=1`: prep, keep, W's neuron choice and fit, `WD` (int8 directions), `Fw`. **Uploaded as soon as it exists** (`wsp.upload pz Fw`) unless FI is already up |
| wi, GPUs 2,3 | 2 | `i168/run_i.py`: our requests, 4 answers each, 4 branches after every finished thinking, 2 preference fits, `I` exported at half strength; it waits for `Fw` and delivers **FI = Fw's ops + I's ops**, one edit on the start, after its dry run, payload and `axonry-check`. If the chain cannot finish in time, it delivers I alone on the start |

Each step leaves a marker in `/workspace/work/v2/` and is skipped when the driver is started again. The chain's and the
instruction stage's own notes apply inside `wi`:
- `python derive/w156/chain.py --case pz --out /workspace/work/pz --status`;
- `python derive/i168/run_i.py --status`;
- a generation step that hangs is restarted by the chain;
- never `pkill -f` a pattern; stop a job by its PID.

Whatever runs late gives way to the upload. FI stands only if it passed its checks; otherwise Fw stands; otherwise
stage A or I alone.

## What to expect, and what to report
- **The puzzle kinds with a working:** shorter, more accurate thinking. Also the kept tasks (maths, word problems,
  multiple choice) near the start.
- **Instruction requests of #168's held-out types:** a higher share kept than the start's (the stage reads it,
  `held-out kinds` in `NUMBERS.md`).
- **Report every measured number with its practice source, and every stage that was skipped or held back with the
  reason.**
- **When time is left after FI is up:**
  - read the start and FI on the chain's held-out test puzzles (`pz/pz_test.jsonl`, the last 32 of every kind) and on a
    few hundred fresh practice problems of the kept sources;
  - if a kept source loses beyond noise, upload Fw or I alone instead, whichever does not lose.
