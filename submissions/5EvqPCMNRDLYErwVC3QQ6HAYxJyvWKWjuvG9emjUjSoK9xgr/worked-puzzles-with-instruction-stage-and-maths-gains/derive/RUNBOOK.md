# Runbook

Commands for `../METHOD.md`. They run in the sandbox, in the work directory, with this submission copied there as
`submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts from the
start whose role in `task.json` is "parent" (the champion of the line when the run started); when `task.json` names
no start, it starts from the base model.

One script drives every step and uploads twice (the floor A + P, then F, which replaces it): `derive/run.py`. It
starts the published instruction stage (`derive/i168/run_i.py`) beside itself on GPUs 2 and 3 and keeps its own jobs
(the toolkit's queue) on GPUs 0 and 1. Your part is to start it, to watch both, to mend what stops them, and to write
the report. Times are wall-clock on four RTX 4090, for Qwen3.5-2B:

| stage (marker in `mr_work/`) | what | about |
|---|---|---|
| (first minute) | the instruction stage is started on GPUs 2, 3 (`inst.log`; its own markers in `ri_work/`: setup, gen, branch, fits, edit; `upload` and `read` are marked done beforehand and not run) | 2 h to 3 h 30 min in all (our estimate) |
| `done.setup` | case on the start, test lists, keep pool and holds, the workings of the twelve puzzle kinds (about 11,000 puzzles, each solved by its writer) and 120 fresh puzzles of each kind, the start's model (built on the CPU) | 10 min |
| `done.own` | the start's own answers through the queue (GPUs 0, 1): the fresh puzzles (`own_rp0`, `own_rp1`, at most 1,152 tokens), maths and word problems at the full limit (`own_hm`, `own_hg`), the maths pool twice at the full limit (`own_ma0`, `own_ma1`), the keep pool (`own_keep0`, `own_keep1`), the start on the test puzzles and on 30 more fresh puzzles of each kind with a writer (`rd_base_bbh0`, `rd_base_bbh1`, `rd_base_dp`) | 75 min |
| `done.fitaq` | layer A: the share of the start's own maths answers left unfinished; if it is 5% or more, three fits of the toolkit's fitter are queued (`fita1` to `fita3`, one after the other on the GPU P's fit leaves free) | 1 min |
| `done.fitp` | layer P: which kinds get workings (the start's own test puzzles at the full limit), the rows, the fit (job `fitp`: one GPU, 1,000 to 1,500 steps of about 2 s), the export | 50 min |
| `done.floor` | layer A's fits averaged (`python -m wsp.fit spec mr A --dose 1.5`), A + P joined, the dry run and the check; **the first upload (the floor, `UPLOADED PA`; `UPLOADED P` when layer A was not fitted)** | 5 min |
| `done.base` | the start on the test lists (`rd_base_*`), queued; the script then waits for the instruction stage | (waits) |
| `done.f` | the instruction stage's layer (`ri_work/done.edit`, edit `Ic` of case `ri`) copied into case `mr`; F = A + I + P joined as one edit on the start, the dry run and the check; **the second upload (F, in place of the floor)** | 10 min |
| `done.read` | F's model, then F on the test lists (`rd_F_*`) | 45 min |

That is 3 h 30 min to 4 h 40 min; the floor is uploaded about 2 h 25 min in, F as soon as the instruction stage's
layer is there too (2 h 40 min to 3 h 50 min in). The script reads the
minutes that are left from `LEFT_MIN` and waits for the instruction stage until 25 minutes are left; **if the stage
has not exported its layer by then, the floor stands** (the script says so and goes on to the readings of the
floor). Jobs carry the environment of the shell that queued them (`PYTHONPATH`, `WSP_CASES`, `WSP_WORK`,
`HF_HUB_OFFLINE`, `HF_HOME`), so start the script from a shell with the sandbox's settings.

## Start
```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py" PYTHONDONTWRITEBYTECODE=1
LEFT_MIN=<minutes of the run that are left, minus 15> nohup python $D/run.py > mr_run.log 2>&1 &
```
`LEFT_MIN` is the only thing to work out: the time the brief gives until the deadline, minus 15 minutes of reserve.
Start no job of your own on any GPU while the script runs: GPUs 0 and 1 are the queue's, GPUs 2 and 3 the
instruction stage's (one process a GPU; a second process on one of them runs it out of memory).

## Watch (every 10 to 15 minutes)
```
tail -5 mr_run.log; $Q status; python $D/i168/run_i.py --status; tail -3 inst.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
- `mr_run.log` has one line per stage with the minutes left; `mr_work/NUMBERS.md` collects the numbers for the report
  (the lists; the workings per kind; the start on the test puzzles of the kinds with a writer and which get
  workings; the rows of layer P; the units chosen per layer; the fit's held-out loss; layer A's pool and fits; each
  edit's ops and payload; the final comparisons). The instruction stage writes its own numbers to
  `ri_work/NUMBERS.md` (how many of the start's answers follow its requests, how many requests give both outcomes,
  the preference pairs, the held-out kinds) and its jobs' logs to `ri_work/logs/`.
- `done.setup` and the two models (the queue's and the instruction stage's copy of the start) are built on the CPU:
  minutes in which the logs stand still. Expected.
- `$Q log fitp 5` shows the fit's `step N train nll (ema)` lines and its held-out loss; `$Q log fita1 5` shows a fit of
  layer A (it prints its example counts per source: maths under `nll`, every other type under `anchor`). A fit of
  layer A that ends with `gave no edit` is left out by the script; with fewer than three, the uploads go without A.
- `UPLOADED PA` (the floor) and later `UPLOADED F` mean that edit passed the dry run and `axonry-check` and is
  uploaded (each upload replaces the one before); `RUN DONE` ends the script. The instruction stage ends with
  `INSTRUCTION STAGE DONE` right after its step `edit`: its steps `upload` and `read` show as done from the start
  (`run.py` marks them, so that the stage uploads nothing itself and does not start its four-job reading on two
  GPUs). Do not delete those two markers.

## When something stops
- **A queue job failed** (the script prints the job's name and stops): `$Q log NAME 30`, mend the cause, start the
  script again with the same command and a new `LEFT_MIN`. Finished stages are skipped, finished files are not made
  again, and the instruction stage is left alone while it runs (its pid is in `mr_work/inst.pid`).
- **The instruction stage stopped** (`inst.log` ends with `step failed` or `a GPU job failed`, and
  `python $D/i168/run_i.py --status` shows a step still waiting): read the job's log in `ri_work/logs/`, mend the
  cause, then start it again by hand with the settings `run.py` gives it; its finished steps are skipped:
  ```
  touch ri_work/done.upload ri_work/done.read
  env GPUS=2,3 REH=1 WSP_PROFILE=qwen35-2b-think I_TRAIN=600 I_TOK=12288 I_SEEDS=51,52 PAYLOAD_MAX=8388608 \
      LEFT_MIN=<minutes left, minus 40> nohup python $D/i168/run_i.py --pz-case none >> inst.log 2>&1 &
  echo $! > mr_work/inst.pid
  ```
  `run.py` (waiting in its stage `f`) sees `ri_work/done.edit` and goes on. (Starting `run.py` again does the same
  by itself: it starts the stage when the stage is neither running nor done.) If `run.py` has already passed its
  stage `f` with the floor standing and the stage's layer is there after all, delete `mr_work/done.f` and
  `mr_work/done.read` and start `run.py` again: it folds and uploads F. If the stage cannot be mended in time,
  leave it: the floor stands, and the report says what stopped it. Its own notes (the header of `run_i.py`,
  `derive/i168/README.md`) apply to its steps.
- **A generation job does not finish.** A job whose log already ends with its `[gen] N answers ...` line but that
  stays in `$Q status` (or in `ri_work/logs/`) for more than a few minutes has a vLLM engine that hung at shutdown
  (the GPU shows memory in use at 0%): stop that engine process (`ps -eo pid,args | grep "VLLM::EngineCore"`, the one
  on that job's GPU; never a pattern that matches your own shell). The job then writes its rows and ends.
- **A job says its GPU is busy or runs out of memory at its start**: something else holds that GPU (`nvidia-smi`).
  The queue must have workers on GPUs 0 and 1 only while the instruction stage runs (`pgrep -af "queue.sh worker"`):
  stop a worker on GPU 2 or 3 by its pid, wait until the memory is free, start the script again.
- **The fit of layer P runs out of memory** (`CUDA out of memory` in `$Q log fitp 30`): stop the script, lower `MAXSEQ`
  in `run.py` (9600) to 7000 and `MAXP` (7600) to 5000, delete `mr_work/P.train.jsonl`, start the script again. (The
  fitter checkpoints its gradients, `--ckpt`: a row of 9,000 tokens took under 6 GB in our runs.)
- **The upload did not go through** (the script says so and goes on): read the message of `python -m wsp.upload` or
  `axonry-check` in `mr_run.log`. If it is the connection, run the three steps of `deliver` in `run.py` by hand.
- Never start a second copy of `run.py` while one runs (`pgrep -f derive/run.py`). Stop it (its pid) before you change
  anything it reads, then start it again; stopping `run.py` does not stop the instruction stage.

## The report
REPORT.md, from `mr_work/NUMBERS.md`, `ri_work/NUMBERS.md` and the two logs: the start; for P, the start on the test
puzzles of the kinds with a writer and which got workings, the workings per kind, the rows and keep rows, the units
per layer, the held-out loss; for A, the share of the start's own maths answers left unfinished, whether the layer was
fitted, each fit's example counts; for I, how many of the start's answers followed the stage's requests, how many
requests gave both outcomes, the preference pairs and fits, the held-out kinds (the start against I alone) if the
stage read them; each edit (ops, payload) and which upload stands; the comparisons if they ran (the start -> F, each
with its error): the practice requests by kind, the test puzzles by kind (read at the profile's full token limit, as
the start's), maths, word problems, multiple choice; what failed or was left out. If the comparisons put F more than
3 points below the start on maths, word problems or multiple choice, say so plainly. Write REPORT.md before the
upload if you can (the script writes a stub if there is none), and upload it with the delivered edit once more at
the end (the edit that `RUN DONE (last upload: ...)` names: `F`, or `PA` / `P` when the floor stands):
```
python -m wsp.upload mr F --task task.json --report REPORT.md
```
