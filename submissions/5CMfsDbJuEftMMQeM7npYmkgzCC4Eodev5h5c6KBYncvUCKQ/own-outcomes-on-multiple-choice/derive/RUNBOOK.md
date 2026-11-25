# Runbook

Commands for `../METHOD.md`. They run in the sandbox, in the work directory, with this submission copied there as
`submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts from the
entry's parent (the start whose role in `task.json` is "parent"), not from any other start.

One script drives every step, in order, and uploads once: `derive/run.py`. Your part is to start it, to watch it, to
mend what stops it, and to write the report. Times are wall-clock on four GPUs:

| stage (marker in `mi_work/`) | what | about |
|---|---|---|
| `done.setup` | case on the parent, check sets, id lists, the drawn puzzles of the anchors | 10 min |
| `done.gen` | the parent's answers through the queue: six to each of the 1,560 multiple-choice questions of the pool (four jobs), one to 400 problems, 1,150 instruction requests and 150 puzzles (the anchors) | 65 min |
| `done.c` | the anchors, the pairs, the parent's neurons, the neuron choice (job `csel`), then four fits queued (`cf_0`, `cf_14`, `cf_7`, `cf_21`: one GPU each, side by side) | 10 min |
| `done.f` | the four fits (about 55 minutes: some 690 steps each, a `[train] step N/M` line every few steps), their average at a quarter of its size in int8 form on the parent, the dry run and the check (about 10 minutes); **the upload (F)** | 70 min |
| `done.screen` | the parent and F on 256 held-out multiple-choice questions (x 2) and on the dev sets (only with 40 minutes left) | 25 min |

That is about 3 h; F is uploaded about 2 h 35 min in. The script reads the minutes that are left from
`LEFT_MIN`. The upload is a complete, checked edit. Jobs carry the environment of the shell that queued them
(`PYTHONPATH`, `WSP_CASES`, `WSP_WORK`, `HF_HUB_OFFLINE`, `HF_HOME`), so start the script from a shell with the
sandbox's settings.

## Start
```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py" PYTHONDONTWRITEBYTECODE=1
LEFT_MIN=<minutes of the run that are left, minus 15> nohup python $D/run.py > mi_run.log 2>&1 &
```
`LEFT_MIN` is the only thing to work out: the time the brief gives until the deadline, minus 15 minutes of reserve.

## Watch (every 15 minutes)
```
tail -5 mi_run.log; $Q status; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
- `mi_run.log` has one line per stage with the minutes left; `mi_work/NUMBERS.md` collects the numbers for the report
  (the parent's neurons left out; the questions with both outcomes, the pairs and anchors; the neurons chosen; the
  fits' lines; the int8 form's payload; the final screen).
- During `done.gen` the four pool jobs (`cp_0` ... `cp_3`) are the long ones: about 2,300 answers each, the long
  answers end last.
- `$Q log cf_0 5` shows a fit's `[train] step N/M` lines and its held-out loss every 50 steps. That loss (the
  likelihood of held-out right answers) rises a little while the preference is learned (0.30 to 0.32 in our run):
  that is expected, not a fault.
- After `F = the parent + C ...` the dry run and `axonry-check` build the edited model on the CPU (some 1,900 small
  ops): about ten minutes in which `mi_run.log` stands still. That is expected.
- `UPLOADED F` means the edit passed the dry run and `axonry-check` and is uploaded.
- `RUN DONE (last upload: F)` ends the script.

## When something stops
- **A queue job failed** (the script prints the job's name and stops): `$Q log NAME 30`, mend the cause, start the
  script again with the same command and a new `LEFT_MIN`. Finished stages are skipped, finished files are not made
  again.
- **`C left out: N pairs`**: the pool gave fewer than 80 pairs (the parent hardly ever answers one question both
  right and wrong); nothing is delivered. Say so.
- **A generation job does not finish.** A job whose log already ends with its `[gen] N answers ...` line but that
  stays in `$Q status` for more than a few minutes has a vLLM engine that hung at shutdown (the GPU shows memory in
  use at 0%): stop that engine process (`ps -eo pid,args | grep "VLLM::EngineCore"`, the one on that job's GPU; never
  a pattern that matches your own shell). The job then writes its rows and ends.
- **A job says its GPU is busy** right after another job ended there: `$Q cancel NAME` if it is still listed, wait a
  minute, start the script again.
- **One fit fails and the others end**: the script averages the fits that are there (two at least) and goes on; say
  in the report which seeds were averaged.
- **Every fit runs out of memory** (`CUDA out of memory` in `$Q log cf_0 30`): stop the script, lower `--max-len 20480`
  and `--rejected-window 20480` of the fit in `run.py` to `16384` (pairs with a longer answer are then left out),
  delete `mi_work/done.c`, start the script again.
- **The upload did not go through** (the script says so and goes on): read the message of `python -m wsp.upload` or
  `axonry-check` in `mi_run.log`. If it is the connection, run the three steps of `deliver` in `run.py` by hand.
- Never start a second copy of the script while one runs (`pgrep -f derive/run.py`), and start no GPU job of your own
  while it runs. Stop it (its pid) before you change anything it reads, then start it again.

## The report
REPORT.md, from `mi_work/NUMBERS.md` and `mi_run.log`: the parent; the pool (questions, answers, how many questions
have a right and a wrong finished answer); the pairs and anchors; the neurons left out and chosen; the four fits'
examples and held-out losses; the dose; the int8 form (payload, how far the encoded change is from the wanted one);
the final screen if it ran (the parent and F on the held-out multiple-choice questions and on the dev sets, each with
its error); what failed or was left out. If the final screen puts F's dev puzzles, dev math or `tuluif` more than 3
points below the parent's, say so plainly. Write REPORT.md before the upload if you can (the script writes a stub if
there is none), and upload it with the delivered edit once more at the end:
```
python -m wsp.upload mi F --task task.json --report REPORT.md
```
