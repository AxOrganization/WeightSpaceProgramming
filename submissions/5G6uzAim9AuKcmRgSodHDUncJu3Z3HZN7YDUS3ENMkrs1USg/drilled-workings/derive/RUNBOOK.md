# Runbook

Commands for `../METHOD.md`. They run in the sandbox, in the work directory, with this submission copied there as
`submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts from the
entry's parent (the start whose role in `task.json` is "parent"), not from any other start.

One script drives every step, in order, and uploads once: `derive/run.py`. Your part is to start it, to watch it, to
mend what stops it, and to write the report. Times are wall-clock on four GPUs:

| stage (marker in `mi_work/`) | what | about |
|---|---|---|
| `done.setup` | case on the parent, check sets, id lists; the drawn puzzles and word lists of the drill | 12 min |
| `done.gen` | the parent's answers through the queue: 800 keep puzzles x 2, instruction requests and calculation problems x 1 | 15 min |
| `done.drill` | the rows, the parent's neurons, **the fit of D on the four GPUs at once** (not through the queue) | 115 min |
| `done.f` | D in int8 form on the parent; **the upload (F)** | 5 min |
| `done.screen` | the parent and F on the dev sets, `tuluif` half B and the check puzzles (only with 35 minutes left) | 25 min |

That is about 2 h 55 min. The script reads the minutes that are left from `LEFT_MIN`. The upload is a complete,
checked edit. Jobs carry the environment of the shell that queued them (`PYTHONPATH`, `WSP_CASES`, `WSP_WORK`,
`HF_HUB_OFFLINE`, `HF_HOME`), so start the script from a shell with the sandbox's settings.

## Start
```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py" PYTHONDONTWRITEBYTECODE=1
LEFT_MIN=<minutes of the run that are left, minus 15> nohup python $D/run.py > mi_run.log 2>&1 &
```
`LEFT_MIN` is the only thing to work out: the time the brief gives until the deadline, minus 15 minutes of reserve.

## Watch (every 15 minutes)
```
tail -5 mi_run.log; $Q status; tail -c 300 mi_work/dfit.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
- `mi_run.log` has one line per stage with the minutes left; `mi_work/NUMBERS.md` collects the numbers for the report
  (the drawn puzzles with a working; the word lists; the rows; the parent's neurons left out; the fit's lines; the
  int8 form's payload; the final screen).
- While D is fitted (`D: fit on 4 GPUs`), the queue is empty and all four GPUs are busy; `mi_work/dfit.log` shows
  `[train] step N/M` (about 6,800 steps, one a second) and the held-out loss every 50 steps (it falls from about 0.4 to about 0.03).
- `UPLOADED F` means the edit passed the dry run and `axonry-check` and is uploaded.
- `RUN DONE (last upload: F)` ends the script.

## When something stops
- **A queue job failed** (the script prints the job's name and stops): `$Q log NAME 30`, mend the cause, start the
  script again with the same command and a new `LEFT_MIN`. Finished stages are skipped, finished files are not made
  again.
- **The fit on the four GPUs failed** (it is tried twice; then `D: the fit on every GPU failed twice`): read the end
  of `mi_work/dfit.log`. If it is `CUDA out of memory`: stop the script, lower both `--max-len 20480` and
  `--rejected-window 20480` of the D fit in `run.py` to `12288` (longer keep rows are then left out) and start the
  script again: the fit starts over on the four GPUs. Otherwise the script fits D by itself on one GPU through the
  queue (job `dfit`, about 5 h: it ends in time only when it starts in the first half hour); let it run, and say
  so in the report if the time ran out.
- **`D left out: N workings`**: the writers gave too few workings (the toolkit's puzzle program is not the one this
  was written for); nothing is delivered. Say so.
- **A generation job does not finish.** A job whose log already ends with its `[gen] N answers ...` (or
  `... answers, followed ...`) line but that stays in `$Q status` for more than a few minutes has a vLLM engine that
  hung at shutdown (the GPU shows memory in use at 0%): stop that engine process (`ps -eo pid,args | grep
  "VLLM::EngineCore"`, the one on that job's GPU; never a pattern that matches your own shell). The job then writes its
  rows and ends.
- **A job says its GPU is busy** right after another job ended there: `$Q cancel NAME` if it is still listed, wait a
  minute, start the script again.
- **The upload did not go through** (the script says so and goes on): read the message of `python -m wsp.upload` or
  `axonry-check` in `mi_run.log`. If it is the connection, run the three steps of `deliver` in `run.py` by hand.
- Never start a second copy of the script while one runs (`pgrep -f derive/run.py`), and start no GPU job of your own
  while it runs. Stop it (its pid) before you change anything it reads, then start it again.

## The report
REPORT.md, from `mi_work/NUMBERS.md` and `mi_run.log`: the parent; the drawn puzzles with a working, by kind, and the
word lists; the rows (workings by kind, keep rows by kind, protect rows); the neurons left out and chosen; the fit's
examples and held-out loss at start and end; the int8 form (payload, how far the encoded change is from the wanted
one); the final screen if it ran (the parent and F on the dev sets and on the check puzzles by kind); what failed or
was left out. If the final screen puts F's dev math or `tuluif` more than 3 points below the parent's, say so
plainly. Write REPORT.md before the upload if you can (the script writes a stub if there is none), and upload it with
the delivered edit once more at the end:
```
python -m wsp.upload mi F --task task.json --report REPORT.md
```
