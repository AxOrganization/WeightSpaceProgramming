# Runbook

Commands for `../METHOD.md`. They run in the sandbox, in the work directory, with this submission copied there as
`submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts from the
entry's parent (the start whose role in `task.json` is "parent"), not from any other start.

One script drives every GPU step through the queue, in order, and uploads twice: `derive/run.py`. Your part is to
start it, to watch it, to mend what stops it, and to write the report. Times are wall-clock on four GPUs:

| stage (marker in `mi_work/`) | what | about |
|---|---|---|
| `done.setup` | case, check sets, id lists, the requests made in the run | 10 min |
| `done.c` | the start's answers to the direction pool; the directions; model C; **upload 1 (C)** | 35 min |
| `done.gen` | model C's answers: requests, practice requests, tether rows | 60 min |
| `done.ipairs` | I: answers re-sampled after the same thinking, and mends | 35 min |
| `done.ifitq`, `done.a` | I: four fits, export; **upload 2 (A = C + I)** | 55 min |
| `done.screen` | the start, C and A on three check sets (only with 35 minutes left) | 20 min |

That is about 3 h 35 min. The script reads the minutes that are left from `LEFT_MIN`; if the run is late, upload 1
stands. Every upload is a complete, checked edit. Jobs carry the environment of the shell that queued them
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
  (the directions' pairs and stability, the follow rates of the requests per kind and per instruction, the mends per
  instruction, the fits' last lines, the final screen).
- The line `UPLOADED C` or `UPLOADED A` means that edit passed the dry run and `axonry-check` and is uploaded. The last
  such line names what the run delivers.
- `RUN DONE (last upload: X)` ends the script.

## When something stops
- **A job failed** (the script prints the job's name and stops): `$Q log NAME 30`, mend the cause, start the script
  again with the same command and a new `LEFT_MIN`. Finished stages are skipped, finished files are not made again.
- **`the fit of I held ... tether rows`**: the fit ran without the answers that hold the other tasks in place. Do not
  deliver it: read `mi_work/keep.jsonl` and the start of `$Q log if_1 400`, mend the cause, remove `mi_work/fits/I*`
  and `mi_work/done.ifitq`, and start the script again.
- **A generation job does not finish.** A job whose log already ends with its `[gen] N answers ...` line but that stays
  in `$Q status` for more than a few minutes has a vLLM engine that hung at shutdown (the GPU shows memory in use at
  0%): stop that engine process (`ps -eo pid,args | grep "VLLM::EngineCore"`, the one on that job's GPU; never a
  pattern that matches your own shell). The job then writes its rows and ends.
- **A job says its GPU is busy** right after another job ended there: `$Q cancel NAME` if it is still listed, wait a
  minute, start the script again.
- **`no stable direction`**: the directions of this start did not pass their stability test; the method does not
  apply. Say so in the report and deliver nothing.
- **An upload did not go through** (the script says so and goes on): read the message of `python -m wsp.upload` or
  `axonry-check` in `mi_run.log`. If it is a payload or change limit, deliver the edit before it (it is uploaded
  already) and say so. If it is the connection, run the three steps of `deliver` in `run.py` by hand for that edit.
- Never start a second copy of the script while one runs (`pgrep -f derive/run.py`), and start no GPU job of your own
  while it runs: the queue is full by design. Stop it (its pid) before you change anything it reads, then start it again.

## The report
REPORT.md, from `mi_work/NUMBERS.md` and `mi_run.log`: which edit was delivered (C or A) and why; the directions
(pairs, stability at their layers); what model C follows of the requests made in the run, per kind and per
instruction; the mends kept per instruction; the pairs and the last lines of the fits of I; the final screen (start,
C, the delivered edit) if it ran; what failed or was left out.
If the final screen puts the delivered edit's puzzles more than 3 points below the start's, say so plainly: the
method did not hold the start's puzzle skill here. Write REPORT.md before the last upload if you can (the script
writes a stub if there is none), and upload it with the delivered edit once more at the end:
```
python -m wsp.upload mi <C|A> --task task.json --report REPORT.md
```
