# Runbook

Commands for `../METHOD.md`. They run in the sandbox, in the work directory, with this submission copied there as
`submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts from the
entry's parent (the start whose role in `task.json` is "parent"), not from any other start.

One script drives every GPU step through the queue, in order, and uploads once: `derive/run.py`. Your part is to
start it, to watch it, to mend what stops it, and to write the report. Times are wall-clock on four GPUs:

| stage (marker in `mi_work/`) | what | about |
|---|---|---|
| `done.setup` | case on the parent, check sets, id lists, 1,800 puzzles drawn from the toolkit's program | 10 min |
| `done.gen` | the parent's answers: 600 math problems x 4 (full limit), the puzzles x 4, instruction requests x 1 | 55 min |
| `done.z` | the pairs, the parent's neurons, the neuron choice; the two fits queued | 15 min |
| `done.f` | the two fits (one GPU each), their average at 0.25 in int8 form on the parent; **the upload (F)** | 60 min |
| `done.screen` | the parent and F on three check sets (only with 35 minutes left) | 20 min |

That is about 2 h 40 min. The script reads the minutes that are left from `LEFT_MIN`. The upload is a complete,
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
tail -5 mi_run.log; $Q status; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
- `mi_run.log` has one line per stage with the minutes left; `mi_work/NUMBERS.md` collects the numbers for the report
  (the parent's neurons left out; the math and puzzle pairs and anchors; the neuron choice; the fits' lines; the int8
  form's payload; the final screen).
- `UPLOADED F` means the edit passed the dry run and `axonry-check` and is uploaded.
- `RUN DONE (last upload: X)` ends the script.

## When something stops
- **A job failed** (the script prints the job's name and stops): `$Q log NAME 30`, mend the cause, start the script
  again with the same command and a new `LEFT_MIN`. Finished stages are skipped, finished files are not made again.
- **A fit runs out of GPU memory** (`CUDA out of memory` in its log): in `run.py` lower `--max-len 20480` to `16384`
  (longer examples are then left out), remove `mi_work/fits/Z*` and `mi_work/done.z`, and start the script again.
- **`Z left out: N pairs`**: the parent no longer gives both outcomes on these problems; nothing is delivered. Say so.
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
  while it runs: the queue is full by design. Stop it (its pid) before you change anything it reads, then start it again.

## The report
REPORT.md, from `mi_work/NUMBERS.md` and `mi_run.log`: the parent; the neurons left out; the math pool and the puzzle
pool (problems with both outcomes, pairs, anchors); the neuron choice; the fits' examples and held-out loss at start
and end; the int8 form (payload, how far the encoded change is from the wanted one); the final screen (the parent and
F) if it ran; what failed or was left out. If the final screen puts F's puzzles more than 3 points below the parent's,
say so plainly. Write REPORT.md before the upload if you can (the script writes a stub if there is none), and upload it
with the delivered edit once more at the end:
```
python -m wsp.upload mi F --task task.json --report REPORT.md
```
