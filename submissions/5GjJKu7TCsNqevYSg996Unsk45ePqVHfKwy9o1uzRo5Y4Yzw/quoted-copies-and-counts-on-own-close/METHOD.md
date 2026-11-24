# Quoted copies, the instruction in front, and counts: two rounds of mended instruction pairs on the line that closes its own thinking, held to the parent's own answers

This entry names the finished entry `0009576150-0005` (own-close-carried-as-change) as its parent in `entry.json` and
**starts from it**. It inherits that model whole (its rules that close a thinking in time, its answer round, and the
champion's procedure workings under them). None of it is derived again or delivered again. The entry adds one layer
and delivers only that layer:

- **R, a preference fit on instruction requests of the kinds the parent still fails**, fitted with a keep term on the
  parent's own right answers. Two rounds of requests are recombined from the practice sets' own material:
  - round 2: the request to repeat with its own words in double quotes (the copy must keep the marks), the repeat
    instruction placed in front of the request as well as behind it, and requests with one counting instruction (a
    letter, capital words, a keyword count, keywords to include, forbidden words), most of them "fewer than N";
  - round 3: repeat requests with the instruction in front, "the letter X fewer than N times" with common letters and
    small N, and "at least N words" on practice requests.
  The parent answers every request four times. For two answers of each request the thinking is kept and four new
  answers are sampled after it: a passing and a failing answer after the same thinking make a sampled pair. A failed
  answer is mended by rule where the instruction has a mechanical remedy (the exact request put where a careless copy
  stands, the instruction sentence copied by mistake cut, a letter or keyword brought under or up to N, the text put in
  capitals, ...), graded again by the toolkit's checker and, if it now keeps every instruction, paired with the answer
  it was made from. One preference fit of about 650 MLP neurons on both rounds' pairs carries a KL term (weight 100) on
  the parent's own right answers to requests, word problems, maths, multiple choice and practice puzzles of every
  family; the parent's own rewritten units are never chosen. Half the fitted change is delivered, as int8 directions.

Where it comes from, plainly.

- `derive/`: the instruction round of the finished entry `0009565562-0012` (mended pairs): `genreq.py`,
  `branch.py`, `branch_req.py`, `mend_pairs.py`, `pipeline.py`, `train_req.py`, `train_sparse.py`, `avg_sparse.py`,
  `dose_sparse.py`, `stack.py`, `null_edit.py`, `cmpreq.py`, `phases.py`, `dirs.py`, `mkpool.py`, `take_unit.py`
  (the branching and fit scripts are those of `0009550031-0005`, the recombined requests of `0009559255-0006`).
  **Ours**: the round-2/3 kinds in `mkreq.py` (quoted words, instruction in front, the "freq", "lett" and "long"
  kinds), the new remedies in `repair.py` (exact copies in place of careless ones, letters and keywords brought under
  or up to N, keywords to include, forbidden words taken out), and `run.py`.
- `keep/`: the keep term and its rows from the finished entry `0009570269-0024` (procedure-workings): `train_keep.py`,
  `train_sparse.py`, `xitems.py`, `keep_rows.py`, `reqgen.py`, `practice_puzzles.py`, `rollouts.py`, `x_rollouts.py`,
  `encode_dirs.py`, `dose_sparse.py`, unchanged. `start_neurons.py` is ours (the parent's absolute units to keep out).

## 1. Why this layer, on this parent

On the public instruction benchmark the parent's family loses most on a few mechanical classes: a request to repeat
that holds double quotes (the copy turns `"` into `'`: the sampler's presence penalty hits the marks the thinking
already wrote), a repeat instruction placed in front of the request (the answer starts with the instruction), and
counting instructions ("the letter e fewer than 3 times", "at least 300 words"). The practice sets' own requests never
show these forms, so the published rounds never learned them. (These classes were read off the published per-question
results of this line; no benchmark item, text or answer is used in the run: every request is recombined from practice
material, and nothing is tuned on held-out data.) In our rehearsals on a start of this line the layer
moved the repeat class from 56% to 70% (with quotes 27% to 59%), "letter at least" 66% to 81%, "at least N words" 65%
to 74%, and the instruction benchmark by about +1.7 points, with maths, word problems, multiple choice and puzzles
unchanged within their errors. **Without the keep term the same pairs break the model** (at a larger dose answers
start with a second "Thinking Process", -6 points): never drop it.

## 2. Setup (first 10 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This entry
starts from its parent `0009576150-0005`. On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive submission/keep sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/     # the start (its own parents are inside it)
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/NOTES.md`. Use only `practice:` data and the
models' own generations; never load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
`task.json` holds the upload credentials: keep its contents out of logs, PROGRESS.md and the report.

## 3. Steps

All GPU work is one detached script. Start it once and look at it every 15 minutes:

```
nohup python derive/run.py > run.log 2>&1 &
grep '^\[run\]' run.log | tail                      # its own progress lines
```

| stage | what | measured (4 x 24 GB) |
|---|---|---|
| setup | cases `w2` and `mp` on the parent, 3,000 requests of `keep/reqgen.py`, 11,400 practice puzzles | 5 min (CPU) |
| keep | the parent's answers to 5,000 keep prompts (x 1); `keep_other.jsonl` (about 800 rows), `keep_held.jsonl`; `START.json` | 15-20 min |
| round 2 | 1,560 requests; the parent's answers x 4; same-thinking answers; mended pairs (`mp/logs/mend.log`) | 30 min |
| round 3 | 705 requests; the same | 15 min |
| fit | about 1,000 pairs + keep rows, 4 GPUs, 2 epochs | 35-45 min |
| edit | half dose `KRh`, full dose `KR`; int8 forms `KRhD`, `KRD` | 2 min |

- `run.py` skips every stage whose output exists: if a stage failed (its log tail is printed), fix the cause and start
  the same command again.
- Write in NOTES: the keep rows per source, the mended pairs per instruction of each round (`mp/logs/mend.log`,
  `mp3/logs/mend.log`; ours: 344 and 398, about 150 and 220 of them repeats), the fit's losses and payload
  (`w2/logs/fit_KR.log`).
- **Few pairs**: fewer than 300 pairs of rounds 2 + 3 together: deliver anyway (the layer is the entry), say so.

**Deliver the half dose `KRhD`** (the int8 form of half the fitted change). It is the edit our rehearsals measured.

```
python -m wsp.upload mp KRhD --task task.json --dry-run --report /workspace/work/NOTES.md
axonry-check $WSP_WORK/deliver/mp__KRhD/edit --task task.json
python -m wsp.upload mp KRhD --task task.json --report /workspace/work/REPORT.md
```

Write NOTES.md and REPORT.md before these commands (both need their report file to exist). If time is left, read
the edit on held-out requests against the parent (two draws: the null edit of `derive/null_edit.py --case mp`):

```
python derive/mkreq.py --case mp --side dev --out mp/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90
python derive/null_edit.py --case mp
python derive/genreq.py --case mp --model base --items mp/req/dev.json --out mp/req/dev.base.jsonl --n 2 --salt mr2
python derive/genreq.py --case mp --model NULL --items mp/req/dev.json --out mp/req/dev.NULL.jsonl --n 2 --salt mr2
python derive/genreq.py --case mp --model KRhD --items mp/req/dev.json --out mp/req/dev.KRhD.jsonl --n 2 --salt mr2
python derive/cmpreq.py mp/req/dev.json --base mp/req/dev.base.jsonl mp/req/dev.NULL.jsonl --edit KRhD=mp/req/dev.KRhD.jsonl
```

and put the table in the report. Nothing is tuned on it. Run `axonry-done` when nothing useful is pending.

## 4. What to expect, and what not to do

- Payload about 2.0 MB (int8 directions of about 650 neurons; limit 4,194,304 B); global relative change about 0.3%.
- Held-out requests: above both draws of the parent, most of it on the repeat requests (quoted, instruction in front)
  and the counting kinds; the other kinds within their errors.
- **Do not drop the keep term, do not raise the dose, do not deliver the full dose `KRD`** (our rehearsals: the full
  dose of a plain preference fit costs instruction following elsewhere; the half dose with the keep term is the
  measured edit).
- Do not fit the layer on another model than the parent's: the pairs are the parent's own answers after its own
  thinking, and the keep rows are its own right answers.

## 5. Rules kept

Practice sets only, and the models' own generations. The requests are recombined from the shipped practice requests
and problems; no instruction is written or generated. A mended answer is the model's own answer with the request's own
material put in, its own text rearranged or cut, graded by the toolkit's checker. Nothing is tuned on held-out data. No
arrays are shipped.
