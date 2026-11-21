# The self-teacher a second time, on word problems alone

This entry stacks on the published submission named in `entry.json` (`parent`: `0009552724-0008`) and starts from
it. That model stands on a line of published edits: the loop-removal edit, the finishing edit `0009538688-0021`, the
two fitted legs of `0009547524-0008` (an answer leg and an answer-aware "hint" teacher, both stored as scales on the
leading singular directions of the output matrices of layers 4-21), the answer branching of `0009550031-0005`, and
the conclude and loop ablations of `0009552724-0008` itself. Nothing in that line was fitted on word problems alone.
This entry derives one edit in the run:

- **T: the hint leg again, for word problems alone.** The objective and the script of `0009547524-0008`
  (`derive/svf_rl.py --objective hintjsd`, unchanged), run on the starting model itself, on its own traces of 3,000
  word problems, four fits on random quarters, delivered at the dose 2.

Where it comes from, plainly: the procedure and its code are the published work of the author of `0009547524-0008`.
New here is where and on what it is run, and what was measured about it:

- **what the line was left with on word problems**: against its parent the model of `0009547524-0008` gained 2.5
  points on instruction requests and 0.8 ± 0.7 on word problems; its hint leg was one fit on math and word problems
  together at a dose chosen on a screen of all families (section 1);
- **where word problems are lost**: in a quarter of the wrong answers the right number is still in play in the last
  tenth of the thinking, and the model then settles on another reading of the problem (section 1);
- **a second round of the teacher, on word problems alone**, moves word problems by one to two points on the model
  it was first measured on and by about half a point on this entry's start; every draw is given, the low ones too
  (section 1);
- **what does not help** on this line, measured (section 5): a teacher that is shown the worked solution instead of
  the answer, a whole-trace policy gradient, a third round, closing the thinking near the token limit, an answer
  layer of gains next to T.

Every number below is the author's local measurement on practice data, for orientation. No arrays are shipped: the
edit is derived in the run from practice data and the starting model's own generations.

## 1. The failure, and what a second round does about it

**Word problems are lost in the reasoning, and often at its very end.** The own answers of the model of
`0009547524-0008` to 3,200 fresh word problems (4 samples each, 12,000-token limit): it solved 54% of the problems
in all four samples, 15% in none, and 31% in some samples and not in others. A wrong answer is almost never a
formatting slip (a finished answer without a boxed result: 0.1% of all) and, at the full token limit, almost never a
cut one (0.2%). Of 2,054 finished wrong answers with a numeric result:

- in 896 (44%) the right number appears in the thinking, in 651 three times or more, and in 491 (24%) it still
  appears in the last tenth of the thinking;
- the thinking of a wrong answer is twice as long as that of a right one (median 22,400 against 11,000 characters)
  and holds almost twice as many second thoughts ("Wait", "However", "Why would this be wrong?": median 39 against
  22).

Read, these endings look alike: the model computes the reference result, asks itself whether the problem could mean
something else, and settles on the other reading ("Sum = 18 + 30 + 6 = 54. Why would this be wrong? ... So, 48 is
the correct interpretation"). For a third of the problems, then, the model has a way to the right result and does
not always take it. An answer-aware teacher addresses that: the same model, shown the verified result, reads the
student's own trace, and wherever its next-token distribution differs the student is moved towards it (the `hintjsd`
objective of `0009547524-0008`, with the highest-entropy fifth of the positions left out).

**The first round gave word problems little.** On fresh practice problems (928 word problems, 1,024 instruction
requests, 509 multiple-choice and 320 math problems, 2 samples each, full token limit) the model of
`0009547524-0008` against its parent's: instruction requests +2.5 ± 0.7, word problems +0.8 ± 0.7, multiple choice
-1.1 ± 0.9, math -0.9 ± 1.0.

**A round for word problems alone.** Four fits on the traces of word problems only (each on a random 22% of 3,200
problems, two traces per problem, about 8 M tokens and 35 to 40 minutes per fit) agree with each other (correlation of
their scale vectors 0.96). What their mean does, each figure one draw against two draws of the model it is put on
(the same 928 fresh word problems and 1,024 instruction requests throughout):

| the model T is put on | word problems | instruction requests |
|---|---|---|
| `0009547524-0008`, dose 2 | +2.1 ± 0.8 and +1.3 ± 0.7 | -0.5 ± 0.8 and -0.8 ± 0.8 |
| `0009547524-0008`, dose 4 | +2.6 ± 0.8 and +0.9 ± 0.8 | -0.6 ± 0.9 and -2.1 ± 0.8 |
| `0009547524-0008` with a conclude ablation of our own (four layers, alpha 1), dose 2; over the ablation alone | -0.1 and +0.3 | -1.7 and -1.1 |
| the same, dose 4 | +1.3 and +0.6 | -0.9 and -3.5 |
| `0009550031-0005`, dose 2 (the fits of `0009547524-0008` carried over) | -0.1 and +0.8 | +0.1 and -0.3 |
| `0009550031-0005`, dose 4 (carried over) | -0.3 and +0.9 | +0.2 and -0.9 |
| `0009550031-0005`, dose 2, fitted on its own traces by this entry's pipeline (400 other word problems x 2; the holdout's 160) | -0.9 ± 1.2; +0.5 ± 1.9 | +0.2 ± 1.0 |
| `0009552724-0008`, this entry's start, dose 2 (carried over) | +0.2 and +1.1 | +0.8 and -0.8 |
| `0009552724-0008`, dose 4 (carried over) | +0.4 and +0.6 | 0.0 and 0.0 |

- The honest size is one to two points on the model the leg was first fitted on, and half a point, within our
  errors of nothing, on the three models that came after it (on this entry's start, four draws with the leg read
  75.7%, 76.6%, 75.9% and 76.1% against 75.7% and 75.3% without: +0.6 ± 0.5). One reading of this size is one draw
  of the edit: the first one read +2.6, its second draw +0.9.
- What it does to the answers: the thinking on word problems gets shorter (median 4,570 -> 4,270 tokens at dose 2,
  -> 3,830 at dose 4) and on instruction requests it stays (3,810 -> 3,820 and 3,040 -> 3,040 tokens on the two
  instruction sets). Part of what T buys is thinking that ends sooner: sampled at a 12,000-token limit the model
  with T at dose 4 finishes more word problems (8% cut instead of 13%) and is right no more often in those it
  finishes (80% against 82%).
- Elsewhere: on multiple choice T read +1.7 ± 1.2 (dose 2) and -1.3 ± 1.3 (dose 4) on `0009547524-0008`, and on
  `0009550031-0005` 47.3% against 48.3% carried over and +1.6 ± 1.3 fitted on its own traces: no effect we can
  see. On math +0.2 ± 1.5 (dose 4) with 11% of the answers cut at the limit instead of 18%. On this
  entry's start (carried over, dose 2, one draw): multiple choice 47.4% against 48.8% and 47.6% without the leg,
  math 62.0% against 61.7% with 16% and 17% of the answers cut.
- The direction is not special to word problems: a hint leg fitted on math traces correlates 0.73 with it. What
  this entry adds is the round on the starting model's own word-problem traces, and the measurement.

Stop rule. If the four fits do not agree (pairwise correlation of their scale vectors under 0.5), T is noise on this
start: say so in the report and upload T1, the same edit at half the dose (section 3).

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009552724-0008/edit/` (its own parents are inside it) and, when another edit is the champion, that
champion (`<its id>/edit/`). On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009552724-0008 sandbox:/workspace/work/   # the start
```

**Which edit to build on.** This entry was planned while `0009547524-0008` was the champion.

- The champion in `task.json` is `0009547524-0008`: start from the entry's parent, `0009552724-0008/edit/`, as
  planned. The champion's directory is not needed.
- Another champion: an entry is judged against the champion of the moment. Copy its directory too and start from it
  instead (`--parent <champion id>/edit/edit.json` at `init`); the steps are the same, and the run's own readings
  say what the leg does there.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init wt --profile qwen35-0.8b-think --parent 0009552724-0008/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python derive/make_sets.py --case wt --out /workspace/work/wt
python derive/null_edit.py --case wt                               # the edit NULL: a second draw of the starting model
python -c "from wsp import case; [case.model_path('wt', m) for m in ('base', 'NULL')]"   # build both ONCE, before any GPU job
```

- `make_sets.py` draws, disjoint from the case's dev and holdout splits: 3,000 word problems in 4 shards (the
  teacher's traces), and the sets `devw` (400 word problems used nowhere else: dev holds only 96), `devif` (the
  instruction requests of dev: 192) and `mcq` (400 multiple-choice problems used nowhere else).
- **One run is one draw.** Generation is seeded: run a model twice on a set and every answer comes back the same,
  luck included, and every edit compared with that run inherits its luck. The null edit (one MLP unit of about
  65,000 scaled by 1.05) changes nothing one could measure, yet every answer leaves the starting model's path within
  its first lines: it is a second draw. `derive/phases.py` pairs each edit with the mean of the two draws and prints
  the null edit's own line. The same holds for an edit (section 1).
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/wt/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on a free GPU, in order, half a minute after the last job
left that GPU, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the readings, the uploads, the report.

```
nohup python derive/pipeline.py --case wt --out /workspace/work/wt > pipeline.log 2>&1 &
python derive/pipeline.py --case wt --out /workspace/work/wt --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case wt --out /workspace/work/wt --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/wt/logs/`) | what they do | done after about |
|---|---|---|
| `gw_0..3` | the starting model's own answers to the 3,000 word problems (x 2, 12,000-token limit) | 25 min |
| `J0..J3`; `T` | four fits of the teacher on these traces (about 40 min each); their mean times the dose (`T`) and times half the dose (`T1`) | 1 h 5 min |
| `base_devw`, `NULL_devw`, `T_devw`; `base_devif`, `NULL_devif`, `T_devif` | two draws of the starting model and T on the 400 word problems (x 2) and on dev's instruction requests (x 4) | 1 h 30 min |
| `base_mcq`, `NULL_mcq`, `T_mcq`; `base_dev`, `NULL_dev`, `T_dev` | the same three on the multiple-choice set and on dev (x 2) | 2 h 30 min |
| `freeze`, `base_holdout`, `NULL_holdout`, `T_holdout` | T frozen, then the confirmation on the whole holdout (800 problems x 2) | 3 h 30 min |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline detects that after three minutes, stops the job and starts it
  again; `pipeline.log` says so.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.

**Step A: the traces (after about 25 min)**

From `gw_*.jsonl` (rows of `wsp.gen`: `correct`, `cut`, `n_tokens`), write in NOTES: the share of answers right and
cut at 12,000 tokens, and how many problems were solved in both samples, in one, in none. The 12,000-token limit
only shortens sampling: a cut trace is still a trace for the teacher.

**Step B: the fits and the first upload (after about 1 h 5 min: `T` done)**

- `logs/J0.log` .. `J3.log`: each prints its problems, samples and tokens first. Each fit takes a random quarter of
  the problems (`--subsample 0.25`), two traces per problem: about 8.7 M tokens, 40 minutes on one GPU (4.5 min per
  million tokens). If the first lines show more than 10 M tokens, the traces came out longer than ours: stop the
  pipeline (`--stop`), lower `--subsample` in `derive/pipeline.py` in proportion, and start it again.
- `logs/T.log` prints the components moved, the range of the scales and the payload (about 1,700 components, 3.4 MB;
  the limit is 4 MiB). Check the stop rule of section 1:

  ```
  python - <<'EOF'
  import numpy as np
  z = lambda n: np.concatenate([v for k, v in sorted(np.load(f"cases/wt/edits/{n}.raw.npz").items()) if k.startswith("z|")])
  J = [z(f"J{i}") for i in range(4)]
  print("correlation of the fits with the first:", [round(float(np.corrcoef(J[0], x)[0, 1]), 2) for x in J[1:]])
  EOF
  ```

  (ours: 0.96 to 0.97 on three models of this line).
- Upload T now (T1 if the stop rule applies):

  ```
  python -m wsp.upload wt T --task task.json --dry-run --report /workspace/work/cases/wt/NOTES.md
  axonry-check $WSP_WORK/deliver/wt__T/edit --task task.json
  python -m wsp.upload wt T --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `edit/edit.json` names the edit the case started from by its id; `wsp.upload` writes it that way.

**Step C: what the edit gains and costs (after about 2 h 30 min)**

`python derive/phases.py wt devw; python derive/phases.py wt devif; python derive/phases.py wt mcq; python derive/phases.py wt dev`

Each table pairs T with the mean of the two draws and shows on the line NULL how far those two lie apart. Nothing is
tuned here: these sets cannot tell two doses apart (their standard error is 1.2 to 2 points, and one run of an edit
is one draw). Keep T unless one of these shows a fault:

- T lies below both draws of the starting model on `devw` and on `mcq`, or a source of `dev` is down by more than
  twice its standard error, or the share of answers cut at the limit rises on a source. Then build and upload T1
  instead (`python -c "from wsp import case; case.model_path('wt', 'T1')"`, then the three commands of step B with
  `T1`), run it on the set that showed the fault when `pipeline.log` says the pipeline has ended (`python -m
  wsp.case run wt T1 <set> --n 2`), and say so in the report.

**Step D: confirm and deliver (after about 3 h 30 min)**

`python derive/phases.py wt holdout`

- The pipeline freezes T before T meets the holdout: T is the recipe's edit, not a choice made on dev. If step C
  made you upload T1, the confirmation is T's and not the uploaded edit's: say so, and run T1 on the holdout if time
  allows.
- REPORT.md: the traces' table of step A; the fits' data counts and their correlation; for `devw`, `devif`, `mcq`,
  `dev` and the holdout the null edit's line and T's gains with standard errors per source; the thinking length and
  the share cut, before and after; what did not work.
- Upload the final report with the edit. Run `axonry-done` when nothing useful is pending; make sure the last
  complete upload is the edit the readings support.

Time left over is best spent on second draws: `python derive/null_edit.py --case wt --name NULL2 --layer 13` gives a
third draw of the starting model, and `python -m wsp.case run wt <model> <set> --n <more>` adds only the missing
samples; `devw` first.

## 4. What to expect

- Against the two draws: word problems within two points of the starting model, on the upper side if the leg does
  here what it did where it was first fitted; instruction requests, multiple choice and math within their errors.
  On the run's own sets the standard errors are about 1.2 points on `devw` (400 problems x 2), 1 on `devif`, 1.3 on
  `mcq`, 2 to 3 on one source of `dev`: the run will not settle a gain of one point, and its report should say
  plainly what it saw.
- The dose. On word problems dose 2 and dose 4 read the same on `0009547524-0008` (two draws each: +2.1 and +1.3
  against +2.6 and +0.9), and dose 4 cost instruction requests twice as much there. On this entry's start they read alike too (carried over: +0.2
  and +1.1 at dose 2, +0.4 and +0.6 at dose 4). The run uses 2; `DOSE` at the top of `derive/pipeline.py` is the
  place to change it. More dose buys nothing.
- Limits. At dose 2 T moves no tensor by more than 3.1% of its norm; the payload is about 3.4 MB (fp16 direction
  vectors), under the 4 MiB limit. The global relative change is far below its limit.
- Measured times on RTX 4090s, from a full-size run of this pipeline on `0009550031-0005`: the traces 24 min on four
  GPUs; a fit of 8.7 M tokens 40 min; the readings 85 min; the holdout 57 min; 3 h 26 min in all.

## 5. What does not work on this line (practice data, measured on the model of `0009547524-0008`; do not repeat)

All against two draws of that model, on fresh problems.

| edit | effect |
|---|---|
| A teacher that is shown the practice set's worked solution instead of only the answer (the same objective; four fits on the word-problem traces, correlation 0.98; four on math traces, 0.96) | the student is pulled six times as hard (the fit's loss) and learns to think less: at dose 4 the thinking on word problems is 39% shorter and word problems read -2.1 ± 0.9 and -1.4 ± 0.9 (two draws), instruction requests -2.9 ± 0.8 and -1.2 ± 0.8; the math fits read -0.9 ± 1.4 on math. A teacher that knows the path teaches brevity, not the path |
| A whole-trace policy gradient on its word-problem samples (every token credited; four fits, correlation 0.72 to 0.77, orthogonal to the hint leg) | at a sixth of the hint leg's size word problems 0.0 ± 0.8; at three quarters of it word problems -9.0 ± 0.9 and instruction requests -4.7 ± 0.9; a sixth of it next to the hint leg takes the hint leg's gain away (-1.1 ± 0.8). The fits agree with each other and the direction is harmful |
| A third round (the teacher fitted on the traces of the model that already carries T) | its scales point where T's point (correlation 0.63 on the directions the two rounds share, four fits agreeing at 0.97): more of the same, and more dose buys nothing (section 4) |
| A hint leg fitted on the model's multiple-choice traces (dose 4); on its math traces (dose 4) | multiple choice +0.4 ± 1.3; math +1.7 ± 1.4 with 13% cut, by a direction close to T's (correlation 0.73), which reads +0.2 ± 1.5 on math: no gain on math we can count on |
| The answer leg of `0009547524-0008` again, on the model's own instruction samples (896 requests x 4, three fits, correlation 0.63 to 0.69), dose 1.5 | instruction requests +1.5 ± 0.8 alone, multiple choice -0.8 ± 1.1 |
| An answer layer of gains (one gain per MLP unit and attention head, fitted on the answer tokens of instruction requests under the sampler) next to T | instruction requests up, but word problems only +0.5 ± 0.8 with it: the layer takes back T's gain |
| Closing the thinking near the token limit (a replay: the 80 math answers still thinking at 28,000 or 30,000 tokens, given back with the thinking closed there and the remaining budget) | the answer goes on deriving and is cut again in 77% to 81% of the cases and right in 4% to 6%; over all math answers +0.5 and +0.7 points. Closed at 20,000 or 24,000 tokens it loses 1.4 to 3.2 points. This line's cut answers are not right answers waiting to be stated |
| More room for the answer (a replay: the 71 math answers of three models of this line that were cut after their thinking had closed, median thinking 28,900 tokens, given back with that thinking and 8,192 tokens for the answer) | 58% finish and 9% are right: an answer phase that fitted the budget would be worth half a point of math |
| Weighting the teacher's loss towards the positions where teacher and student differ most | looked at before fitting: three quarters of the divergence sits in 5% of the positions, and most of those are places where the trace quotes the prompt or sets out its format, where the teacher differs only because its prompt carries the hint. Not fitted |

- **One reading is one draw, of the edit too.** The first reading of T said +2.6 on word problems; its second draw
  said +0.9. Take a second draw (the edit with the null edit's unit on top) before believing a gain.
- **Shorter thinking is cheap to buy and is not accuracy.** The hint leg and the worked-solution teacher both shorten
  the thinking; only what they do to the answers counts.
