# Residual pairs on the decisions start: instruction pairs only where a failure lies in the thinking, fitted in the run on the start's own answers

Parameters of this method, fixed in `derive/run.py`; nothing below changes them.

```
DOSE_I=0.35       # share of the one instruction fit's change delivered (0009565562-0012's four-fit dose, not raised)
NEURONS=630       # MLP neurons of the layer (the choice of 0009575910-0010)
KEEP=100          # weight of the keep term; 20,000 protected neurons (the keep term of 0009570269-0024)
```

This entry names `0009580212-0015` as its parent in `entry.json` and **starts from it**, inheriting it whole: none of
its chain is derived or delivered again. The run fits one neuron layer on the start's own answers and delivers it:

- **I, an instruction layer of residual pairs.** The start answers 1,320 instruction requests recombined from the
  practice sets' own material, four times each; each failed finished answer is sampled again four times after its own
  thinking. If one passes, the failure lies in the answer step (*answer-locus*), else in the thinking
  (*thinking-locus*). Thinking-locus failures give the pairs: a passing whole attempt at the same request against this
  one and, where a mechanical remedy exists, the mended answer against the answer as sampled; answer-locus failures
  give same-thinking pairs only as a fixed top-up. One fit of 630 neurons with the keep term, off every neuron a link
  of the start's chain writes as a unit, delivered as int8 directions (about 1.9 MB, estimate).
- **No math layer.** This method is the published method of `0009579800-0016` with its math layer left out (section 1).
- **A fixed check** on practice sets delivers the first of F -> Fh whose floors hold (section 3).

**Credits** by publishing entry (every file in `derive/` but `run.py` is a published file, byte for byte, as shipped by
`0009579800-0016`; some docstrings name scripts not shipped here):

- `0009570269-0024`: the keep term `train_keep.py` (on the trainer `train_sparse.py` of `0009550031-0005`),
  `xitems.py`, `keep_rows.py`, `reqgen.py`, `practice_puzzles.py`, `x_rollouts.py`, `encode_dirs.py`, and
  `rollouts.py`, `dose_sparse.py`, `branch.py` (those three of `0009550031-0005`); the check's puzzles (items 96-127 of
  each kind, from its published practice test part);
- `0009565562-0012`: `mkreq.py`, `genreq.py`, `branch_req.py` (those three of `0009559255-0006`, which recombines the
  requests), `repair.py`, `mend_pairs.py` (request kinds, remedies exactly as published, and its point that a mended
  answer gives a pair where no answer sampled after the same thinking passes); the half-dose fallback (`F1`);
- `0009575744-0005`: the whole-attempt pairs `if_data.py`; `mkdata.py` (of `0009554110-0010`, its `ids` command only);
  the guard pattern (its step D);
- `0009575910-0010`: an own-outcome layer composed on a start that carries gains (its section 1.3); `0009576150-0005`:
  the floors rule (its section 3); the toolkit: resampling the answer after a sample's own thinking
  (`wsp.case branch`); `0009579800-0016` (ours): this method with its math layer, and `run.py`, `route_pairs.py`,
  `mkfit.py`, `chain_units.py`, `floors.py`.

**Ours** (published in `0009579800-0016`, used here on a new start): `route_pairs.py` (the routing by failure locus,
and its table); `chain_units.py` (the units of every link of the chain, by the rule of `0009575744-0005`'s
`exclude_neurons.py`, gains left out since they compose with a layer added as a change); `mkfit.py` (pairs, top-up,
keep rows into one fit file); `floors.py` (the check set, its floors); `run.py` (the run, its gates, the order F, Fh).

## 1. Why, from practice data

**Routing.** Pairs of answers after one and the same thinking teach the answer step. The start's chain was already
fitted on that signal (the every-thinking-branched instruction pairs of `0009570269-0024`, the answer-branch gains of
the start's own edit), and each round fitted on it leaves less for the next:

| answers sampled again after the same thinking | model | outcome mixed, pairs |
|---|---|---|
| recombined requests (`0009576222-0011`, its section 1.2) | that entry's start (`0009570269-0024` with twelve ops of `0009569144-0010`), which carries such a layer | 5.3% of 2,555 thinkings, 175 pairs |
| the same requests, as that entry quotes its parent's report | the parent's own model | 9.6%, 302 pairs |
| practice requests `ifsynth` / `tuluif` as they are (the same table) | that entry's start / its parent's model | 2.2% / 4.3% against 4.1% / 4.7% |
| our runs of the round of `0009565562-0012` (1,320 requests x 4) | `0009568831-0002` | 299 pairs from 2,527 thinkings |
| the same | `0009569144-0010`, whose own round was fitted on requests of this kind | 172 pairs from 2,529 thinkings |

Yet the two starts of our runs fail about a third of their answers to these requests (64.4% and 69.7% of 5,280
answers follow every instruction). Two signals still reach a thinking-locus failure: a passing whole attempt (one
practice request in five on a related start has one, lost to the draw; passing attempts are shorter, 3,250 against
4,902 tokens at the median: `0009575744-0005`, section 1.1) and the mended answer (our run on `0009568831-0002`: 891
of 1,458 failed finished answers mended, 736 then kept every instruction). Our practice
ablation on `0009576150-0005` (held-out recombined requests x4, 401 requests): the routed layer +3.9 ± 1.1 over
the start, the same layer fitted on unrouted pairs +2.6 ± 1.0; unrouted against routed -1.4 ± 1.0.

**Keep term.** A preference round on instruction requests moves more than requests: on the practice puzzles
`0009577063-0016` read -1.36 ± 0.47 for the published round as it is, -0.49 ± 0.47 with the start's own finished puzzle
answers held in the fit (its section 1.2) and -0.37 ± 0.47 for its method as shipped (its section 1.3). So the fit
holds the next-token distribution to the start's (KL, weight 100) on its own right answers (450 `reqgen.py` requests,
300 word problems, 300 math problems, 150 multiple-choice questions, up to 900 puzzle answers of 19 kinds)
and never choose the 20,000 neurons that the non-puzzle rows lean on most.

**No math layer.** In our practice runs on `0009576150-0005`, the link this start is built on, the published
math preference with the keep term gave no gain on 300 fresh practice math problems x4 at any dose read (0.1:
-1.0 ± 1.1; 0.25: about -1.0; 0.5: -1.2 ± 1.1), and the instruction layer alone read -0.6 ± 1.1 there. So this
entry delivers the instruction layer alone.

## 2. Setup (first 10 min)

`task.json` -> `starts` lists the published edits the bundle offers. **This entry starts from `0009580212-0015`,
whichever other start the bundle offers.** On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009580212-0015 sandbox:/workspace/work/     # the start (its own parents are inside it)
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/NOTES.md`; never `pkill -f` / `pgrep -f` a
pattern (it matches your own shell); keep the contents of `task.json` (upload credentials) out of logs and reports.

## 3. Running

One resumable script does all GPU work, the check and the upload: start it once with the minutes left on the sandbox
clock (`axonry-time`) and look every 15 minutes (logs `k12/logs/<stage>.log`, decisions `k12/state.json`). If a
stage fails, `run.log` prints its log tail: fix the cause and start the same command again; finished stages
(`k12/done/`) are skipped and the first start's clock is kept (`--left` is ignored; only if that clock is wrong, give
`--reset-clock --left <minutes left now>`). Start no other GPU job while it runs.

```
setsid nohup python derive/run.py --task task.json --left <minutes left> >> run.log 2>&1 < /dev/null &
grep '^\[run\]' run.log | tail                     # its progress lines (stage start / done / decisions)
python derive/run.py --status                      # stages done, gates, time guards, the edit chosen
python derive/run.py --dry-run                     # every command it runs, in order
kill $(cat k12/run.pid)                            # stops the run and its running job
```

| stages | what | clock (estimate, 4 x 24 GB) |
|---|---|---|
| `s0_*`, `s1_keep` | case `kc` on the start; chain units; check set `chk`; 1,320 requests; the start's keep rows | 0:00-0:23 |
| `s4_req`, `s5_*`, gate I | the requests x 4 (12,000 tokens); resampling, routing, pairs | 0:23-0:50 |
| `s6_I0`, `s7_*`, `s8_check_F`, upload | the instruction fit; `F`; the start and `F` on `chk`; floors; upload | 0:50-2:00 |

Each fallback adds about 25 min (estimate). **Gates and guards** (fixed; `run.log` and `--status` show the gates
and time guards):
- gate I: below 200 thinking-locus pairs, same-thinking pairs of answer-locus failures are added in a fixed random
  order up to 300 (`mkfit.py --min 200 --cap 300`); below 100 in all, no instruction layer. Requests that repeat a
  multiple-choice or math body (`repc`, `repm`) give no whole-attempt or mended pair (`--skip-kinds repc repm`; the
  published round leaves their mended answers out); the top-up is not filtered;
- units: over 15,000 neurons counted in the chain, only the start's own link's units are kept off (`[units] FALLBACK`
  in `k12/logs/s0_units.log`);
- time guards (minutes left): the instruction fit 90 (else
  no instruction layer), each fallback 45 (else not tried), the upload 20 (else upload by hand). No layer: no edit.

**Delivery.** Each form is read once against the start on `chk` (800 problems of the case's practice holdout split,
608 practice puzzles: items 96-127 of each kind), one answer per problem. The first form with a gain of at least -2.5
standard errors on every practice source (`floors.py`) and a payload of at most 4,194,304 bytes is delivered, in this
order only: `F` (the layer at its dose), `Fh` (dose 0.175); else nothing. A floor catches a task that breaks, not a point or two: no form is chosen on its gains or re-dosed.

## 4. Uploading and the report

With `--task`, `run.py` uploads the delivered edit (`"delivered"` in `--status`) with `/workspace/work/REPORT.md`
(copied from its record `k12/REPORT_run.md` if you have not written one). It runs:

```
python -m wsp.upload kc <edit> --task task.json --report /workspace/work/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/kc__<edit>/edit --task task.json
python -m wsp.upload kc <edit> --task task.json --report /workspace/work/REPORT.md
```

Run the three yourself after writing REPORT.md, and when `run.py` stopped before its upload while `--status` shows an
edit as `"delivered"`. `axonry-check` must print `"passed": true`; if it refuses that edit, no other form is uploaded
in its place. The last complete upload must be the edit the check chose; run `axonry-done` when nothing useful is
pending. REPORT.md starts from `k12/REPORT_run.md` (the run's lines, gates, **routing table**, floors, the edit
delivered); add three rows each of `k12/br/wpairs.jsonl` and `k12/br/mpairs.jsonl` as read (a mended `chosen` that
reads as another answer altogether is a fault of the remedy), every time guard that fired and everything that failed.

**Rules kept.** Practice sets and the start's own generations only. Training requests are recombined from the shipped
practice material (`mkreq.py`, published kinds only), with no body from the case's dev or holdout split or `chk`; keep
requests are written the way the toolkit writes `practice:ifsynth` (`reqgen.py`); keep puzzles are taken in order
from items 0-95 of each kind (80 a kind); a mended answer is the start's own answer with the request's own material
put in, or its own text rearranged or cut, graded by the toolkit's checker. Every rule is fixed; `chk` is read only
for the floors. No arrays are shipped.
