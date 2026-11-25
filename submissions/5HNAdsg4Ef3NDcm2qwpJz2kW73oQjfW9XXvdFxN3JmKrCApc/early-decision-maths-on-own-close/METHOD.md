# Early-decision maths credit, fitted in the run on the declared parent's model, at full strength

This entry names the published entry `0009576150-0005` as its parent in `entry.json` and starts from the start the task
offers (task.json `starts`; written while `0009570269-0024` was the champion). Its upload is the declared parent's model
with one more layer on it:

| part | what | source |
|---|---|---|
| 1 | the declared parent's model: its published ops, verbatim, read openly from the bundle (`<root>/0009576150-0005/edit/edit.json` + `edit.npz`) | the declared parent `0009576150-0005` |
| 2 | **EC**, an early-decision credit layer for maths: one gain per MLP unit of layers 4-21 and one per attention head, fitted in the run from the model's own maths answers, exported at **strength 1.0** | the published method of `0009568722-0012`, `0009568726-0005` and `0009568831-0002`, run with the scripts of `0009568831-0002` unchanged (`derive/branch2.py`, `pref_gains2.py`, `fit_gains.py`) and its steps `ec_bank`, `ec_rows`, `ec_branch`, `ec_fits` with the same flags; the strength is this entry's choice (section 4) |

The run delivers part 1 first, early and in the background, as a complete upload (upload 1). It then fits part 2 on
the GPUs and, if a fixed practice check passes and the time rule allows, delivers parts 1 + 2 as upload 2, which
replaces upload 1. Every path ends with one complete upload. No arrays, id lists or data files are shipped: everything
is derived in the run from the bundle's published edit of the declared parent, the toolkit's practice sets and the
model's own answers.

What this entry adds, plainly: the published early-decision credit, fitted in the run on the declared parent's model
from that model's own answers; its strength fixed at 1.0 by the author's practice dose study (section 4) in place of
the published in-run screen, with that screen's 2-point cut-off bound kept as one clause of a fixed in-run practice
check (section 5) that keeps the layer only if it holds up on the run's own answers; and a time rule (section 6) that,
when the run is short, uses fewer mixed problems and never changes a published step or any other constant.

## 1. The behaviour this layer changes

On practice maths problems the model solves only some of the time, the published method takes one finished wrong and
one finished right answer of the same problem and lets the model continue each from 10 %, 25 % and 50 % of its
thinking. Continuations from the right answer's prefix end right more often than those from the wrong one's, and the
gap grows through the first half of the thinking: the answer is largely decided early, well before the end where
closing or answer-stage credit acts. The layer turns these same-prefix continuations into preference pairs (a
continuation that ends right against one that ends wrong, from the same prefix) and fits a gain per MLP unit and per
attention head with the loss on the first 2,048 tokens after the branch point, the stretch where the decision is made.
Unfinished continuations count as failures. Credit for this measurement, the pairs and the windowed loss belongs to
the published entries named in the table.

## 2. Start and uploads

`derive/resolve.py plan` reads task.json and the bundle and fixes the mode; the declared parent's ops are only ever
read from its own published files.

| mode | when | start | B (the model the layer is fitted on) | upload 1 | upload 2 |
|---|---|---|---|---|---|
| ON_116 | the champion offered is the start the declared parent's published edit was made on (its edit.json `parent`, `0009570269-0024`) | the champion | the champion + P0 = the declared parent's model | P0 on the champion: ALL 4,992 of the declared parent's published ops, verbatim (`derive/parent_ops.py`), i.e. the declared parent's model re-delivered | one edit on the champion: P0 + EC (the local chain start -> P0 -> EC, flattened by `wsp.upload`) |
| ON_CHAMPION | the champion's lineage holds `0009576150-0005` | the champion | the champion's model | the null edit on the champion | EC on the champion |
| ON_PARENT | no other champion is offered, or one that is neither | the declared parent (a start the task offers) | the declared parent's model | the null edit on it | EC on it |
| ON_START | the declared parent is not offered or its edit.json is missing | the champion offered | its model | the null edit on it | EC on it |

P0 is a verbatim copy: every op of the declared parent in its order, every array under the same key, the references
rewritten from `edit.npz#k` to `P0.npz#k`; `parent_ops.py` refuses (and the mode becomes ON_START) unless the declared
parent's edit.json names the start as its parent. Its payload equals the published edit's. The null edit is one
`scale_rows` of factor 1.0 on the start (the start unchanged). No op or value of any entry other than the declared
parent enters an upload; in ON_CHAMPION the champion's chain is read only for its lineage.

## 3. The layer, step by step (published constants; this run's own draw)

The run's draw index is DX = 85 (the published runs used 0-1): bank order `random.Random(4096 + 85)`, rows order
`random.Random(11 + 85)`, fit seeds 8501-8504, salts `ecbank_u85` (bank) and `ec_u85` (branches).

1. **Bank** (`derive/ec.py bank-ids`, then `derive/gen1.py` on each GPU): the `math_train` practice problems of levels 3-5
   (pinned rows, section 7), ids sorted, minus every id of this run's case splits and practice check problems, shuffled
   with `random.Random(4181)`, the first 600; each answered 4 times by B (salt `ecbank_u85`, at most 32,768 tokens,
   the profile's decoding), one call per GPU on interleaved shards. `gen1.py` is the toolkit's `wsp.gen._generate`
   (same prompts, sampling, per-request seeds and grading) on one GPU.
2. **Mixed problems** (`derive/ec.py rows`, the published step `ec_rows` line for line): answers grouped by problem, ids
   sorted and shuffled with `random.Random(96)`; in that order, each problem with at least one finished wrong answer
   (kind `finished_wrong`) and one finished right answer, both of at least 3,000 tokens, gives one of each (drawn with
   the same generator), up to C problems. C = 200 (the published cap) unless the time rule of section 6 lowers it.
3. **Branches** (`branch2.py --kinds wrong|right --points 0.1 0.25 0.5 --k 4 --max-new 24000 --salt ec_u85 --penalty
   prefix`): four continuations of each kept answer from 10 %, 25 % and 50 % of its thinking, the rows of each kind in
   interleaved shards queued on the GPUs.
4. **Fits** (`pref_gains2.py --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --max-len
   24576 --chunk 1024 --allow-think --bad-cut --window 2048 --points 0.1 0.25 0.5`, both branch files): four fits,
   seeds 8501-8504, one per GPU in one batch. At least two finished fits are needed.
5. **Export** (`fit_gains.py export --k 1.0 --levels 256`, then `compact.py`): the fits' log gains averaged, scaled by
   the strength 1.0 and exported as `scale_rows` (MLP units) and `scale_cols` (heads) ops with 256 levels per layer;
   `compact.py` stores the index arrays as int16 (the same model bit for bit, half the index bytes).

## 4. Why strength 1.0: the author's practice dose study

The published method chooses the strength in its run: the averaged gains at 0.5, 0.75 and 1.0 answer 256 fresh
practice maths problems once each, the largest strength whose thinking length and cut-off share stay within its bounds
is used, and 0.5 is the fallback (its notes report the cut-off share rising from 2.2 % to 5.2 % at full strength on
its own model). This entry fixes the strength at 1.0 instead, from the author's dose study below, and bounds the
cut-off share in its own in-run check (section 5). The author tested strength 1.0 under a rule written
before any of its reads, on practice data only:
- Model pair: an earlier lab model of the author (B) against the same model + an EC layer exported at strength 1.0. That
  layer was fitted from 174 mixed problems (about 1,100 preference pairs) of a bank of 512 problems x 6 answers (32,768
  tokens); this entry's run follows the published form (600 problems x 4 answers, up to 200 mixed problems).
- Primary set: 1,200 `math_train` problems (levels 1-5) used in no fit, list or choice of the layer, 4 answers each,
  paired per problem and answer; the guard sets with fresh answers of both models.
- Rule: maths d >= +2 se; no guard set below -2 se; the maths share of answers cut at the length limit at most 2 points
  above B's.
- Result (all sets complete, the rule passed):

| set | problems x answers | B % | B + EC at 1.0 % | d +- se |
|---|---|---|---|---|
| maths (`math_train`) | 1,200 x 4 | 67.81 | 70.62 | **+2.81 +- 0.57** (z +4.97) |
| puzzles (`bbhsynth`) | 608 x 4 | 87.79 | 87.83 | +0.04 +- 0.63 |
| requests (`ifsynth`) | 896 x 4 | 92.97 | 93.30 | +0.33 +- 0.47 |
| multiple choice (`mmlupro`) | 821 x 8 | 46.76 | 47.76 | +1.00 +- 0.58 |
| word problems (`gsm8k_train`) | 500 x 2 | 78.60 | 81.40 | +2.80 +- 1.19 |

  The maths cut-off share went from 0.73 % to 1.15 % (+0.42 points); maths answers grew about 4 % longer (12,719 ->
  13,261 tokens on average).
- At strength 0.5, EC layers of this kind read +0.75 to +0.96 +- 0.6 points on another set of 1,200 x 4 practice maths
  draws in three of the author's builds (the dose study's own fits at 0.5 on its model B, the same fits on a second
  lab model, and the published entry's own merged gains on B): under 2 se each time. At 1.0 the gain is about three
  times as large and clear of noise, with no guard set below its floor.
- The same design on the declared parent's model (a bank of 600 problems x 4 answers, 200 mixed problems, four fits,
  strength 1.0, with another draw index than this run's), read on the 1,200 x 4 problems of the primary set against
  that model, with a bar of +1 se fixed before the read: 67.44 % -> 69.25 %, d +1.81 +- 0.57 (z +3.19); maths
  cut-off share 0.71 % -> 0.88 %.
- So this entry exports at 1.0, and the in-run check (section 5) still holds the layer back if it does not reproduce.

## 5. The practice check (fixed) and what it decides

- Problems (`derive/chk.py ids --case st`, practice data only, none in the bank): pz 608 test puzzles
  (`bbhsynth` items 96-127 of each kind), math 256 (`holdout_math` of `derive/sets.py`), req 256 (128 `holdout_ifsynth`
  + 128 `extra_tuluif_b`), mc 256 (`holdout_mmlupro`); 2 answers each, salt `ecchk` for every model (paired draws).
- Reads: REF = the start on every set; U1 = upload 1's model on the 256 maths problems (ON_116 only; in the other
  modes, and when upload 1 is the null edit, upload 1's model is the start, so REF's rows are U1's); G = upload 2's
  model on every set.
- Rule (`chk.py rule final --set math --cut 2.0`): upload 2 goes out only if all three hold, else upload 1 stands:
  (a) floors against the start: on pz, math, req and mc, paired d(G - REF) >= -2.5 se (se = 0 needs d >= 0; a missing
  set fails); (b) maths not below upload 1's: d(G - U1) >= 0 on the 256 x 2 maths draws; (c) G's maths cut-off share at
  most 2.0 points above U1's.
- Why these clauses: (a) keeps every practice set within noise of the start (a layer that changes nothing falls below
  -2.5 se on one of the four sets in about 2.5 % of runs); (b) is the layer's purpose, measured against what stands
  otherwise; (c) is clause 3 of the dose study's rule and the published method's own 2-point bound. On 4,000 random
  256-problem x 2-answer subsamples of the dose study's maths rows, (b) failed in 1.7 % and (c) in 0.3 %; with the
  run's own fresh answers an estimated 3-5 % of runs would hold back a layer as good as the study's; on the same
  subsamples of the rows of the read on the declared parent's model (section 4, +1.81 points), (b) failed in about 10 %.

## 6. Steps, gates and time (minutes on 4 x RTX 4090)

`python submission/derive/run_e1.py all` runs the steps below, each marked done in `run_e1/` (a restarted driver skips
what is done). `left()` is the run's own clock: 360 minutes from the driver's first start minus `T_OFFSET_MIN`, or the
task's deadline when it is known, whichever is earlier.

| # | step | where | what | gate (minutes left) | time |
|---|---|---|---|---|---|
| 1 | start | CPU | pin check (section 7); resolve the mode; case `st` on the start; ON_116: P0 and case `ec` on P0; case splits, check problems, bank ids | - | 2.7 measured |
| 2 | base | CPU | build B; start the background job | rule D1 checked first | 8.6-12.4 measured |
| bg | background job | CPU, beside steps 3-6 | upload 1 (package, payload, `axonry-check`, send); if it fails before sending, the null edit; ON_116: build the start's model for the check | upload 1: 20 + 20 | 14.75 measured + send |
| 3 | bank | GPU | section 3 step 1 | **D1** | 48.3 measured |
| 4 | rows | CPU | section 3 step 2 with C from **D2** | **D2** | 0.1 measured |
| 5 | branch | GPU | section 3 step 3 | C rows still fit | 7 + 0.39 C (80.8 measured at C = 192) |
| 6 | fits | GPU | section 3 step 4 | **D3** | 2 + 0.20 C (40.4 measured at C = 192) |
| 7 | export | CPU | section 3 step 5; then build upload 2's model in the background | - | 0.2 measured |
| 8 | refs | GPU | REF (and U1 in ON_116), beside that build | **D4** | 18.5 measured (+ 10) |
| 9 | check | GPU | G, then the rule of section 5 | **D5** | 20 |
| 10 | up2 | CPU | upload 2, if the rule said FINAL | 15 + 30 | 29.05 measured (ON_116) + send |

Times: "measured" on 4 x RTX 4090 in the author's rehearsals with the same scripts and calls (start, model builds,
practice read, upload checks; the bank of 600 problems x 4 answers: 48.3 minutes, the four calls 39.5-47.4;
the branches of 192 mixed problems: 80.8 minutes, wrong-trace calls 40.0-41.8 and right-trace calls 35.8-38.1, so
0.39 minutes per mixed problem after 7 minutes of engine starts; the four fits on their branch rows: 40.4 minutes,
39.6-40.0 each, so 0.20 minutes per mixed problem after 2 minutes of start). The practice check takes T_CHK = 49
minutes in ON_116 (REF 19 and U1 10 while upload 2's model builds, then G 20) and 39 in the other modes. Upload 2
keeps R_UP2 = 30 + 15 = 45 minutes for its package, check and send.

**Rules** (fixed; FIX = 7 + 2 + 1 + T_CHK + R_UP2 = 104 in ON_116, 94 otherwise; 0.59 minutes per mixed problem for its
branches and fits; a planning margin of 10 minutes in D1 and D2 only):
- **D1** (before B is built and again at the bank start): EC starts only if left >= 49 + 0.1 + FIX + 0.59 x 174 + 10
  (265.76 in ON_116, 255.76 otherwise).
- **D2** (rows): C_time = floor((left - FIX - 10) / 0.59). If C_time >= min(200, A) (A = mixed problems the published
  rule finds in the bank), C = min(200, A): the published rule unchanged. Else if C_time >= 174, C = C_time: the first
  C_time mixed problems in the published order. Else EC stops.
- **D3** (fits): left >= 2 + 0.20 C + 1 + T_CHK + 45 (137 in ON_116 at C = 200). **D4** (reads): left >= T_CHK + 45
  (94 / 84). **D5** (G read): left >= 20 + 45 = 65. Upload 2: 15 + 30 before packaging, 15 before sending (20 + 30
  and 20 while upload 1 has not landed).
- A stop at any gate, a failed EC step, fewer than 2 fits or an UP1 verdict all leave upload 1 standing.

Why the reduction cuts mixed problems, not the bank: the bank costs about 0.08 minutes per problem and yields one mixed
problem per about three (192 and 215 of 600 in the author's two banks), about 0.25 minutes per usable problem, against
0.59 for that problem's branches and fits; a smaller bank saves little and pushes the mixed count under the floor.
174 is the smallest EC data set read on practice at strength 1.0 (the dose study's layer, section 4), so the time rule
never cuts below it (a bank that by itself yields fewer mixed problems is used as the published rule takes it, C = A,
and the check of section 5 decides); the published method's notes also report a much smaller effect from a much
smaller version (130 problems, branch points 10 % and 25 % only, about 550 pairs). With C between 174 and 199 the
layer is fitted on the first C mixed problems of the same draw, 87-100 % of the full data, every other constant
unchanged.

**Plan from the driver's start** (start 3 and B 12.4, then the step budgets above, C = 200): ON_116: B built at 15.4;
upload 1 lands at about 32; bank ends 64.4; branches end 149.4; fits end 191.4; export 192.4; REF + U1 reads end
221.4; G read and rule 241.4; upload 2 lands at about 272 (about 88 minutes left). The other modes run about 10
minutes shorter. With L minutes left when the driver starts: full C = 200 when L >= 296.4 (ON_116; 286.4 otherwise),
C between 174 and 199 when L >= 281.4 (271.5 otherwise), and below that EC does not start and upload 1 is the run's
upload. For example, with 285 minutes left at the start (75 used before it), ON_116 uses C = 179 and upload 2 lands
with about 25 minutes left.

## 6b. Running it (4 x RTX 4090)

From the unpacked bundle, put `submission/`, `task.json` and every start directory task.json `starts` lists in
`/workspace/work`. **task.json holds upload credentials: keep it out of logs and REPORT files.** Then, in a tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/work && python submission/derive/resolve.py plan --task task.json
T_OFFSET_MIN=20 setsid nohup python submission/derive/run_e1.py all > /workspace/work/run_e1.out 2>&1 &
tail -n 5 /workspace/work/run_e1/logs/run.log
```

- `resolve.py plan` prints the mode and the start; if it names a missing directory, copy it from the bundle first.
- Set `T_OFFSET_MIN` to the minutes of the run used so far (20 above); it is saved on the first start, and the driver
  uses the earlier of its own count and the run's deadline when it can read it (task.json, or `/etc/axonry-deadline`).
  Every gate reads this clock, so the minutes used before the start decide how much of section 6 runs.
- Start the driver once (it refuses a second start while the first one's pid is alive). It is resumable: a step with
  a `.done` marker in `run_e1/` is skipped; a restarted driver first stops the GPU calls, model builds and sends a
  killed one left, keeps a running background job and starts it again if it was lost before finishing.
- `python submission/derive/run_e1.py status` shows the steps done, the mode, C, the rule's verdict and what landed.
  Check it every 10-15 min with short commands. The driver makes every upload itself; never run `wsp.upload` by hand.
  Run `axonry-done` only after `RUN_E1_END`; on `RUN_E1_INCOMPLETE`, once `status` says the driver is not running,
  run `python submission/derive/run_e1.py up` again at once.
- While GPU calls run, `run.log` gets a progress line at least every 10 minutes. If it shows no new line for 60
  minutes while the driver runs (an upload in progress writes to `logs/<tag>*.log`: that counts as progress), kill
  the driver's process group and start it again once.
- Use only `practice:` data and the models' own generations. Never `pkill -f` / `pgrep -f` a pattern.

## 7. Uploads, budgets, and what stands on each failure

| upload | content | payload | check budget | gates |
|---|---|---|---|---|
| upload 1, ON_116 | P0 on the start (4,992 ops) | about 3.52 MB (= the published edit's); edit.npz 4.72 MB | 20 (14.75 measured) | 20 + 20 |
| upload 1, other modes | the null edit | tiny | 22 | 12 + 22; without `axonry-check` at >= 15 (one `scale_rows` of factor 1.0 on the start) |
| upload 2, ON_116 | P0 + EC, one edit on the start | about 3.72 MB (P0's 3.52 MB + EC's about 0.2 MB; limit 4,194,304 B); edit.npz about 6.1 MB (limit 8 MiB) | 30 (29.05 measured) | 15 + 30 (20 + 30 while upload 1 has not landed) |
| upload 2, other modes | EC on the start | about 0.2 MB | 30 (16.3 measured) | 15 + 30 (20 + 30 while upload 1 has not landed) |

Check budgets are the measured package + payload + `axonry-check` minutes x 1.35, except upload 2's: its ON_116 form
(one edit of about 9,840 ops) measured 29.05 minutes (with a GPU read beside it for the first ten), rounded up to 30
and kept for the smaller form of the other modes. Every send attempt of `wsp.upload` is stopped after 10 minutes (its
process group ended) and counts as failed; an `axonry-check` running past twice its budget, or packaging past 30
minutes, counts as failed. Success is `wsp.upload`'s own exit status.
- Upload 1 fails before sending (ON_116): the null edit is sent instead.
- Upload 1 or the null edit fails while sending: the same files are sent again (8 minutes needed).
- Upload 2 fails before sending (package, payload, file size, `axonry-check`): upload 1 stands.
- Upload 2 fails while sending: every send attempt of it is ended early enough to leave 7 minutes; it is sent again
  only with at least 8 + 6 minutes left; else, or if that fails, upload 1, already checked, is sent again (6 minutes
  needed: its packaging and one send) and upload 2 is not tried again. Once upload 2's send has started, nothing else
  sends upload 1 or the null edit.
- Nothing landed and nothing was sent (for example a stop in step start): upload 1 (or the null edit) now.
- A restarted driver ends the GPU calls, builds and sends an earlier driver left, keeps a running background job, and
  starts the background job again if it was lost before finishing (a landed upload 1 is kept, a started one is sent
  again). `run_e1.py up` runs the upload step and these end rules again.
- The pin check (section 7) failing, an EC step failing, a gate refusing or the rule saying UP1 never touch upload 1.

Each upload carries `REPORT_<tag>.md`, written from the run's own log: mode and start, what the upload holds, the draw
(DX, salts, seeds), bank answers, A, C and C_time, branch rows, the fits' pair counts, the export line, the practice
table and the rule's verdict, and the reason EC stopped if it did.

**Pins.** `derive/pools.py` pins the practice rows every list of the run is built from: the first 1,024 rows of
`ifsynth` and `tuluif`, the 7,274 rows of `math_train` and the 2,432 rows of `bbhsynth`, as the toolkit held them when
this entry was written; `pools.check()` in step start stops the maths layer (not upload 1) if a pinned set has fewer
rows; the id digests are reported as information.

**Threads.** NCPU = the CPUs this process may use. Model builds, export and upload 1's package and check (beside the
GPUs): max(4, min(32, (NCPU - 8) // 2)); upload 2 and late uploads: max(4, min(32, NCPU - 4)); each fit: max(2, min(16,
(NCPU - 8) // 4)). GPU calls run one per GPU, each in its own process group and kernel cache; a bank call is stopped
after 75 minutes, a branch call after 80, a fit after 100 (none retried); a practice read after 45 (once retried if it
still fits).

## 8. Constants and their reasons

| constant | value | reason |
|---|---|---|
| strength | 1.0 | the author's practice dose study (section 4) |
| bank | 600 problems, levels 3-5, 4 answers, 32,768 tokens | the published step `ec_bank` |
| mixed problems | at most 200 | the published step `ec_rows` |
| time-cut floor | 174 | the smallest layer read on practice at strength 1.0 (section 4) |
| branch points, k, new tokens, penalty | 0.1 0.25 0.5; 4; 24,000; prefix | the published step `ec_branch` |
| fit flags | section 3 step 4 | the published step `ec_fits` |
| fits, minimum | 4, 2 | 4 fill the 4 GPUs in one batch; the author's EC fits of one bank correlate 0.83-0.84 seed to seed, so the mean of 2 keeps most of what 4 give |
| levels | 256 | the export default, as published |
| DX | 85 | a draw index not used by the published runs or the author's lab builds |
| check salt | `ecchk` | not used by anything else |
| check rule | section 5 | section 5 |
| time constants | section 6 | 4 x RTX 4090 times measured with the same calls (bank, branches, fits, builds, reads, checks), rounded up; the maths-only read of upload 1's model is 10 (10.0 measured) and the read of upload 2's model (20) is set from the measured read of the start (18.5) |
| check budgets 20 / 30, UP minimum 20, upload 2 minimum 15, null edit 12 + 3 | section 7 | measured minutes x 1.35 (upload 2: its measured 29.05, rounded up); a send and a re-send |
| upload 1 re-send 6, upload 2 attempt reserve 7 | section 7 | upload 1's re-send is its packaging and payload count (2.65 measured) and one send (about 1); a hanging attempt of upload 2 must leave that much |
| call caps 75 / 80 / 100 / 45 | section 7 | about 1.6, 1.9, 2.5 and 3.3 x the slowest measured call of each kind (bank 47.4, branch 41.8, fit 40.0, practice read 13.6), so that a slower host still finishes them; the gates of section 6 decide what follows a slow call |

## 9. The author's earlier entry

The declared parent `0009576150-0005` is the author's: it carried a published answer-closing layer onto the champion as
a change and fitted request gains on the result; it fitted no maths layer. This entry keeps that model unchanged as
upload 1 and fits the maths layer on it.

## 10. Credits, literature and files

- Early-decision credit, its measurement, the same-prefix pairs, the windowed loss, the published constants and the
  scripts: `0009568722-0012`, `0009568726-0005`, `0009568831-0002`. `derive/branch2.py`, `pref_gains2.py`,
  `fit_gains.py` and `sets.py` are byte-identical to `0009568831-0002`'s files; its steps `ec_bank`, `ec_rows`,
  `ec_branch`, `ec_fits` are followed command for command, with this run's draw index (`ec_rows`' code is in
  `derive/ec.py rows`; the bank's answers come from `derive/gen1.py`, the toolkit's own generation function on one
  GPU).
- The answer-branch preference parameterisation (one gain per MLP unit and per head; `derive/sets.py`) that the method
  extends: `0009552831-0015`.
- The declared parent `0009576150-0005`: upload 1 and the start model.
- The champion `0009570269-0024`: named only as the start at the time of writing; its published puzzle split defines
  the check puzzles (items 96-127).
- Literature: Rafailov et al. 2023 (direct preference optimisation); Lightman et al. 2023 (process supervision);
  Wang et al. 2024, Math-Shepherd (step credit from continuations of a shared prefix); Setlur et al. 2024 (rewarding
  progress).
- Files: `derive/run_e1.py` (the driver), `resolve.py` (modes), `parent_ops.py` (P0), `ec.py` (bank ids, mixed
  problems), `chk.py` (practice check), `pools.py` (pins), `gen1.py` (`wsp.gen._generate` on one GPU), `compact.py`
  (int16 indices), and the published `branch2.py`, `pref_gains2.py`, `fit_gains.py`, `sets.py`.
