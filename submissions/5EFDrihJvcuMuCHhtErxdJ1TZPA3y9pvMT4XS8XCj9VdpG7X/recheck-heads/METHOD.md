# recheck-heads: stop a few attention heads from voting for another re-check

## The failure
The parent (the loop-removal champion) still leaves many traces unfinished: the model overthinks. After reaching an
answer it keeps opening new rounds of checking ("Wait, ...", "But let me double-check ...", "Maybe ...") until the 32k
token limit; finished traces are mostly right, so unfinished ones are lost accuracy. Measure on the parent's own
practice traces: share of cut traces, accuracy when finished.

## Suspected cause
The choice between "re-check" and "move on" at a sentence boundary is read from the residual stream along one
direction: the doubt readout u (raises the logits of re-check openers against continue/conclude openers). Attention
is how earlier context reaches that decision: a few heads of the full-attention layers copy a "there is something to
re-check" signal from earlier text into the boundary position, and their output writes directly onto u. These heads
vote for another re-check at every boundary, whatever the state of the solution. The parent's edit removed one loop
direction from the whole attention output of two layers; here the target is the component along the readout
direction u in the output blocks of individual heads, found by measuring each head's write onto u.

The fix: for each chosen head, remove (and mildly reverse) its write along u, on that head's columns of the
attention output projection only: `W_o[:, block_h] -= alpha * u (u^T W_o[:, block_h])` (a `dir_ablate` on the head's
column block). The head keeps everything else it writes.

## Method
Code in `derive/` (uses the wsp toolkit; run from the WeightSpaceProgramming checkout like `python -m wsp.case`).
Keep `PROGRESS.md` current; every step fits in one background job.

0. **Case and parent traces.** `python -m wsp.case init <case> --profile qwen35-0.8b-think --include
   cases/screening/sets/screen64.json --subset dev128=dev:128 --subset holdout256=holdout:256 --parent
   parent/edit/edit.json`; `python -m wsp.case run <case> base dev128 --n 2` (about 35 min on one GPU). In parallel on
   another GPU: `python derive/ifs_set.py <case> --n 128`, then `python -m wsp.case run <case> base ifs128 --n 2`
   (instruction-following guard set from practice:ifsynth).
1. **The readout direction.** `python derive/onsets.py <case> --tags dev128`: doubt openers (Wait, However, But,
   Maybe, Perhaps, Hmm, Alternatively, Actually) against progress openers (Therefore, Thus, So, Now, The, First, Next,
   Let), single-token ones only; `u = normalize(mean E[doubt] * g - mean E[progress] * g)` with E the output
   embedding and g the final norm's effective gain. The lists are defined by role; do not select words by enrichment
   in unfinished traces (topic confound; late halves of cut traces are verbatim repetition).
2. **The heads.** `python derive/heads.py <case> --tags dev128`. At the position before each sentence opener (late
   doubt openers in unfinished traces, early doubt openers and progress openers in finished traces) it reads every
   full-attention head's output and computes its direct write onto u, `c_h = u . (W_o[:, block_h] x_h)`. Take the
   heads with the largest positive mean write at doubt positions that also write at progress positions with a
   similar size (boundary heads: they vote for re-checking everywhere). Skip a head whose late write is several times
   its early write: suppressing doubt only late inside a repetition removes the model's way out of it.
3. **Candidate edits.** `python derive/make_head_edit.py <case> --heads L:h [L:h ...] --alphas ...` for the top one,
   the top two and the top four heads, each at three doses, and `--random-control` for the top-four set at its middle
   dose (random unit direction orthogonal to u, same head blocks). A single head's write is small, so useful alphas
   are well above 1 (start with 4, 8, 12 and let step 4 place them).
4. **Selectivity screen and dose window (minutes).** `python derive/heads.py <case> --model <edit> --logp-only` for
   every candidate and `base`: mean log-prob of the opener at late / early / progress positions (same positions for
   all models). Keep candidates whose EARLY doubt log-prob falls by about 1 to 2 nats, whose late drop is not more
   than about twice the early drop and stays under about 3 nats, and whose progress log-prob does not fall. The
   random control must change almost nothing. If no candidate reaches a 1-nat early drop, add heads from the ranking
   or raise alpha; if the late drop runs ahead of the early drop, drop the most late-heavy head from the set.
5. **Generation tests.** `python -m wsp.case run <case> <edit> dev128 --n 2` and `run <case> <edit> ifs128 --n 2` for
   two or three candidates inside the window (one per GPU). The report pairs them with the parent; read each practice
   source separately (`by_source`) and count verbatim repetitions among cut traces (`kind` == `cut_loop`). Pick the
   candidate with the largest dev128 gain whose accuracy on each dev128 source and on ifs128 is not lower than the
   parent's by more than about one standard error, whose dev128 `cut_loop` count is not higher, and whose ifs128 cut
   rate stays under about 12%. When two candidates are close, take the weaker dose.
6. **Deliver.** Upload the first candidate that passes steps 4-5 at once (`python -m wsp.upload <case> <edit> --task
   task.json`), then freeze the best, run it and the parent on holdout256 if time allows, and re-upload. Report the
   head ranking with the write profiles, the selectivity numbers of every candidate and the control, and the
   dev/holdout/ifs results.

## Time plan (4 GPUs, 6 h)
0:00-0:35 step 0; 0:35-0:55 steps 1-3; 0:55-1:10 step 4; 1:10-1:50 step 5 for three candidates in parallel and the
first upload; then a second round around the best candidate, holdout256, the final upload (keep the last 30 min).

## What to expect
Fewer unfinished traces on dev128 (competition-style math first), accuracy up by most of the rescued share, accuracy
of finished traces about the same, no more verbatim repetitions. The edit touches the output projection of the
attention in a few full-attention layers, a quarter-thousand columns per head, with a relative change of a few
percent per matrix.

## If it does not work
- No boundary head stands out: use the top four by mean write and a larger alpha; if the log-prob still does not
  move, say so in the report and deliver the best candidate that passes the guards.
- The random control moves the log-probs as much as the edit: the direction is wrong (step 1).
- Gains on dev128 come with losses on ifs128 or more verbatim repetitions: step down the dose or drop the head with
  the largest late write.
