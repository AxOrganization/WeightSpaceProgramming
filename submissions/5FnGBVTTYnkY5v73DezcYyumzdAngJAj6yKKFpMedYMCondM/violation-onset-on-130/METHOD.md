# Violation-onset credit for instructions, on the published model of 0009576150-0005 (#130)

This entry adds one layer, **VO**, to the published model of its declared parent `0009576150-0005` (#130,
"own-close-carried-as-change"; `entry.json`), and uses that model as it is. VO is derived in the run from the
toolkit's practice instruction requests and the model's own answers: where an answer first breaks its request (found
by the request's own checker on growing prefixes of the visible answer), the model continues from that point; the
continuations that keep the request are preferred to those that break it, with the loss on the continuation only,
fitted into per-unit and per-head gains. The upload is VO alone, stacked on #130's published edit. One driver,
`derive/run_a.py`, runs every step and the upload (section 3).

```
Fixed settings (derive/run_a.py and the scripts it calls; the run sets only --gpus)
start: the declared parent's published edit, 0009576150-0005/edit/edit.json, as it is (START_FROM_PARENT = True)
VO   : 1,920 requests (960 ifsynth + 960 tuluif) x4 (12,000 tokens) + recombined requests (seeds 0-4) x2; word and
       sentence onsets; 12 continuations per point; at most 2 pairs per point; beta 0.1, NLL 0.2; layers 4-21 with
       heads; 2 epochs; seeds 201-208, log-mean; strength 1.0; KEEP_W 0
other: no other layer (NO_R = True); if VO fails, nothing is uploaded
```

## 1. The start

The start is the edit that `0009576150-0005` (#130) published, read from the bundle's `0009576150-0005/edit/` together
with the chain it stands on. It is used as it is: none of its ops is changed, left out or applied a second time. The
case of the run starts from it, and the upload names it as its parent, so the delivered edit is VO alone on top of it.

What that model is, in our words (#130's METHOD describes it in full; its REPORT.md is in the bundle). #130 stands on
the champion of its time, `0009570269-0024` (#116, "procedure-workings"). It takes the published layer of
`0009569144-0010` (#111, "keep-the-answer-own-close": fixed rules - wrap-up units silenced, exit, break and commit
units, four direction removals on attention outputs - and a fitted round that favours answers which keep the result
they found and close their own thinking) and adds it to the champion as a change to the weights rather than as their
values, so that neurons touched by both layers keep both changes; the round goes in at half size and stays off the rows
the champion's own set ops write. On that merged model it fits one gain per MLP unit and per attention head that
prefers instruction answers keeping their request: the request-gain method of `0009552831-0015` as published by
`0009568726-0005` (#106). Its own practice checks decide what its final upload holds; the bundle holds that edit. The
champion of the moment is not used: the driver only checks that its directory is present and names it in the log.

## 2. VO: violation-onset credit for instructions

**The behaviour.** The start follows most practice instruction requests. When it fails one, the request's own
checker (the toolkit's instruction checkers, as `wsp.grade` uses them) tells three shapes apart on the visible answer:
- *growing*: once broken, more text cannot repair it (a comma under "no commas", a capital under "all lowercase", a
  forbidden word, the word that crosses a "less than" limit, a keyword once too often): the answer was right up to
  one word;
- *missing*: more text could repair it (postscript, end phrase, sections, placeholders, "at least" counts);
- *start*: decided by how the answer begins (JSON, repeat the request, quotation, language, two responses).

**Why a local fix.** For a growing failure everything before the onset was fine; the loss is one decision. Credit
spread over the whole answer is spent on hundreds of tokens that were right, and resampling the whole answer mostly
changes how it is written, not that decision.

**How the edit is found.**
1. *The start's answers.* 4 answers to 1,920 practice instruction requests outside the case's dev and holdout splits (960 from
   `ifsynth` and 960 from `tuluif`, each after a fixed shuffle, so that the mix does not follow the sets' sizes; 12,000 tokens), and 2 answers to each of the requests
   recombined from the practice material by #94's `mkreq.py` (seeds 0-4; about 6,600 requests), which fail more
   often and more variously.
2. *Onsets* (`derive/if_onset.py`). For a failed, finished answer each failing instruction is classed. For a growing
   one the checker runs on growing prefixes of the visible answer (word boundaries, binary search, valid because a
   growing failure stays broken; the two case instructions use the first letter of the wrong case): the onset is the
   start of the word whose inclusion first breaks it. For a missing one it is the start of the last paragraph. Start
   failures are left out, and so is any branch point at the very start of the visible answer. The visible answer is
   the text after the last `</think>`, as the grader reads it. Each onset gives two branch points: just before the
   breaking word, and the start of its sentence or line, one decision earlier.
3. *Continuations* (`derive/onset_branch.py`, the branch kind `onset`). Prompt = the chat prompt plus the answer's own
   text up to the branch point (thinking, `</think>`, the visible answer so far); 12 continuations with the profile's
   sampler (the prefix's presence penalty as a fixed logit bias); each graded by the request's checker on the full
   answer. The original answer counts as one more failure of its point.
4. *Pairs and fit.* At each point a continuation that keeps the request against one that breaks it (or the
   original), at most 2 pairs per point, cut continuations counted as failures. The fitter is #107's published
   `pref_gains2.py`, unchanged: DPO-style preference (beta 0.1) plus 0.2 x NLL of the kept continuation, one gain per
   MLP unit and per attention head of layers 4-21, 2 epochs; the loss covers the continuation only (the prefix is
   context). Optionally (`KEEP_W` > 0) each step also holds one of the start's own answers to 480 practice reasoning
   puzzles (`practice:bbhsynth`, x2) at the start: `KEEP_W` x KL(start || edit) over the vocabulary on its generated
   positions. This entry runs `KEEP_W` = 0, `pref_gains2.py` without its keep term. Eight fits (seeds 201-208, rounds
   of one per GPU), averaged in log space: single fits move different units, and their average is what keeps the other
   tasks in place.
5. *Strength* fixed at 1.0 (the merged gains as fitted), exported with #107's `fit_gains.py export`. It is not chosen
   in the run: a screen of a few hundred held-out requests cannot separate strengths a point apart.
6. *Check.* VO has no paired screen of its own in this run. What it measures on practice data is printed on the
   way: the start's answers and how many fail, the onsets by class, per branch point how many of the 12
   continuations keep the request, and the pairs the fits use (status log and REPORT.md); each fit's loss and
   preference accuracy on its pairs (fit logs). If any VO step fails, nothing is uploaded.

**What is changed.** Per-unit gains (`scale_rows` on `up_proj`) and per-head gains (`scale_cols` on the attention
output projections) of layers 4-21, about 0.4 MB.

**What sets it apart.** The new branch kind `onset` branches where the request's own checker, run on growing
prefixes of the visible answer, first finds it broken, and one sentence earlier; the loss covers only the
continuation; the fitter is #107's published code, unchanged. The material is local (a kept prefix and the model's
own continuations from it), not whole answers and not answers regenerated after the thinking.

**Order.** VO is fitted on the start itself (the parent's published model), so its pairs are the answers of the model
that will be delivered. The parent's own instruction layer (its request-gain round, which resamples whole answers
after the thinking) stays as it is; VO adds branch points inside the visible answer.

## 3. Running it (4 x RTX 4090, 6 hours)

**Copy** (agent machine, from the unpacked bundle; each start directory under its id, next to task.json):
```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/      # the declared parent: the start
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/        # the champion, when task.json lists it
```
task.json holds the upload credentials: it stays out of PROGRESS.md, notes and logs.

**Launch** (sandbox; `export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp` if not set):
```
cd /workspace/work && mkdir -p "$WSP_WORK" "$WSP_CASES"
setsid nohup python derive/run_a.py --gpus 0,1,2,3 > "$WSP_WORK/run_a.out" 2>&1 < /dev/null &
tail -n 5 "$WSP_WORK/runa/status.log"
```
One line per step goes to `$WSP_WORK/runa/status.log` (VO's own steps also to `$WSP_WORK/onset/status.log`);
decisions to `$WSP_WORK/runa/state.json`. Started again (same `--gpus`), the driver skips what is done. Every ~10
minutes make the first line of `/workspace/work/PROGRESS.md` one plain sentence of at most 20 words on where the run is.
A build prints nothing for up to ~20 minutes (python at ~100% CPU): not a hang, never kill it. Jobs that print their
last line and then hang at vLLM shutdown are killed after 300 s and count as finished (the drivers' guards).

**Time plan** (from runs of the VO steps on 4 x RTX 4090): VO case on the start and its build 0:10-0:30; answers
~22; recombined ~39; onsets + continuations ~14; 8 fits ~64; export and build 5-10; report, package, check,
upload ~20 -> about 2 h 45; the rest is reserve.

## 4. Guards and failures (all in the driver; minutes left are read from `task.json` -> `run.deadline`)

- A start directory missing: the driver stops at once and names it; copy it and start the driver again.
- VO is not started with fewer than 175 minutes left and is stopped when fewer than 25 remain. A VO step that fails
  stops VO; a driver started again takes VO up from its last finished step and delivers it. If VO cannot finish,
  nothing is uploaded and the run ends with `axonry-done`.
- The upload: `axonry-check` runs when the sandbox has it and at least 25 minutes are left; each send is tried up to
  three times; the status log says "UPLOADED ..." when it stands.

## 5. Credits and files

- **The start**: `0009576150-0005` (#130, "own-close-carried-as-change", published), used as it is. By its METHOD it
  rests on `0009570269-0024` (#116, the champion of its time), the published layer of `0009569144-0010` (#111) added
  as a change, and the request gains of `0009552831-0015` fitted with the scripts and steps of `0009568726-0005` (#106);
  its way of merging follows task arithmetic (Ilharco et al., 2022), with TIES-merging (Yadav et al., 2023) on where
  merged changes interfere. Nothing of these is re-derived or changed here.
- **VO**: #107 (`0009568831-0002`) and #106 (`0009568726-0005`), which publish the same files byte for byte, for
  `branch2.py` (`onset_branch.py` = it plus the kind `onset`), `pref_gains2.py`, `fit_gains.py`, `merge_gains.py` and
  `shard_gen.py`; #94 (`0009559255-0006`) for `mkreq.py`, `genreq.py` and the item-index patch in `with_req.py`; #96
  (`0009565562-0012`) as the closest prior idea. `if_onset.py`, `run_onset.py`, `report_onset.py`, `up_onset.py`: ours.
- `run_a.py` (the driver: the start, VO, the report, the package check, the upload): ours. Its recall steps are off in
  this entry (NO_R); the recall scripts it ships are not called.

## 6. Data for the run

Use only the toolkit's `practice:` sets, the bundle's start directories and the models' own generations; never load a
`bench:` set, and download nothing. The start's arrays are read from its published directory; every array of R is
fitted in the run.
