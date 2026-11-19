# doubt-readout: make the last layer less eager to open yet another check

## The failure
The parent (the loop-removal champion) fixed the thinking loop that kept the base model from ever answering, but a
large share of its traces still never finish: the model overthinks. It reaches an answer, then opens round after
round of checking ("Wait, ...", "But let me double-check ...", "Hmm, maybe ...") and eventually slides into a
repetition that runs into the 32k token limit. When a trace does finish, the answer is usually right, so every
unfinished trace is mostly lost accuracy. Measure it on the parent's own practice traces: the share of cut traces and
the accuracy of the finished ones.

## Suspected cause
Whether the next sentence re-checks or moves on is a token decision at the readout. At sentence boundaries the last
layer writes a "doubt" readout direction into the residual stream: the direction that raises the logits of the
words that open a re-check against the words that continue or conclude. Some last-layer MLP neurons act as a steady
sentence-boundary bias along it: they fire at a similar level before every sentence opener, so they make re-checking
a little more likely everywhere; over thousands of sentences that adds up to more check cycles, longer traces and
more chances to fall into a repetition. The parent's edit removed a loop-state direction from attention outputs in
the middle of the network; this readout bias is a different mechanism (check: the cosine between the parent's
ablated directions and the readout direction should be near zero).

The fix: make such a boundary neuron stop writing the doubt readout. For the chosen MLP neuron j, replace its output
column by `W_down[:, j] - alpha * s_j * u` (u = the unit doubt readout direction, s_j the sign of
the neuron's activation before openers): wherever it fires, it now pushes towards moving on. One column, a few
kilobytes of payload.

What NOT to do (measured on this model): do not pick a neuron that fires mainly at late re-checks inside unfinished
traces and suppress doubt only there. In a repetition, a doubt opener is the model's way out; removing it exactly
there traps the model in verbatim cycles (unfinished traces and accuracy got clearly worse). The useful edit lowers
re-checking about equally early and late, so fewer cycles start, while a late re-check stays possible.

## Method
The code is in `derive/` (uses the wsp toolkit; run it from the WeightSpaceProgramming checkout with the toolkit's
environment, like `python -m wsp.case`). Keep `PROGRESS.md` current; each step below fits in one background job.

0. **Case and parent traces.** `python -m wsp.case init <case> --profile qwen35-0.8b-think --include
   cases/screening/sets/screen64.json --subset dev128=dev:128 --subset holdout256=holdout:256 --parent
   parent/edit/edit.json`, then `python -m wsp.case run <case> base dev128 --n 2` (the parent model, 32k tokens,
   two samples per problem; about 35 min on one GPU). In parallel on other GPUs (`CUDA_VISIBLE_DEVICES`): `python derive/ifs_set.py <case> --n 128` then
   `python -m wsp.case run <case> base ifs128 --n 2` (instruction-following guard set from practice:ifsynth), and
   `python -m wsp.case run <case> base holdout256` if time allows.
1. **The readout direction.** `python derive/onsets.py <case> --tags dev128`. Doubt openers (words that open a
   re-check or a change of course: Wait, However, But, Maybe, Perhaps, Hmm, Alternatively, Actually) and progress
   openers (words that continue or conclude: Therefore, Thus, So, Now, The, First, Next, Let) are defined by their
   role; the script keeps the single-token ones, prints how often the parent's traces use them, and builds
   `u = normalize(mean E[doubt] * g - mean E[progress] * g)` from the output embedding E and the final norm's
   effective gain g (Qwen3.5: 1 + weight). Do not choose the words by enrichment in unfinished traces: unfinished
   traces are mostly hard problems (topic words look enriched) and their second halves are mostly verbatim
   repetition, where every opener is rarer.
2. **The neurons.** `python derive/neurons.py <case> --tags dev128` reads the MLP activations of the last four
   layers at the position just before each opener (late doubt openers in the second half of unfinished traces,
   early doubt openers in the first 30% of finished traces, progress openers in finished traces) and ranks the
   neurons by `level = mean_late a_j * (u . W_down[:, j])`, the push each one gives the doubt readout before
   openers (sign-aware). From the top ten writers of each scanned layer, take the boundary-bias neurons: activation of
   the same sign and similar size at late, early and progress positions (within a factor of about two), whose
   column has a clearly positive component along s_j * u. Skip neurons whose |a_late| is several times |a_early| and
   |a_prog|: they are late-only. (On this model boundary neurons of the final layer and of the layers just before it
   both worked once dosed into the window of step 4; let the ranking and step 4 decide.)
3. **Candidate edits.** `python derive/make_edit.py <case> --layer <l> --cols <j> --alphas 0.5 0.75 1 1.25` for the
   best boundary neuron, the same for the second-best one (the useful alpha scales with the neuron's activation size
   and its alignment with u: if every dose overshoots the window of step 4, halve the alphas; if every dose
   undershoots, raise them; the window, not alpha, defines the dose), and `--random-control` for the best one at alpha 1 (random
   unit direction orthogonal to u, same column and dose). One neuron per edit: the useful dose window is narrow, and
   two boundary neurons together roughly double the suppression (in our tests that overshot badly: more unfinished
   traces and lower accuracy everywhere). The specs (`set_cols`, column values inline) go to `cases/<case>/edits/`;
   keep the values inline: the upload packager copies u/cols/rows arrays but not `values`.
4. **Selectivity screen and dose window (minutes).** `python derive/neurons.py <case> --model <edit> --logp-only`
   for every candidate and for `base`: mean log-prob of the opener at the late / early / progress positions (same
   positions for every model). The doubt log-prob should fall smoothly with the dose, about equally at early and
   late positions. Keep the doses whose EARLY doubt log-prob falls by about 1 to 2.2 nats with the progress
   log-prob within 0.1 nat of the parent's. Reject drops of 2.5 nats or more and doses where the progress log-prob
   starts to fall: in our tests a 1.4-nat drop gave the best balance, a 2-nat drop more gain on math but a risk to
   instruction following, and drops above about 3 nats (e.g. two boundary neurons at once) made everything worse.
   Reject a candidate whose late drop is several times its early drop (late-only suppression: the trap above).
   The random control must change almost nothing; if it moves as much, recheck steps 1-2.
5. **Generation tests.** `python -m wsp.case run <case> <edit> dev128 --n 2` for two or three doses inside the
   window (one per GPU, in parallel; two samples per problem halve the noise) and `run <case> <edit> ifs128 --n 2` for each;
   the parent's dev128 run from step 0 already has two samples. The report pairs them with the parent; also look at each practice
   source separately (`by_source`) and count the verbatim repetitions among the cut traces (`kind` == `cut_loop`).
   Pick the dose with the largest dev128 gain that passes every guard: accuracy on EACH dev128 source and on ifs128
   not lower than the parent's by more than about one standard error, dev128 `cut_loop` not higher, and the ifs128
   cut rate not above about 12%. Expect the ifs128 cut rate to rise a few points at every useful dose (a few more
   checklist repetitions) without an accuracy cost; a dose that pushes it far higher is breaking instruction
   following. When two doses are close, take the lower one.
   Competition-style math gains first (many of its traces are cut); GSM8K-style word problems can show fewer cut
   traces but more wrong finished answers when the dose is too strong.
6. **Deliver.** Upload the first candidate that passes steps 4-5 at once
   (`python -m wsp.upload <case> <edit> --task task.json`), then freeze the best one, run it and the parent on
   holdout256 if time allows, and re-upload. The report lists the opener words, the chosen neurons with their
   activation profiles, the selectivity numbers of every candidate and the control, and the dev/holdout/ifs results.

## Time plan (4 GPUs, 6 h)
- 0:00-0:35  step 0 (dev128 on GPU0, ifs128 on GPU1, holdout256 on GPU2 and 3 split in two halves if wanted).
- 0:35-0:50  steps 1-3 (CPU and one GPU, minutes).
- 0:50-1:00  step 4.
- 1:00-1:40  step 5 for three candidates in parallel; upload the best at once.
- 1:40-4:00  second round around the best dose (and the next boundary neuron of the ranking if the first is weak);
  holdout256 of the frozen edit; re-upload. Leave the last 30 minutes for the final upload and report.

## What to expect
On dev128, unfinished traces should fall clearly (in our tests by about half on competition-style math) and accuracy
rise by most of the rescued share, with finished-trace accuracy about the same; GSM8K-style problems gain less (fewer
of them are cut). Verbatim repetitions must not increase. The edit changes one column of one matrix (relative
change a few percent).

## If it does not work
- No boundary neuron among the top writers of the last layer: look at the second-to-last layer.
- No dose of the best neuron lands in the 1-2.2 nat window: try the second neuron, or finer steps (0.6, 0.8, 0.9).
- The random control moves the log-probs as much as the edit: the opener lists or the direction are wrong (step 1).
- Every dose that helps dev128 hurts ifs128 or adds verbatim repetitions: take the lowest dose that passes the guards;
  if none does, deliver the best guard-passing candidate and report the trade-off.
- The playbook advises against editing the last layer: that lesson is about ablating a direction from all of a
  matrix's columns; here only one or two columns change, and steps 4-5 decide on measurements.
