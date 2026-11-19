# Cycles that survive the loop edit: an on-policy branch contrast at real cycle onsets

This entry stacks on the published loop-removal edit (`parent` in entry.json). It removes what that edit left: the
model still falls back into earlier content - sometimes verbatim, often reworded - and runs to the token limit.

## The problem
After the parent edit most answers no longer loop verbatim, yet many still end cut at the 32k token limit. Measure
this first, on the practice sets only (GSM8K train, MATH train, MMLU-Pro practice and the IFEval-style `ifsynth`;
never the benchmarks), with the parent as "base" and at least 2 samples per problem:
- the share of answers cut at the limit, per source, and whether the cut came inside the thinking or after
  `</think>` (an answer that itself cycles);
- for every cut answer, where its cycling starts and how it cycles (`derive/onsets.py find`): verbatim (the
  toolkit's `tail_dup`) or reworded - the same results, numbers and option letters stated again and again in new
  words. The model-card sampling has `presence_penalty` 1.5, which pushes a cycle out of verbatim repetition into
  rewording, so a low `tail_dup` does not mean the model moved on;
- how many cut answers are on problems that another sample of the same problem finishes correctly (the model can
  solve them; the cycle hides it), and the accuracy of the answers that finish.
Read about ten onsets by hand, from several sources, before going on: the onset should be where the text starts
going back over itself, not a normal re-check. If cycles are rare among the cut answers (say under a quarter on
every source), this method does not apply: report that instead of editing.

## Why we think it happens
A cycle is one state - "go back to something already said instead of moving on" - entered in several ways. The
parent's direction was learned from synthetic copies (a correct math trace's last sentences repeated) and removed
from two attention outputs only. The cycles that remain start later in long traces, in other task formats, in
rewording or after the thinking closed, so other components still write that state. If so, the real decision point
is the onset itself, and it is best isolated by the model's own two futures from the very same point: one that
falls back into the cycle, one that moves on. Same prefix, same content, same position in the trace; only the
behavior differs, and both sides are the model's own samples, so the contrast measures the state the model is
actually in when it starts to cycle. If no stable direction comes out of these branches, the remaining cut answers
are not one removable behavior, and the run should say so rather than edit blindly.

## Approach
1. **Case on the parent.** `python -m wsp.case init <case> --profile qwen35-0.8b-think --parent
   parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --subset dev128=dev:128 --subset
   holdout256=holdout:256`. Generation is the bottleneck and `run` uses one GPU, so spread every set over the 4
   GPUs: write one id list per GPU (e.g. by source), run `CUDA_VISIBLE_DEVICES=<i> python -m wsp.gen --profile
   qwen35-0.8b-think --model <model dir> --ids <ids_i.json> --n 2 --out <part_i.jsonl>` on each, concatenate the
   parts into `$WSP_WORK/runs/<case>/<model>__<set>.jsonl`, then `python -m wsp.case run <case> <model> <set> --n
   2` grades and pairs that file without generating again (the per-problem seeds are the same). The model dir of
   "base" (the parent here) or of an edit is `wsp.case.model_path(<case>, <model>)`, built on first use. Start with
   `base` on `dev`: it is the longest step.
2. **Onsets.** `python derive/onsets.py find <runs of the parent> --out onsets.jsonl` (copy `derive/` to the
   sandbox's `/workspace/work/` and run it there). It marks, in every cut answer, the first sentence from which
   most later sentences add nothing new (few new word shingles, or only numbers already stated), up to the cut.
   Tune `--share` / `--min-chars` by reading examples, not by results. Keep onsets of both phases (thinking and
   answer).
3. **Branches at each onset.** `python derive/onsets.py branch onsets.jsonl --runs <same runs> --model <parent model
   dir> --profile qwen35-0.8b-think --out branches.jsonl --k 8`: 8 continuations of each onset prefix, sampled from
   the parent with its model-card sampling. vLLM applies the presence penalty to generated tokens only, so the
   penalty the prefix had already built up is carried over as a logit bias (on by default). Each continuation
   gets its returning share: how many of its sentences go back to earlier content (zero if it closes the
   thinking). The continuations of one prefix differ a lot in it, and that difference is the contrast.
4. **Pairs.** `python derive/onsets.py pairs branches.jsonl --case <case> --out pairs_branch --heldout 0.25`:
   same-prefix pairs, good = the continuation that returns least, bad = the one that returns most, when they
   differ clearly (`--min-gap`), measured over the first few hundred characters of the continuation; problems
   are split, a quarter held out for selectivity only. Check the mix by source and phase. Build the toolkit's
   synthetic loop pairs from the parent's own correct traces as well (`python -m wsp.case pairs <case> --sets dev
   --out pairs_synth`, held-out ones with `--exclude-pairs pairs_synth --out pairs_synth_heldout`): they are the
   second source of the contrast. In a first trial on the parent, 256 problems x 2 samples gave only ~50 branch
   pairs and a direction of moderate stability (split-half ~0.6), so plan for more onsets (more problems or
   samples per problem) from the start.
5. **Direction.** `python derive/onsets.py extract <case> --pairs pairs_branch` on the parent (as `wsp.case
   extract`, same output files, but reading whole texts: the toolkit's probe keeps only the last 16k tokens, and
   onset prefixes are often longer): per layer the mean difference (cycle minus progress), split-half stability,
   separation, writer scores. Do the same for `pairs_synth` and for the union of both files, and keep the
   direction that is stable (split-half cosine above about 0.8) and separates the held-out pairs of both kinds;
   report the cosine between the branch and synthetic directions per layer. The toolkit's record of a second loop
   edit on this base (`cases/qwen35-0.8b-think_loop2/NOTES.md`: a residual direction removed from MLP writers on
   top of a first loop edit) is the closest evidence for this step: read it first. Compare the chosen direction
   with the parent's own direction (`parent/edit/edit.npz`, cosine per layer) and report it: the new layer must
   remove what the parent left, not repeat it.
6. **Writers.** Rank who writes along the direction on the cycle side: MLP units (`extract`'s writer scores) and,
   per layer, the attention outputs. Prefer few strong writers in the middle of the network; never the last
   layer's MLP; the tensors the parent edited can be edited again only if they still write most.
7. **Edit and controls.** `python -m wsp.case edit <case> --layers ... --topk ... --alpha ...`, always with
   `--random-control` on the same writers and dose. `--targets o_proj` exists only in the full-attention layers
   (`self_attn.o_proj`, one layer in four); for the linear-attention layers write the `dir_ablate` op on
   `linear_attn.out_proj` by hand into `cases/<case>/edits/<name>.json` (format in the skill). Then `python
   derive/onsets.py selectivity <case> <edit> --pairs pairs_branch_heldout`: the cycle side's log-prob must drop
   clearly, the progress side must stay within ~0.02 nats/token, and the random control must do neither. Screen
   many variants this way (minutes each); generate only the best few. In a first trial on the parent, a branch
   direction from ~50 pairs, removed from the top MLP writers of a few middle layers, lowered the cycle side only
   about twice as much as the progress side (~0.02 vs ~0.01 nats/token) and did not improve generation; stronger
   doses pushed the progress side past the limit. Selectivity of that kind is not enough: get a cleaner direction
   first (more pairs, the synthetic pairs, the attention outputs as writers).
8. **Dose on real generation.** The best few on `dev128` with 2 samples (one edit per GPU, as in step 1), paired
   with the parent automatically. Choose the smallest edit (fewest writers, lowest dose) that lowers the cut share
   on every source and does not lower the accuracy of the answers that finish on any source. A source that gets
   worse decides against the edit even if the mean rises: the evaluation needs every task type to hold. With 128
   problems x 2 samples a source moves by up to ~5 points from noise alone, so run the last two or three
   candidates on as many problems as the time allows before choosing.
9. **Upload early**, as soon as one edit passes steps 7-8: package it (`python -m wsp.upload <case> <edit> --report
   <REPORT.md> --dry-run`), check the package (`axonry-check $WSP_WORK/deliver/<case>__<edit>/edit --task
   task.json`), then upload (`python -m wsp.upload <case> <edit> --report <REPORT.md> --task task.json`), and
   improve. **Freeze** the best, confirm it on `holdout256` against the parent, and upload the final edit with a
   report of every measurement: cut shares by source and phase, the branch labels, the direction's stability and
   its cosine with the parent's direction, the writers, selectivity, random controls, doses tried.

## What to expect
Fewer answers cut at the limit on every task type, also where the parent left reworded cycles in math and cycles
inside instruction-following answers; answers that finish stay as accurate as before, and some get longer because
harder problems now finish. If selectivity shows the progress side dropping, narrow the writers or the dose before
giving up.
