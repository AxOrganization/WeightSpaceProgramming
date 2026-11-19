# Damp the copy circuit the remaining loops run on

## The problem
After the parent's loop edit the model still runs out of tokens on many answers. Measure it first on the practice
sets, per source (GSM8K / MATH train, MMLU-Pro and the IFEval-style practice items; never the benchmarks): cut
share, the share of cut answers that end in a repeated block (`cut_loop`), and accuracy when finished. Where
finished answers are far more accurate than all answers, the remaining loops still hide an ability the model has.

Look at where those loops begin: the model starts a sentence it has already written and then writes the same block
again and again. Writing out text that is already in the context is copying. A loop is the copy machinery running
on the model's own reasoning.

## Why we think it happens
Reproducing earlier text is done by a small set of attention heads (in this hybrid model both full-attention heads,
whose outputs are column blocks of `self_attn.o_proj`, and linear-attention value heads, column blocks of
`linear_attn.out_proj`). Once the text starts to repeat, these heads find the earlier occurrence and make its
continuation the most likely next text; that is what turns a single repeated sentence into a loop that never ends.
A decoding-side repetition penalty is the blunt form of the same fix: it reduces copying everywhere. It is expected
to shorten loops on reasoning tasks but to hurt where repeating is required (instruction following: repeating the
prompt's words, keeping a format), so it is not a model fix and its cost must be watched.

The weight-space version is to weaken only the heads that are specific to copying: heads whose removal makes
repeated text much harder to predict while leaving the prediction of normal, non-repeated reasoning almost
unchanged. Copying is redundant (many heads share it) and also needed for normal reasoning (numbers, names,
quoted terms), so three things matter as much as the choice of heads:
- **damp, do not remove**: zeroing heads that copy is expected to break reasoning; a partial scale of their output
  columns lowers the pull of the earlier occurrence without deleting the ability;
- **select by the ratio, not by size**: the heads with the largest copy cost are usually also the most important
  for normal text; keep only those whose normal cost is small next to their copy cost;
- **protect instruction following**: following a format instruction is also a copy, from the prompt into the
  answer (keywords, casing, a closing phrase, a sentence count). A head that loops copy through is not
  automatically one that instructions need; measure both and keep the heads instructions need.

## Approach
Plan the GPU time first (see the skill's time budget). With four GPUs, run independent generations in parallel,
one per GPU: a 128-problem DEV run of this model takes roughly 10-15 minutes on one GPU, a guard run (step 6)
about 10, the head scan (step 2) about 10. That leaves room for a dose series, controls, a guard check of every
serious candidate and a holdout. All choices are made on practice sets.

1. **Screen the starting model** on all four practice kinds: `init` a case with `--parent` the parent edit and
   `--practice gsm8k_train math_train mmlupro ifsynth` (e.g. `--dev 256 --holdout 256 --subset dev128=dev:128`),
   then `run <case> base` on the DEV problems (split the list over GPUs with extra id lists under `sets/` if that is
   faster). Read per source: cut share, `cut_loop`, accuracy when finished, and read a few cut answers.
2. **Score every head causally.** `python derive/copyheads.py <case> --normal-run base__<set> ...` takes spans of
   the starting model's own correct reasoning, shows each span twice, and zeroes one head at a time: copy cost =
   loss increase on the second occurrence, normal cost = loss increase on the first (non-repeated) text,
   instruction cost = loss increase on the model's own correct answers to IFEval-style practice items (so the base
   run must include that source). It writes the table to `cases/<case>/results/copyheads.json`. Increase `--n` /
   `--length` if the ranking is unstable between two seeds of spans.
3. **Choose the candidate set**: heads with a clear copy cost and a high ratio of copy cost to the larger of the
   normal and instruction costs; exclude heads whose normal or instruction cost is large in absolute terms (they
   carry general reading or instruction recall, not loop copying), and expect the set to span several layers and
   both head types.
4. **Dose series with controls.** `python derive/copyheads.py <case> --spec NAME --heads L.H ... --factor F` writes
   a `scale_cols` spec on those heads' output columns, stacked on the parent. Write a few factors (a series between
   strong damping and none) and two controls: the same number of heads with the highest normal cost and low ratio,
   at the same factor, and a set of random heads. The removal (factor 0) of the candidate set is a useful negative
   control too.
5. **Generate, paired, per source.** `run` each spec on the same DEV problems as the base, then
   `python derive/persource.py <case> <edits...> --sets <sets>` prints the paired gain per practice source and the
   cut share. Keep an edit only if it cuts loops and gains on several reasoning sources and **loses on none**. Prefer
   the mildest factor and the fewest heads that keep the gain. Upload the first edit that passes, then improve.
6. **Instruction following is the binding constraint; give it its own guard set.** A DEV slice of a few dozen
   IFEval-style items cannot see a loss of a few points, and that is the size of loss to fear: the decision allows
   no benchmark to drop beyond noise. Build a separate guard list of at least ~150 IFEval-style practice items not in
   DEV or HOLDOUT (an id list under `sets/`), and run the base and every serious candidate on it with two samples
   per item; these answers are short and rarely cut, so the runs are fast. The teacher-forced instruction cost of
   step 2 is only a pre-screen: confirm on the guard set. If a candidate loses there, first make the whole set
   milder (a factor closer to 1): the cost to instructions grows with the strength of the damping, and the loop
   gain often survives a milder factor. Only then look for single heads, by leaving one out at a time (or giving it
   a milder factor than the rest) and re-running the guard set. Report the guard result of every candidate.
7. **Refine** if time remains: drop or add heads one at a time (by the table's ratio) and keep a change only with
   a paired gain on DEV and no loss on the guard set; try a per-head factor in proportion to the head's copy
   specificity.
8. **Freeze, confirm on a holdout** of practice items never used for any choice (two samples per problem if time
   allows; the decision is made on four), and upload with a report: the head table, the chosen heads and factors,
   the controls, the per-source paired results and every attempt that did not work.

## What to expect
Fewer answers cut on the reasoning tasks where the parent still loops (math, GSM8K, MMLU-Pro), so a gain there, and
instruction following kept within noise. Removing the heads outright and damping the high-normal-cost heads should
both lose; if the candidate set behaves like the controls, the loops are not copy-driven in this model and the
report should say so instead of forcing an edit.
