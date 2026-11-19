# Real loops: remove the champion's own loop state from its MLP writers

## The problem
The starting model (the parent: the published loop-removal edit) rarely repeats itself word for word any more, but
on long problems its thinking still does not end: it re-derives, doubts the wording of the question ("maybe the
question means ..."), re-checks, paraphrases the same cycle, until the token limit cuts it. The thinking never closes
and the answer is lost, although the answers it does finish are nearly always right. Measure it first on the practice
sets (all four task types: GSM8K / MATH train, MMLU-Pro, IFEval-style): the share of answers cut at the limit, how many
of those end in a repeated block and how many do not, and the accuracy of finished answers, by task type. Read some of
the cut traces.

The parent's edit removed a loop direction (found from synthetic repeats) from the **attention outputs** of two
layers. Its MLP units, which write most of the loop state in this model family (`playbook.md`), are untouched; that is
the lever here.

## Why we think it happens, and what we change
When the parent wanders without converging, its residual stream is in a recognisable state: the state of its own
looping traces (the repeated suffix of a cut thinking) differs strongly from its state in traces it gets right, layer
by layer, and a few hundred MLP units in the middle-to-late layers write most of that difference. The difference of
class means is not only "repetition": it is the state of a thinking that does not converge (deep in a long trace,
unresolved), which is what has to go. Removing that direction from its strongest writers (`dir_ablate`, side "out", on
`down_proj` columns) should make the model leave the state, close its thinking and answer, on the long problems where
it now runs to the limit. A random direction of the same size must not do this (control).

## Approach
Everything uses the starting model's own generations on practice problems; benchmarks are never used.
`derive/realloops.py` implements the steps around the toolkit (`python derive/realloops.py <step> -h`); it uses the
toolkit's case layout, so `wsp.case edit / run / report` work as usual and every generation is paired with the
starting model.

1. **Case.** `python -m wsp.case init <case> --profile qwen35-0.8b-think --parent <parent edit.json>
   --practice gsm8k_train math_train mmlupro ifsynth --dev 768 --holdout 768 --subset dev128=dev:128
   --subset holdout256=holdout:256` (all four task types).
2. **Traces.** `python derive/realloops.py gen <case> base dev --shards 4` (the starting model, one sample per problem,
   all GPUs). The parent loops rarely, so you need many traces: at least ~15 looping traces in the fit part (step 3
   prints the count); generate more practice problems if there are fewer.
3. **Label** (`label <case> --sets dev`): in-loop positions = the repeated suffix of a cut thinking (token n-grams
   already seen, from their third occurrence on); normal positions = every position of the traces scored right
   (traces that end cut or wrong are neither). Fit / held-out split by problem.
4. **Direction and writers** (`collect <case>`, then `directions <case>`): per layer the normalized difference of the
   mean residual state after the layer in loops and in correct traces, and each MLP unit's writer score (its mean
   activation difference times what it writes along the direction), in the toolkit's `directions.npz` format. The
   table shows where the state is written: expect the largest writer totals in the middle-to-late layers.
5. **Candidates** with the toolkit: `python -m wsp.case edit <case> --layers ... --topk ... --alpha ...` on a contiguous
   block of the layers where the writers concentrate (never the last layer), a series from small to large edits
   (alpha from weak to strong, and the number of writers per layer from few to many), each with `--random-control`.
6. **Selectivity** (`check <case> <edit>`, minutes, held-out traces): the log-prob of the loops' own continuations
   falls clearly, several times more than that of whole correct traces; the random control is null.
7. **Choose on long problems, with enough samples.** The gain is where the parent runs to the limit: on the long
   (MATH-type) problems, and it grows with their difficulty; on easy problems it is invisible or slightly negative.
   Decide on a set of MATH-train problems from the holdout split (`derive/realloops.py subset <case> --split holdout
   --sources math_train --n 192 --name hmath`), at least 2 samples per problem: the starting model first
   (`derive/realloops.py gen <case> base hmath --n 2`; it runs longest, its traces go to the limit), then each candidate
   next to its random control (`derive/realloops.py gen <case> <edit> hmath --n 2`). `gen` spreads one run over all
   GPUs: run them one after another, or give each run its own GPU with `--gpus <i>`, never two runs on the same GPU.
   Accuracy and cut share there first. Among the candidates within about one standard error of the best there, prefer
   the one that cuts about half as many answers as the parent: weaker doses leave long answers cut (and give up the
   gain on the other long tasks), stronger ones lose more answers the parent had right and cost more on instruction
   following. Then `holdout256` (all task types, 2 samples) for the chosen ones: the other task types must not be
   worse beyond noise. One sample per problem is too noisy to choose (any change of the weights re-samples every
   trace; a do-nothing difference flips 10-20% of answers both ways). Read the losses: answers the parent got right
   that the edit loses (it concludes too early, or wrongly) mark a dose that is too strong, and so do cut traces that
   now end in a word-for-word repeat. On instruction-following items, read the final answers (see below).
8. Freeze, confirm on the rest of the holdout split, report, upload (`axonry-check` first). Upload the first candidate
   that gains on the MATH-type items against both the starting model and its control early, and re-upload
   improvements.

Time plan (6 h, 4 GPUs): traces ~30 min; label, collect, directions ~20 min; on the decision set (192 problems x 2,
all GPUs) the starting model ~15-25 min, each candidate or control ~10-15 min; `holdout256` x 2 ~20 min a run. Keep
the last 30 min for the final check and upload.

## What to expect (a pilot on the same parent, practice problems only)
- The parent's own traces: MATH 46% cut at the limit, almost all without a repeated block, and 96% right when finished;
  GSM8K 18% cut; MMLU-Pro 23%; IFEval-style 7%. ~6% of all traces end in a repeated block: enough to label.
- The direction's writers concentrate in the middle-to-late layers below the last. The direction is stable: from
  all 768 dev traces and from 360 of them it agreed to a cosine of 0.99 in every layer of the block, with ~85% of the
  top writers shared.
- On a MATH-train decision set from the holdout split (64 problems x 2), the parent was right on 37.5% of the answers
  and cut on 55%. Ablating the direction from the top writers of a contiguous block of those layers gained 6 to 11
  points over a range of doses and writer counts (standard errors 3-4 points; cut share down to 27-36%); the stronger
  doses cut more but lost more of the answers the parent had right (7 vs 4 of 128) and ended more of their cut traces
  in word-for-word repeats. A random direction on the same writers at a moderate dose cut nothing (61% cut) and lost
  4.7 points.
- On all task types (holdout, 128 problems x 2) a moderate dose cut the share of answers cut at the limit from 26% to
  13% and tokens per answer by a quarter: MATH-type +7.8 points, the other task types a few points lower, within
  noise. The instruction-following items lost about 6 points at the moderate dose and 11 at a strong one (a random
  direction: 2): that cost belongs to the direction and grows with the dose. A random direction on the same writers
  at a strong dose cut nothing (26% -> 28%) and lost 3.5 points overall. Expect larger gains on harder problems (where
  the parent runs to the limit more often) than on easy ones.
- The final answer after the thinking gets a little shorter (10-25% in the median). On instruction-following items
  that can cost a few points; keep the dose moderate and do not trade instruction following away for the long
  problems.

## What does not work (measured on the same parent, practice problems only)
- Ending the thinking by force when a loop is detected (a one-unit gate): it fires right, but the forced answers are
  almost never right (0 of 12).
- DPO on "conclude here" vs "go on" preference pairs, trained as one gain per MLP unit: the model concludes too early
  (-6.2 points at full strength, nothing at a quarter).
- A second round removing a fresh loop direction from the same MLP units a parent has already edited undoes the
  parent's correction (loops come back, -15 points): relevant only if the parent edits MLP writers (this one does not).
