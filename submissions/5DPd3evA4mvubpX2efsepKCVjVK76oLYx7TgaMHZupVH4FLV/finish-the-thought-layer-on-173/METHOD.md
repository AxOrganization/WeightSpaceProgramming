# The finish-the-thought layer derived from the base, delivered on the published 0009598520-0002 (Qwen3.5-2B)

This entry's parent is the published edit `0009598520-0002` ("finish-after-the-loop-edit", PR #173: the loop
removal, then fitted directions that make the loop-free model finish its own right answers). The entry adds one
layer onto it: the layer of the published entry `0009599014-0016` ("finish-the-thought", PR #178), derived in this
run **from the base model**, exactly as that entry derives it, with that entry's code, and then delivered unchanged
onto `0009598520-0002`.

## Sources and credits
- **The start:** `0009598520-0002` (PR #173). Its published edit is inherited as is. Nothing of it is derived again.
- **The layer, its idea, its code and its numbers in the method:** `0009599014-0016` ("finish-the-thought", PR #178,
  hotkey 5FyCs7tdcqPLn8p5LJManbqzZes6JsrbNji2gfCgY5fkNFFP). Its two parts are
  (a) self-distilled gains: one gain per MLP unit and attention head of the output projections (`scale_cols`), fitted
  with the toolkit's `wsp.fit` on the model's own finished, right answers, three seeds averaged in log space. The
  unit-and-head gain method was published first as `0009538688-0019` (self-distilled gains, 0.8B line) and carried
  on this base by `0009598598-0002`.
  (b) Wrap-up units: in the last four layers, 24 MLP units per layer rewritten (`set_rows` / `set_cols`) so that in
  the last tenth of the token budget the thinking closes at a line end and, where a boxed answer is asked, a short
  stem and the model's own immediate answer follow. Everywhere else the units are silent.
  `derive/run.py`, `derive/units.py`, `derive/gen.py`, `derive/family.py` and `derive/loops.py` are that entry's
  files, byte for byte as published with that entry. Its METHOD describes every step they run. The step list below
  repeats only what this entry does differently.
- **The toolkit** (WeightSpaceProgramming): prompts, seeds, sampling, grading, `wsp.fit`, building, packaging.
- **Ours:** three things. (1) The finding that this layer, fitted on the base, composes with `0009598520-0002`:
  on the parent the layer removes the answers the parent still loses to the token limit, and the instruction
  answers hold (section 2, practice data). (2) The driver: set up from the
  base whatever start the task offers, run the derivation unchanged, then deliver its edit on the parent
  (`derive/compose.py`). (3) The fixed delivery rule: the edit the derivation chose, at its full size, nothing
  chosen afterwards.

How this differs from entries with related layers on the same parent:
- `wrap-up-units-on-the-champion` (PR #181) refits the wrap-up units on the parent's own answers and delivers them
  alone, without gains.
- `focused-finishing-on-173` (PR #180) fits self-distilled gains on the parent's own answers, one maths-only and one
  multiple-choice-only layer, with every other task type anchored.
- `finish-on-the-champion` (PR #182) runs the finish-the-thought method again with the champion as its start, so
  both parts are refitted on the champion's answers.
This entry refits nothing on the parent. Both parts, gains and units, are fitted on the base's own answers (where
the base still loses about a third of its answers to the limit, so the units have many cut traces to learn from),
and the whole layer is added onto the parent.

## 1. The behaviour to change
On practice problems the parent still loses answers to the token limit with the thinking open, mostly in maths and
multiple choice (section 2). An answer cut there scores nothing. `0009599014-0016` showed on the base that its layer
ends nearly every such answer, either by closing the thinking late and answering or by distilling the base's own
economical answers. It also reported the cost: in one of its runs a leaky unit fit lowered instruction-following
answers on the holdout. The parent's own layers (loop removal, then finishing directions fitted with a tether to the start) keep
instruction answers.

## 2. Diagnosis on practice data (ours)
The model this method delivers (the published `0009599014-0016` layer on the published `0009598520-0002`), against
`0009598520-0002`. Practice problems of the toolkit, 80 per family, two answers per problem, the profile's sampling
and full budget, paired by problem and seed.

| family | problems | parent right | parent cut | delivered right | delivered cut | change, points (paired) |
|---|---|---|---|---|---|---|
| math (math_train) | 80 | 85.0% | 12.5% | 89.4% | 1.2% | +4.4 +- 2.2 |
| gsm8k (gsm8k_train) | 80 | 95.0% | 0.6% | 91.9% | 0.0% | -3.1 +- 1.6 |
| mmlupro | 80 | 64.4% | 9.4% | 70.0% | 0.0% | +5.6 +- 2.7 |
| mmlucf | 80 | 71.9% | 0.6% | 66.9% | 0.0% | -5.0 +- 2.9 |
| bbhsynth | 80 | 96.2% | 1.9% | 93.8% | 0.0% | -2.5 +- 2.3 |
| ifsynth | 80 | 92.5% | 0.0% | 95.6% | 0.0% | +3.1 +- 2.4 |
| tuluif | 80 | 90.6% | 0.0% | 94.4% | 0.0% | +3.8 +- 2.9 |
| all | 560 | 85.1% | 3.6% | 86.0% | 0.2% | +0.9 +- 0.9 |

(Cut: the answer reached the token limit. Right: graded right by the toolkit and not cut. Change: delivered minus
parent, mean over problems of the two-answer difference, with its standard error.)

What this shows. The parent still loses answers to the limit where the thinking is longest: 12.5% of its maths
answers and 9.4% of its multiple-choice answers in the exam-style family are cut. With the layer on top, almost
none are cut (1.2% and 0%), and those two families gain about 4 and 6 points. The instruction-following families,
where the layer cost answers in one of `0009599014-0016`'s runs on another start, do not lose on this parent
(ifsynth +3.1, tuluif +3.8). The parent's finishing directions and the layer act on different answers. The units
act only deep in the budget, the parent's directions shape the answers that finish, and the gains re-weight units
and heads that neither of the parent's layers fixes.
Where the parent cuts almost nothing (gsm8k, the generated puzzles, and the general multiple-choice family), the
layer has nothing to end and its gains move the policy toward the base's answers instead. These families lose 2.5 to
5 points here, each on 80 problems and within about twice its standard error. Overall the delivered model is
+0.9 +- 0.9 points on these 560 problems, with the cut share 3.6% -> 0.2%. Expect the gain in the families that
still lose answers to the limit, and a small loss where the parent already finishes.

## 3. Steps (in the run)
Work in the sandbox under `/workspace/work` with the bundle's `task.json` and `submission/` there.
0. Copy the parent start `0009598520-0002` from the bundle to the sandbox under its id (`0009598520-0002/edit/`, next
   to `task.json`). `0009599014-0016`'s METHOD asks the same for any start it uses.
1. **Set up from the base:** `python submission/derive/run.py setup --task task.json`, with **no** `--parent`. This
   is the one difference from `0009599014-0016`'s chain, which would otherwise start from the start the task offers.
   The layer is fitted on the base's own answers.
2. **The derivation, unchanged:** `nohup python submission/derive/run.py chain --task task.json > chain.log 2>&1 &`.
   It runs `0009599014-0016`'s steps 2 to 8 on the base: the base's answers on fit, diagnosis, replay, dev answers,
   self-distillation beside the unit fit, the merge and its evaluation on dev against the base, and further unit
   seeds if a fit is leaky. It plans its own time against the deadline and keeps the last 40 minutes. Follow
   `$WSP_WORK/wrapup/LOG.txt`. Its last line, `CHAIN DONE: deliver <name> ...`, names the edit.
   If it ends `CHAIN DONE: no edit passed`, take the last-resort edit that line names, as `0009599014-0016`'s step 8
   says.
3. **Deliver on the parent:** `python submission/derive/compose.py --task task.json` writes `<name>_P`: the same ops
   and arrays, stacked on `0009598520-0002` instead of on the base. It builds the model and logs payload and
   relative change. Nothing is re-fitted or scaled.
4. **Report and upload:** `python submission/derive/run.py report --name <name>` drafts `$WSP_WORK/wrapup/REPORT.md`
   with every table the chain measured (on the base). Add in your own words that the layer was derived on the base
   and delivered on `0009598520-0002`, the compose line from LOG.txt, and anything that deviated. Then run
   `python submission/derive/run.py deliver --name <name>_P --task task.json`. It packages by the upload protocol
   (parent `0009598520-0002`), runs `axonry-check` and uploads only if the check passes.
5. **End** with `axonry-done`.

## 4. Time and limits
On four RTX 4090s, `0009599014-0016`'s derivation on this base takes about 2 h 50 min (its measurement: answers on
fit 43 min, then about 4.1 times that for the full plan). The chain shortens its plan by itself if the run is slow.
Compose takes about 5 minutes (one model build) and delivery with `axonry-check` about 6 to 10 minutes, both inside
the 40 minutes the chain keeps. Total: about 3 h 10 min of the 6 h.
The delivered layer is `0009599014-0016`'s kind of edit: about 6,200 `scale_cols` ops, and 12 `set_rows` /
`set_cols` ops for the 96 wrap-up units of the last four layers.
Its payload is about 1.9 MB against the 2B line's limit of 8 MiB (4,096 bytes per hidden unit x 2,048). Its relative
change against the parent is about 0.07 globally (limit 0.2) and at most 0.5 per tensor.
