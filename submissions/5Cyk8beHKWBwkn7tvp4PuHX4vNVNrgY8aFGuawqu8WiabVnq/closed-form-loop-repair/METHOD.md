# closed-form-loop-repair: the 2B's thinking loop removed by a closed-form rank-one edit, measured against the published loop recipe, with a readout edit against re-opening

## The failure
In thinking mode Qwen3.5-2B often never answers. Its reasoning falls into a loop (the same few sentences until the
token limit) or re-opens settled steps until the limit. The answers it finishes are mostly right, so the ability is
there and every unfinished answer is lost accuracy. Measure it first, on practice problems only and per task type:
the share of answers cut at the limit, the share of those whose tail repeats (`cut_loop` in `wsp.grade`), and the
accuracy of finished answers against all answers. If few answers are cut, this method does not apply: say so and stop.

## Credits, and what is ours
- **The published loop recipe** (toolkit `playbook.md`, "The loop edit": a residual direction separating loop
  continuations from normal reasoning, removed from the 64 MLP units that write most of it in layers 15 and 19,
  alpha 6; first entered as #2 "loop-removal", 0009531781-0011; the toolkit's own transfer to this base is
  `cases/qwen35-2b-think_loop`). Here it is derived again in the run from the base's own traces and serves as the
  **floor**: the first upload, and the reference every other candidate has to beat.
- **The closed-form rank-one edit** follows the continuous minimiser of the Tree-Ising objective (arXiv 2611.14210,
  "Continuous edits"): lower what a layer's MLP writes along the loop direction on loop tokens, for the least change of
  its output on protected tokens, in closed form from forward statistics.
- **The readout edit** is the idea of the 0.8B entry "doubt-readout" (0009538688-0028; cf. the "Wait" readout of
  arXiv 2611.20953), made closed-form.
- **Ours:** (1) the closed-form edit under this competition's limits: restricted to the columns that carry the dose
  (solved again inside them), capped in relative change per tensor, written as `set_cols`, with its token sets taken
  from natural loops in the base's own full-length answers over every practice task type instead of constructed
  repetitions; (2) the comparison in one run of the two loop edits and of their composition on the same problems;
  (3) the readout edit as rows of the model's own output head, with no fit; (4) the floor-first rule and the fixed
  replacement rule below.

## Why we expect the closed-form edit to add something
The published recipe removes one direction from 64 units per layer and leaves everything else as it is. The base's
answers that are still cut afterwards are, per the toolkit's notes, mostly long unfinished reasoning. The closed-form
edit differs in three ways that may matter there:
1. its direction is `u = N(N(h_R - h_P) - 0.5 N(h_Z - h_M))`: the loop direction with half of the *concluding minus
   continuing* direction taken out, so what it removes is "repeat" and not "conclude";
2. it weighs every MLP unit by how much it writes on loop tokens relative to how it co-varies with the others on
   protected tokens (`k = (C_P + ridge)^-1 mu_R`), instead of taking the 64 largest writers at full strength;
3. its statistics come from natural loops of every practice task type, not from maths alone (Step 0 measures how
   often each type loops on this base).
Whether this is better than, worse than, or additive to the published recipe on this base is **not known in advance**:
the run measures it, and the floor is kept unless a candidate beats it by the fixed rule. The readout edit is the
least certain part (suppressing re-checks can cost hard maths problems); it is tested last and kept only by the same
rule.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/` and
`task.json` are there. The start is the base (`parent` null). Keep `PROGRESS.md` current. Every `run` uses all free
GPUs; give independent jobs their own `CUDA_VISIBLE_DEVICES`. Long commands: run them detached with a log and wait.

**Step 0. The case and the base's answers (about 20 min).**
```
python -m wsp.case init lp --profile qwen35-2b-think --contrast loop \
    --practice math_train gsm8k_train mmlupro ifsynth tuluif bbhsynth --dev 288 --holdout 288
python -m wsp.case run lp base dev
```
Read the result per task type (the measurement above).

**Step 1. The floor: the published recipe, derived here (about 10 min).**
```
python -m wsp.case pairs lp --sets dev
python -m wsp.case extract lp          # layers 15 and 19 should be among the largest stable writers below the last
python -m wsp.case edit lp --layers 15 19 --topk 64 --alpha 6 --name A
```

**Step 2. The closed-form edit, the readout edit and the compositions (about 10 min, one GPU; start the run of A of
Step 3 on the other GPUs meanwhile).**
```
python submission/derive/treeising.py --case lp --profile qwen35-2b-think \
    --answers $WSP_WORK/runs/lp/base__dev.jsonl --name L --scales 1.0,0.6        # -> L_100, L_060
python submission/derive/join.py lp LA L_060 A        # the closed-form edit at 0.6, then the published recipe on top
python submission/derive/readout.py --case lp --profile qwen35-2b-think --name AR --alphas 0.35 --on A
```
`treeising.py` prints, per layer, the layer's own write on loop tokens, the dose (in multiples of it), the relative
change and the payload (800 columns per layer as float16: about 6.6 MB, inside this base's payload limit). It caps
the relative change at 0.45 per tensor. If it finds fewer than about 20 looping
traces, add the base's answers to the holdout split (`python -m wsp.case run lp base holdout`, then pass both run
files to `--answers`). In `LA` both edits touch the same two tensors: build it and read the relative change per
tensor in the build's manifest; if a tensor is above 0.5, make `LA` from a weaker closed-form edit (`--scales 0.4`).

**Step 3. Measure on `dev`, the profile's own token limit, one answer per problem, paired on the same problems.**
```
python -m wsp.case run lp A dev
python -m wsp.case run lp L_100 dev
python -m wsp.case run lp LA dev
python -m wsp.case run lp AR dev
python -m wsp.case compare lp A L_100 dev ; python -m wsp.case compare lp A LA dev ; python -m wsp.case compare lp A AR dev
```
- **Upload the floor at once**: as soon as A's run shows a pooled gain over the base of at least twice its SE with no
  task type lower than the base by more than twice that type's SE, upload A (commands below). If A does not pass
  but `L_100` does, `L_100` is the floor.
- **Replacement rule (fixed before any reading).** A candidate replaces the uploaded edit when, compared with it on
  `dev` (`compare`), its pooled gain is at least +2.0 points and at least one SE, and no task type is lower by more
  than twice that type's SE. Candidates are read in this order: `L_100`, `LA`, `AR`. When a candidate other than
  `AR` replaces A and `AR` also beat A by the rule, add the readout to the new edit
  (`readout.py --name XR --alphas 0.35 --on <edit>`), run it on `dev`, and apply the same rule against the new edit.
- A norm-matched control for the report: `python -m wsp.case edit lp --layers 15 19 --topk 64 --alpha 6
  --random-control` and its `dev` run, if time allows.

**Step 4. Confirm once on `holdout`** (`freeze` the chosen edit first): `run lp base holdout`, `run lp <chosen>
holdout`, and, when the chosen edit is not A, `run lp A holdout` and `compare lp A <chosen> holdout`. If the chosen
edit is below A on `holdout` by more than twice the SE, upload A again as the final edit and say so in the report.

**Uploads.** For edit E (a later upload replaces the one before):
```
python -m wsp.upload lp E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/lp__E/edit --task task.json
python -m wsp.upload lp E --report REPORT.md --task task.json
```
REPORT.md: the measurement of Step 0, every candidate's `dev` and `holdout` result per task type with its paired
difference and SE, the printed dose and relative change of each edit, which rule decided, what did not work, and the
control. Rewrite it after every measurement, so that the upload never lags behind what is known.

**Time (4 GPUs).** Step 0 by 0:25, Steps 1-2 by 0:45 (A's run starts as soon as A exists), the floor uploaded by about
1:05, the four `dev` runs by about 1:45, Step 4 by about 2:45, final upload by 3:00. Start no new generation after
4:30; the final upload is complete by 5:00. When the final edit is uploaded and verified, end the run (`axonry-done`).

## What to expect
- After A: most verbatim loops gone and a large gain on every task type where loops were common; correct answers
  get longer, because harder problems now finish.
- The closed-form edit: a similar gain alone; composed with A, fewer of the remaining cut answers if its direction
  covers what A leaves. A null or a loss here is a result: report it, and the floor stays.
- The readout edit: fewer re-opened steps and fewer answers cut by long reasoning; the risk is a loss on hard maths
  problems and on instruction following. It is kept only by the rule.
