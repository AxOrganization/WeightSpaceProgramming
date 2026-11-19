# Loop exit: brake the re-check where the loop lives

## The problem
The starting model is the parent named in `entry.json`: the base with a loop-removal edit already applied. It
finishes far more answers than the base, but a share of its practice answers still run to the token limit, and on
hard problems that share is large. Few of those are verbatim loops: most are long re-checking tails, a "Wait"
every few lines, each re-derivation worded a little differently, never converging. Many of the cut answers belong
to problems that another sample of the same model finishes correctly. That is the prize: the model can solve
them, it just never stops re-checking.

Measure this first on the starting model, on practice data of all four task types (GSM8K / MATH train, MMLU-Pro
practice, IFEval-style `ifsynth`), with several samples per problem: the share cut at the token limit, how many of
those are verbatim loops (`cut_loop`) versus non-repetitive re-checking (`cut_other`), the number of "Wait" per cut
trace, the accuracy of finished answers, and the share of cut samples with a finished sibling. Read ten cut traces.
If fewer than one practice answer in twenty is cut at the token limit, this method has nothing to remove: report
the measurements and stop. (Do not stop because detected verbatim loops are rare: the re-checking tails are the
target, and the brake acts on them whether or not they repeat verbatim.)

## Why we think it happens
- A cut answer is a decision repeated at step boundaries. At the end of a line the model either moves on, re-checks
  ("Wait"), or closes its thinking. In a re-checking tail it puts nearly all probability on re-checking.
- The model's own weights contain a veto against that. Among the MLP units of the last layers, one unit is active
  almost only right before a "Wait" inside a re-checking tail, and its output column lowers the "Wait" logit through
  the final norm and the tied unembedding; on ordinary text it is nearly silent. It is too weak to win.
- Scaling that one output column (`scale_cols`) is a conditional intervention written into the weights: it acts only
  where the unit fires, i.e. inside the tail, at the moment of the decision, and leaves ordinary reasoning alone.
  Fewer re-checks, so derivations run to their end and finish with an answer.
- The veto is a property of the late layers, not of the loop-removal edit upstream: in our measurements the same kind
  of unit stood out by a factor of 40-70 over the runner-up on several different parents. The scan below finds it on
  whatever parent the run starts from.
- What should falsify this: no late-layer unit stands out as a selective "Wait" veto (then the brake is weak and the
  dose must stay low), or the brake turns correct answers wrong as often as it rescues cut ones (losses close to
  rescues on the paired DEV run). Both are measured below.

## Approach
Helpers in `derive/` (copy them to the sandbox; they import the toolkit: run them with `WSP_REPO=<competition
toolkit path from BRIEF.md>` or the toolkit on `PYTHONPATH`): `branch.py` (splits, loop diagnosis), `gate_exit.py`
(activation statistics and the brake scan), `screen.py` (the fixed-unit control, tables). Everything else is
`python -m wsp.case`. One job per GPU (`CUDA_VISIBLE_DEVICES`), detached (`tmux`). Build the starting model once
before parallel jobs use it (`python -c "from wsp.case import model_path; model_path('brake', 'base')"`). A DEV run
(128 problems x 4) takes about 40 minutes on one GPU; HOLDOUT halves (128 x 2) about 20 minutes each.

1. **Case, splits, starting runs** (parent as in BRIEF.md):
   ```
   python -m wsp.case init brake --profile qwen35-0.8b-think --parent parent/edit/edit.json \
       --practice gsm8k_train math_train mmlupro ifsynth --subset dev128=dev:128 --subset holdout256=holdout:256
   python derive/branch.py split brake --of dev --minus dev128 --name mine --parts 3
   python derive/branch.py split brake --of holdout256 --minus dev128 --name ho --parts 2
   GPU0: python -m wsp.case run brake base dev128 --n 4       # the paired reference for every edit
   GPU1-3: python -m wsp.case run brake base mine_<j> --n 4   # traces for the statistics (never evaluated on)
   ```
   The statistics come only from `mine_*`; edits are chosen on `dev128` and confirmed on `ho_0`/`ho_1`.

2. **Diagnosis** (as soon as the mining runs finish): `python derive/branch.py onsets brake --sets mine_0 mine_1
   mine_2` writes `results/onsets_diagnosis.json` (verbatim loops, where they start, siblings that finish); the DEV
   run's `kinds` give `cut_loop` vs `cut_other`. Count "Wait" in the cut traces' tails. Record it all in NOTES.md and
   apply the stop rule above (cut share, not loop share).

3. **Activation statistics** - `python derive/gate_exit.py collect brake --sets mine_0 mine_1 mine_2 --layer 23
   --act-layers 20 21 22 23` (one GPU, minutes): for every MLP unit of the last four layers, its mean activation at
   the positions right before a "Wait" inside the cut traces' tails (a verbatim loop's repetitive stretch, or the
   second half of a cut trace that does not repeat verbatim), inside those tails, and on ordinary text. The report
   counts the pre-Wait positions; with fewer than 300, mine more traces (a fourth `mine` part from the rest of `dev`)
   before scanning.

4. **The brake scan, the dose series and the control** - `python derive/gate_exit.py brake brake --layers 20 21 22 23
   --factor 3 --name BRAKE_x3`, then the same with `--factor 2` (`BRAKE_x2`) and `--factor 4` (`BRAKE_x4`): among the
   units of the last four layers whose contribution to the "Wait" logit on ordinary text is at most 0.3 of their
   contribution right before a "Wait" in the tails, the one whose contribution there (activation times its output
   column's readout through the final norm) is the most negative; its output column is scaled. The report lists each
   layer's best unit and its ratio to the runner-up: expect one unit to stand out by an order of magnitude or more;
   if none does, keep the dose at 2. Brake one unit only: a second brake from another layer cut more answers short in
   rehearsal, and they finished wrong. The control: `python derive/screen.py heads brake --name CTRL_x3
   <layer>:mlp:<j>:3`, with `<j>` a unit of the same layer drawn at random (not the braked unit). Check every edit's
   relative change per tensor in its built model's manifest (the rules' limit is 0.5; keep every edit under 0.45).

5. **DEV round 1** (four GPUs, ~40 min): `BRAKE_x2`, `BRAKE_x3`, `BRAKE_x4` and `CTRL_x3`, each `python -m wsp.case
   run brake <edit> dev128 --n 4`. A dose is accepted if its paired gain is at least one standard error and **no
   practice source** (gsm8k, math, mmlupro, ifsynth) drops by more than its own standard error. Read the gain
   through `rescues` (samples cut or wrong before, correct after) and `losses` (correct before, wrong or cut after):
   with no real effect the two are equal up to sampling noise (at temperature 0.6 some 30-40 of 512 samples flip
   each way between any two runs), so a real edit must have rescues exceed losses by at least the square root of
   their sum. Report the accuracy of finished answers but do not gate on it: it falls whenever many cut answers start
   finishing, because the rescued problems are harder than the ones that already finished. The control should be
   flat (within one standard error); report it either way, it does not change the choice of dose. Expect the doses
   to sit on a plateau within noise (a DEV gain has a standard error of about 2 points); prefer the middle dose, and
   the lower one if losses grow with dose. If no dose passes, keep the dose with the largest gain minus its standard
   error among those with a positive gain and no source down by more than two standard errors, and say so.

6. **DEV round 2** (three GPUs, ~40 min): the chosen dose's neighbours half a step away (e.g. `--factor 2.5` and
   `3.5` around a chosen 3) on `dev128`, and the chosen dose on `ho_0` with the starting model's `ho_0` run beside it
   (`--n 2`). Choose among the doses by DEV, using `ho_0` as a check; ties (within one standard error) go to the
   smaller factor. `freeze` it, `axonry-check`, and **upload it now** (the last complete upload counts).

7. **HOLDOUT** (~25 min): the starting model and the chosen dose on `ho_1` (`--n 2`), with `ho_0` from round 2.
   Judge on DEV and HOLDOUT together (1024 samples), per practice source: the chosen dose stays unless its combined
   gain is negative or a practice source is down by more than two standard errors on the combined samples; then the
   next dose down replaces it if it does better on the same test. One HOLDOUT half moves by several points per source
   from sampling alone (128 problems x 2), so never drop a dose for one half. Upload the confirmed edit with the report
   (every table, including what was dropped and why).

**Time plan (6 h, 4 GPUs).** 0:00-0:10 setup; 0:10-0:50 starting runs (mining traces finish by ~0:30);
0:30-0:55 diagnosis, activation statistics, the scan, the dose series and the control; 0:55-1:40 DEV round 1;
1:45-2:30 DEV round 2; 2:35 first upload; 2:40-3:10 HOLDOUT; 3:15 upload the confirmed dose; 3:15-5:00 a HOLDOUT
run of the runner-up dose if it was within one standard error; 5:15 final upload; the rest is buffer. The 30- and
10-minute reminders stop all work: check and upload the best confirmed edit.

## What to expect
A drop in cut answers wherever the parent still re-checks to the limit, largest on the hardest math; finished
answers nearly as accurate as before (the rescued ones are harder than the easy ones); ordinary reasoning and
instruction following untouched, because the unit is silent outside the tails. The remaining cut answers are
verbatim derivation cycles that never pass through a re-check word; nothing here rescues them, and forcing them to
exit yields wrong answers. If no dose passes the DEV guards, keep the dose with the largest combined gain, say so,
and upload it: the brake acts only inside re-checking tails, and the practice sets are small enough that a real gain
can sit inside their noise.

## Rules this method keeps
Practice sets and the model's own generations only; no benchmark items, answers or near-duplicates; model-card
sampling for every judged generation; no outside weights or data; the readout vectors are read from the model's own
frozen embedding and final norm, never imported. Which layer, unit and dose to use all come out of the run's
measurements; this method fixes none of them in advance.
