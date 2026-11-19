# Method: task-dependent routing with two one-neuron gates

**Failure to investigate.** Long thinking can loop instead of reaching an answer. The working hypothesis is that
the useful intervention depends on the task: some mathematical and multiple-choice practice prompts benefit from
no-thinking, arithmetic word problems should retain thinking, and instruction-following prompts should retain
thinking but may benefit from a late exit. Test these hypotheses on practice data before writing either gate.

**Data roles.** Use only the shipped `math_train`, `gsm8k_train`, `mmlupro` and `ifsynth` practice sources.
Create the question groups once, before inspecting model outputs:

| Group | Questions per source | Purpose |
|---|---:|---|
| FIT | 128 | Both gates' fitting, negative traces and gain calibration |
| DEV | 64 | Routing diagnostics, candidate selection and controls |
| HOLDOUT | 128 | One final comparison after all choices are fixed |

This gives 512 FIT, 256 DEV and 512 HOLDOUT items. Group normalized underlying questions together, including an
`ifsynth` prompt and its original arithmetic question. Different traces, samples and prompt constraints of a
question cannot move between groups. The split manifest records identities and fingerprints; both fitting
scripts verify it and reject an old, changed or overlapping split before loading a model. An existing case
cannot be overwritten. Do not reuse runs from the previous split procedure.

**Procedure.** Run commands from the toolkit workspace. The supplied parent edit is the published parent named
in `entry.json`; use that bundled edit throughout, without fetching another starting model.

1. Create a fresh case and its fixed split manifest:

   ```sh
   export WSP_WORK="${WSP_WORK:-$PWD/.work}"
   python submission/derive/prepare_case.py n6 --parent parent/edit/edit.json
   ```

2. Check the mode hypothesis on FIT and confirm it on DEV, using the same sampling and token limit. Generate
   thinking runs for both groups; also compare the template-only `base@think=0` control. Split the thinking
   runs by practice source:

   ```sh
   python -m wsp.case run n6 base fit
   python -m wsp.case run n6 'base@think=0' fit
   python -m wsp.case run n6 base dev
   python -m wsp.case run n6 'base@think=0' dev
   python submission/derive/split_runs.py "$WSP_WORK/runs/n6/base__fit.jsonl" "$WSP_WORK/split"
   python submission/derive/split_runs.py "$WSP_WORK/runs/n6/base__dev.jsonl" "$WSP_WORK/split"
   ```

   If no-thinking does not help the mathematical/multiple-choice practice prompts, or long instruction-following
   traces do not show a problem that a late exit could address, stop and report that the premise failed.

3. Fit the routing unit in layer 4 on FIT; use DEV only for diagnostics:

   ```sh
   python submission/derive/route_gate.py n6 R --layer 4 --keep-think gsm8k_train --n 128 --check-n 64 --margin 0.3 --keep-q 0.5
   ```

   Read: shrinkage LDA at the MLP input, mathematical/multiple-choice last prompt positions versus other prompt
   positions, instruction-following and keep-thinking prompts, and generated positions from `base__fit` only.
   The threshold exceeds non-keep negatives and is at least the median keep-thinking score. Write the mean
   same-question no-thinking/thinking residual difference through a low-use MLP unit's gate/up rows and down
   column. The median threshold can still trigger on some keep-thinking prompts: inspect DEV results rather
   than assuming protection. Require high routing recall per target practice source and low unintended
   instruction-following/keep-thinking closure; otherwise stop or revise using FIT/DEV only.

4. Generate R traces separately on FIT and DEV. Their identities remain in the original fixed groups:

   ```sh
   python -m wsp.case run n6 R fit
   python -m wsp.case run n6 R dev
   python submission/derive/split_runs.py "$WSP_WORK/runs/n6/R__fit.jsonl" "$WSP_WORK/split"
   python submission/derive/split_runs.py "$WSP_WORK/runs/n6/R__dev.jsonl" "$WSP_WORK/split"
   ```

5. Fit the last-layer late-thinking unit on R's FIT traces. Pass DEV traces through separate check arguments:

   ```sh
   python submission/derive/late_gate.py n6 LG --on R \
     --think "$WSP_WORK/split/R__fit__ifsynth.jsonl" \
     --neg "$WSP_WORK/split/R__fit__math.jsonl,$WSP_WORK/split/R__fit__mmlupro.jsonl,$WSP_WORK/split/R__fit__gsm8k.jsonl" \
     --neg-all "$WSP_WORK/split/base__fit__math.jsonl,$WSP_WORK/split/base__fit__gsm8k.jsonl" \
     --check-think "$WSP_WORK/split/R__dev__ifsynth.jsonl" \
     --check-neg "$WSP_WORK/split/R__dev__math.jsonl,$WSP_WORK/split/R__dev__mmlupro.jsonl,$WSP_WORK/split/R__dev__gsm8k.jsonl" \
     --check-neg-all "$WSP_WORK/split/base__dev__math.jsonl,$WSP_WORK/split/base__dev__gsm8k.jsonl" --q 0.9999
   ```

   Read: LDA separating FIT instruction-following thinking beyond 8k tokens from early thinking below 3k, answers
   and negative runs. Only `--think` supplies positives. `--neg` and `--neg-all` remain negative at the final
   prompt position and throughout generation even beyond 8k. These labels are retained in calibration and DEV
   checks; a mathematical trigger is a false positive. Every input row is checked against its designated group
   before any trace limit is applied, so a HOLDOUT row cannot be hidden after a truncation boundary.
   Write the output-embedding direction for closing thinking minus the usual predictions at positive sites,
   adjusted through the final norm. Calibrate the gain on FIT only; stop if no gain meets the closing-probability
   target. Check DEV for useful late triggering and protection of early thinking, answers and other tasks.

6. Compare parent, R and LG on DEV, including actual generated answers, before selecting LG. Record the choices,
   controls and unsuccessful attempts. Check the edit's payload and change limits. All fitting, gain selection,
   layer selection and stopping choices must be finished before the final group is used.

7. Run the parent and the single fixed LG candidate on HOLDOUT once:

   ```sh
   python -m wsp.case run n6 base holdout
   python -m wsp.case run n6 LG holdout
   ```

   Look for the intended improvement without losses on protected practice sources. Do not retune thresholds,
   gains, layers or candidates using these outcomes. If the final comparison fails, report that result rather
   than repeatedly selecting against this HOLDOUT.

**Deliverables.** Submit this method and its Python source. Every replacement vector is derived during the run
from the parent and allowed practice data; ship no precomputed weights or binary arrays. R stores inline numeric
`values` in `R.json`; LG appends its inline operations to R's, preserving the same external parent. Gate diagnostics
are written under the case's `diagnostics/`, not in the toolkit's result-table JSON namespace.
Write `REPORT.md` describing observations, controls and final practice results. Inspect the package with
`python -m wsp.upload n6 LG --report REPORT.md --dry-run` before using the same command without `--dry-run`.
The exported edit is self-contained JSON plus its report, without unresolved NPZ references.
