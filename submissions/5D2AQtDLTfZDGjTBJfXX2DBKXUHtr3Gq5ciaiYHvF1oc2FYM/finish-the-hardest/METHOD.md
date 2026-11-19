# Where self-taught finishing stops: a context-gated hesitation penalty for the hardest calculations

Close step: ON
Re-check close: ON

(The two lines above are this method's only switches. Close step OFF: skip step 10 entirely. Re-check close OFF: skip
step 11 entirely. ON: run the step when the time gates of the time plan allow it; each step stays in the upload only
if its own rule holds.)

## The idea
This entry stacks on the published edit named in `entry.json` (`parent`): a finishing edit fitted on the model's own
answers. It raised the likelihood of the model's finished, correct answers and lowered that of looping tails, with
small direction changes spread over every layer. That teaches finishing wherever the model already produces a
finished correct answer to imitate. Most answers to easy and medium calculation, multiple-choice and instruction
prompts now close, and answers are shorter.

It cannot teach finishing where the model never finishes correctly. On the hardest calculation problems there is no
finished correct answer to imitate, and the habit that keeps those answers open is untouched. After a slip or an
unexpected intermediate number, the model re-opens a result it has already checked, sentence after sentence. Each
re-check starts with the same kind of word, a hesitation interjection set off by a comma at the start of a sentence.
The parent's never-closing answers are therefore now concentrated on the hardest problems, and they still spiral
through that opener. In our exploration on the parent (practice dev problems, 4 samples each), no gsm8k-type or
instruction answer and 2% of the multiple-choice answers ran to the token limit, but a fifth of the math answers did,
none of them in a verbatim loop, while the math answers that closed were right five times out of six.

This method adds a lever that does not depend on whether a problem was ever solved: it acts on the decision itself.
- A penalty on the opener, written by new units of the last layer, makes re-opening a checked result unlikely at the
  positions where the sampler would draw it. It does not steer the model's internal state: it writes against one
  token's logits at the output, so it changes only which word opens the next sentence, and only where that word was
  a live option.
- The opener has a second use. In instruction and writing tasks, it is how the model checks a draft against a
  constraint. Suppressed there, the model keeps redrafting and fewer instruction answers close. So the penalty is
  gated by context: the full penalty where the model reasons through a calculation, none where it follows
  instructions.
- The two layers act on different things. The parent shifts whole answers toward the ones it has seen finish. This
  layer changes one token-level decision, only in calculation context. Where the parent already finishes, the opener
  is rarely drawn and the layer has little to do. Where the parent leaves a spiral, it acts there.

In our own exploration on practice data, with a finishing edit of the same kind (gains fitted on the model's own
correct answers), half of the answers to the hardest math practice problems still never closed. Units built by this
procedure lowered that share to about a third and raised exact accuracy on those problems (several hundred problems,
one draw each).

Deep inside a long calculation trace, the moment the model is about to re-open its result is itself a sign that the
answer is ready: by then it has been derived and checked, and the re-check will only derive it again. A re-check close
step turns that moment into a close. One more unit of the last layer fires where three things hold at once: the
sampler is about to draw the hesitation opener (a decision position of the penalty), the context is calculation, and
the trace is deep. There it moves the opener's mass to the thinking close tag. Everywhere else, early or late, it is
silent: it never raises the close tag where the model was not about to start a re-check.

On the parent's own dev traces (the half of the problems not used in the fit), the emulated extra close of such a unit
was 1.0 on the math traces that never closed, 0.02 on the math traces that closed, at most 0.05 on the other
calculation traces that closed, and 0 on instruction traces.

Not every opener that is frequent in never-closing traces is a hesitation. Those traces are long and repetitive, so
first-person planning ("I will write ...") and calculation openers ("So", "Then", "Let") are over-represented there
too. They also open ordinary reasoning: lowering them changes the model everywhere without closing the spirals. The
penalty targets a hesitation interjection only, and a decoding test checks that lowering it closes spirals (step 2).

## The behavior to measure first
Measure on the parent, per practice source (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`), on the hardest level
of the math practice source (`dev_hard`, written by `derive/sets.py`), and on the self-written instruction prompts in
`data/aux_if.jsonl` (practice data and self-written prompts only):
- accuracy;
- the share of answers that never close (`cut_share`: cut at the token limit, almost always inside the thinking;
  `compare.py`'s `uncl` rows count the ones whose thinking never closed);
- the accuracy of the answers that close;
- which hesitation interjections are over-represented in never-closing traces, compared with finished correct traces
  (`derive/markers.py`).

The lever needs spirals to act on. If fewer than 10% of the answers never close on both `dev_math` and `dev_hard`, or
no hesitation interjection stands out (step 2), the method has nothing to remove on this model. Then say so in
REPORT.md, and on uploads follow the brief: if one is required, upload the best edit that has passed the checks.
Decide on the toolkit's strict grade: the boxed answer equals the gold after `wsp.grade._norm` (exact match, no
symbolic equivalence). This is the `exact` grade of `derive/compare.py`; report the lenient one (`mv`) beside it.

## Why it happens, and why the edit can be computed exactly
- Each spiral is a chain of decisions to re-open a checked result, and each re-opening starts with a hesitation
  opener. Once the opener is written, the next sentences re-derive the same thing. Making the opener unlikely lets the
  model commit to the result it has and close its thinking.
- The finishing edit learned from answers that finish. The decision to re-open is made deep inside answers that it has
  no finished example of, so that decision keeps the parent's old weight there.
- The eval sampler (temperature, top-k, top-p, presence penalty of the profile) only draws the opener where the model
  gives it real probability. On the parent's own traces almost all of the opener's sampling mass sits at a small set
  of DECISION POSITIONS: thinking positions where the sampler would draw it with probability at least tau. A gate only
  has to be right at those positions.
- A unit of the last decoder layer's MLP (a design choice of this method, made for the reason that follows; which
  units, directions, thresholds and writes is all found by the run) writes into the final residual, which nothing
  reads except the final norm and the tied output head. It changes the next-token distribution of its own position
  only. So its effect on the parent's traces can be computed exactly from one stored teacher-forced pass, before any
  generation, with the eval sampler emulated (`derive/dplib.py`). A built-model check confirms it.
- At decision positions, the last MLP's input carries which kind of task the model is working on. A linear read (LDA)
  separates calculation reasoning from instruction reasoning there. Two units on that read make a saturating clamp:
  no penalty below a low threshold, the full penalty above a high one. One SiLU unit would not do: its write grows
  with the score, so it is weak at low-scoring calculation positions (the opener leaks through, and a weak penalty
  only delays a spiral) and too strong at high-scoring ones.
- The gated write is exactly the always-on unit's write. The two edits differ only in WHERE the penalty applies, so
  comparing them isolates the gate.
- The parent changes every layer, the last one included, by small direction ablations. The new units are written as
  whole rows and columns of the least-used units of the parent's last layer, on top of the parent's weights: they
  replace what those units held and leave every other unit as the parent has it.
- The positions where a re-check starts are known from the parent's own traces: the positions whose next token is the
  opener. Deep in long calculation traces, the last MLP's input at those positions is linearly separable from every
  other position (shallow ones, positions followed by any other token, other tasks, answers): an LDA on that input,
  with its threshold at a high quantile of the other positions' scores, gives a unit that fires at late re-check
  starts. Its write is one direction through the parent's own final norm and tied head that raises the close tag and
  lowers the opener together, a swap of the two logits, so the mass the opener loses goes to closing instead of to
  another sentence.

## Approach
Work in the toolkit's case workflow (`python -m wsp.case`), in the sandbox, from `/workspace/work`. Start the case
from the parent (`init ... --parent parent/edit/edit.json`), so "base" below is the parent model.
Use only practice data, the entry's self-written prompts and the model's own generations.
Every number in the edit (opener, unit, read direction, thresholds, write) comes out of this run.
Choices are made on the dev sets (`dev_<src>`, `dev_hard`) and the self-written prompts. The holdout sets only
confirm the edit that the dev rules chose.

1. **Splits and base runs.** Init the case over the four practice sources: dev 64 and holdout 128 problems per source.
   Write the per-source set files and the two hardest-level math sets (`derive/sets.py`: `dev_hard` and
   `holdout_hard`, 64 problems each, in no split). Build the parent model once (`pre base`) before any parallel job.
   Base runs at the profile's own token limit (never lower it: the never-closing share only shows at the real limit):
   - every dev set and `dev_hard`, 4 samples per problem;
   - the self-written prompts' `fit` split, 1 sample (gate negatives);
   - their `test` split, 2 samples (the self-written instruction check, never in any fit), with
     `derive/run_items.py`.
2. **Diagnosis, opener candidates, decoding test.** Read each base run's summary (`results/base__<set>.json`: `acc`,
   `cut_share`, `acc_when_finished`, `kinds`). The PENALIZED sets are the calculation sets: `dev_math`, `dev_hard`,
   `dev_gsm8k`, `dev_mmlupro`. On this parent most of their never-closing answers are on `dev_hard` and `dev_math`; the
   others stay in because they are the same kind of reasoning, gain less, and rule D guards them. The INSTRUCTION
   sets are `dev_ifsynth` and the self-written prompts.
   - Candidates: `derive/markers.py` on the penalized base runs. Its `candidates` are the hesitation interjections:
     sentence openers set off by a comma in at least half of their uses in never-closing traces, that open sentences
     at least 1.3 times as often there as in finished correct traces, ranked by the excess per 1,000 sentences (at
     most three). Read a few never-closing traces at each: it should open re-checks of a result already found. No
     candidate: once more with `--min-ratio 1.2`; still none: the lever does not apply (as above).
   - Decoding test (for the choice only, never shipped): each candidate on `dev_hard` with its logits lowered by 3 at
     decoding time (`run <case> base@penalize=<candidate>,bias=-3 dev_hard --n 2`, the base run's seeds), one GPU
     each, compared with the pooled base (`compare.py`). Rule M: among the candidates whose test lowers the
     never-closing share (`uncl` d < 0) and keeps exact accuracy within its noise band (d >= -max(2 se, 1 point)),
     W1 is the one with the largest exact gain, except that a candidate ranked higher by markers.py is preferred
     when its gain is within 2 points of that largest gain. None passes: the lever does not apply (as above).
   - With a single candidate, start step 3 with it at once and confirm rule M before step 6. The test also bounds
     what the weight edit can do: the gated edit should come near its gain.
3. **Always-on units, a dose series** (`derive/hesitation_unit.py`). The write direction lowers W1's logits: the mean
   gradient through the parent's own final norm and tied head, scaled by bisection so the mean logit change is -B. It
   goes into the least-used unit of the last layer as an always-on unit. Build H_B for B = 2, 3, 4. Report the
   realized change, the most affected other tokens and the KL on the parent's text.
4. **The gate** (`derive/hesitation_gate.py`). One teacher-forced pass of the parent over base traces of every
   penalized and instruction run (cached, reused by every build). Its parts:
   - decision positions: p(W1) >= tau under the eval sampler (tau 0.01, the tool's default);
   - positives: the penalized runs' decision positions;
   - hard negatives: the instruction runs' decision positions; easy negatives: their prompt and answer positions;
   - each run's problems alternate between a fit half and a test half; the self-written `test` prompts are measured
     only;
   - read direction: LDA on the last MLP's input; the clamp starts at a low and saturates at a high threshold on that
     score; the gate rows may change gate_proj by at most 0.25;
   - write: H_B's own write (`--ref-spec H_B`).

   A VARIANT sets the two thresholds from the fit hard negatives' scores: `LO,HI` takes their LO and HI quantiles
   pooled over the instruction runs; `LO,HI,per-run` takes the largest of each instruction run's own quantiles (the
   hardest instruction source sets the bar, so adding instruction traces can only tighten the gate). The GATE ORDER
   is (0.98, 0.995), (0.99, 0.999), per-run (0.98, 0.995), per-run (0.99, 0.999), per-run (0.995, 0.999), and last
   the looser (0.95, 0.99). Run the pass once with `--dry` over the whole order: the report
   (`results/HXdry__dry.json`) evaluates checks G1-G4 for every variant and names the first one that passes them
   (`first_passing`). Build HX_B for each dose with that variant (`--choose`). None passes: see G3 in the checks.
5. **Built-model check and limits.** `derive/unit_check.py` builds an edit with the toolkit and teacher-forces base
   traces through the parent and the edit: the opener mass at decision positions before and after, and the top-1
   change and KL elsewhere. Check the always-on H_B first, then HX_B against it (`--twin H_B`): checks B1-B2. Then
   `derive/merge.py --dry-run --allow-parent-overlap` on the edit alone checks the subnet's limits exactly as the
   manifest computes them: per-tensor change vs the parent, global change, payload. (`--allow-parent-overlap`
   accepts that the parent's direction ablations touch every unit of the last layer; collisions between this run's
   own components are still refused.)
6. **Dose by generation.** Run HX_2, HX_3 and HX_4 on the four penalized dev sets (4 samples, the same seeds as the
   base run). Also run a second base draw (another salt) on `dev_hard`, `dev_math` and the self-written test prompts
   (`derive/aa_run.py`, `run_items.py --salt s1`): it gives the A/A noise of the parent and a pooled base.
   Compare with `derive/compare.py --pool calc=<penalized sources>`. Rule D, among the doses:
   - the `ALL math` exact row (dev_math and dev_hard pooled) is clearly positive: d - 2 se > 0;
   - the `POOL calc` exact row is positive (d > 0), and every penalized source stays within its noise band
     (d >= -max(2 se, 1 point));
   - pick the dose with the largest `ALL math` exact gain among those; a tie within 1 point goes to the smaller dose.
7. **Instruction check** (in parallel with step 6). Run the strongest dose HX_4 (the worst case for any leak of the
   gate) on dev_ifsynth and on the self-written test prompts. If a GPU is idle, also run the always-on H_4 on the
   test prompts: it shows the hazard the gate removes. Rule I, on `POOL if` and on the test prompts alone:
   - the never-closed rows (`uncl`) must not rise: d <= max(2 se, 1 point);
   - accuracy stays within its noise band: d >= -max(2 se, 1 point).
8. **First upload.** Package the edit chosen by D that passes I and the limits. Self-check it with `axonry-check` and
   upload it, with a REPORT.md (what was found, how the spec was derived, controls, what failed).
9. **Holdout confirmation** of the chosen edit (the final one: X, XC if step 10 replaces it, or XL if step 11
   does). Base and the edit on
   every holdout set and `holdout_hard` (2 samples per problem; the base holdout runs fill idle GPU slots earlier).
   Rule H:
   - the `ALL math` exact gain (holdout_math and holdout_hard pooled) is positive (clearly positive: d - 2 se > 0,
     otherwise say so);
   - no source loses beyond its noise band;
   - holdout_ifsynth `uncl` does not rise (as in rule I).

   The confirmation is reported as it comes out. It is not a contest between edits: the choice was made on the dev
   sets.
10. **Close step (optional; the switch at the top).** `derive/close_unit.py` builds a gated CLOSE unit:
    - it raises the thinking close tag where the parent itself already gives it real probability (the model is ready
      to answer), by the tool's default raise of 2 logits at a default tau of 0.02;
    - it is fitted with the decision positions of the calculation runs (dev_math, dev_hard, dev_gsm8k) as positives,
      those of dev_mmlupro and dev_ifsynth as hard negatives (the self-written prompts stay out of this fit; auxtest
      is the instruction check), and every other stored position (prompts, answers, all other thinking) as easy
      negatives, so it stays silent at ordinary positions (it may also fire at other tasks' decision positions, where
      the model was about to close anyway);
    - its read threshold is the default 0.999 quantile of the fitted negatives' scores;
    - it never reuses the gate's units or the re-check close step's unit (`--avoid-specs`).

    Where the penalty removed a spiral, thinking often ends in an "answer is ready" planning tail; this unit lets
    that tail close a little earlier. The unit, its read direction, threshold and write are computed by
    `close_unit.py` from this run's traces with these defaults, and the unit stays only if rule C holds. Merge it with
    the chosen HX_B (`derive/merge.py --allow-parent-overlap`, which refuses unit collisions and checks the limits)
    and check the merged edit with the close tag as the word (`unit_check.py`, 8 whole traces per run; read
    `after_vs_before` in `results/<XC>__check.json`): its mass at its decision positions rises on the calculation
    runs (dev_math, dev_hard, dev_gsm8k), rises by at most 30% on the instruction runs (dev_ifsynth, auxtest; the unit
    may fire at instruction decision positions, where the model is about to close anyway; +15-20% is normal), and B2
    holds. Run the merged edit on the four penalized dev sets and the instruction sets, with the same seeds as HX_B.
    Rule C:
    - against HX_B, with the four penalized dev sets in one `compare.py --ref HX_B` call: the `POOL calc` exact row
      d >= +1 point with d - se > 0;
    - in the same call: no `ALL math` / `ALL gsm8k` / `ALL mmlupro` exact row with d < -2 se, and the `POOL calc`
      `uncl` row (thinking never closes) d <= max(2 se, 1 point);
    - rule I for the merged edit itself, against the base, on the instruction sets.

    Only then does the merged edit replace the upload (and become the edit that step 9 confirms). Otherwise drop the
    close step and report it.
11. **Re-check close (optional; the second switch).** A unit that turns late re-check starts into closes:
    - Pass (`derive/sp_collect.py --opener-words W1`): one teacher-forced pass of the parent over all of its own
      dev_math base traces, 160 each of dev_hard, dev_gsm8k and dev_mmlupro, and 96 of dev_ifsynth (about 830 traces).
      It stores the last MLP's input at every position whose next token is W1, at every 32nd thinking position and
      every 16th answer position, and the eval sampler's view of the close tag at every thinking position. It needs
      only the base runs; it runs while the first upload is prepared, when the dose series has freed the GPUs (time
      plan).
    - Unit (`derive/dx_build.py`, the tool's defaults): positives = positions followed by W1 at depth >= `--late` in
      the calculation runs (dev_math, dev_hard, dev_gsm8k, dev_mmlupro); negatives = every other stored position; an
      LDA on the last MLP's input, fitted on half of the problems; threshold = a high quantile (`--theta-q`) of the
      negatives' scores; write = one direction through the parent's own final norm and tied head that raises the close
      tag by `--B` and lowers W1 by `--B-open` (bisection on their margin), at firing positions of long fit traces;
      down column = write / mean activation. The unit is the least-used unit of the last layer that neither the parent
      chain nor any gated penalty (`--avoid-specs 'HX*'`) sets; the close unit of step 10 then avoids it in turn.
    - Check E, on the build report (`results/LT__build.json`, `emu`, test half): the emulated extra close
      (`extra_close`, the eval sampler with the unit's raise, on whole stored traces) is >= 0.5 on the never-closing
      dev_math and dev_hard traces, <= 0.2 on the traces of every calculation set that closed, <= 0.05 on dev_ifsynth.
    - Merge it onto the close step's edit (`merge.py --name XL X CL LT --allow-parent-overlap`; with the close step
      OFF: `X LT`), which refuses unit collisions and checks the limits. Check U, the built edit on whole traces
      (`unit_check.py --words '</think>' --tau 0.02 --tokens 0 --traces 8`): the close tag's mass at its decision
      positions rises on dev_math, dev_hard and dev_gsm8k, rises by at most 30% on dev_ifsynth and the self-written
      test prompts, and B2 holds.
    - Runs of XL with the same seeds as X: the four penalized dev sets and dev_ifsynth (4 samples), the self-written
      test prompts (2 samples).

    Rule K, on the dev sets only:
    - against XC (with the close step OFF or not chosen by rule C: against X), the four penalized dev sets in one
      `compare.py --ref` call: the `ALL math` exact row (dev_math and dev_hard pooled) d >= +1 point with d - se > 0,
      and the `POOL calc` exact row d > 0;
    - in the same call: no `ALL math` / `ALL gsm8k` / `ALL mmlupro` exact row with d < -2 se, and the `POOL calc`
      `uncl` row d <= max(2 se, 1 point);
    - rule I for XL itself, against the base, on the instruction sets.

    Only then does XL replace the upload (and become the edit that step 9 confirms); the close unit then stays as part
    of it. Otherwise drop the re-check close step and report it; rule C's choice stands.

12. **Final upload and report.** The chosen edit, uploaded well before the deadline. Report every measurement,
    including what failed and the holdout confirmation. Once the final upload is verified and nothing is left worth
    trying, end the run with `axonry-done`, as the brief says.

## Checks, and what to do when one fails
"Not used" below means: report it in REPORT.md; on uploads follow the brief (if one is required, the best edit that
has passed the checks so far).

| Check | Pass | If it fails |
|---|---|---|
| M opener (step 2) | rule M | the next candidate; none left: the lever does not apply here, not used |
| G1 gate report `gate_fit.test_auc_dp` | >= 0.98 (held-out calculation vs instruction decision positions) | more traces (`--traces 64`, new cache file); still below: the context is not separable at this layer, not used |
| G2 `dp_mass_share` (every run) | >= 0.95 | lower `--tau` to 0.005 (new cache file) |
| G3 penalty fraction `test_dp.mean_frac` | >= 0.9 on every penalized run; <= 0.05 on every instruction run (incl. the self-written test prompts) | the dry report walks the gate order and names the first variant that passes (`first_passing`). None passes: take the variant with the lowest instruction leak among those at >= 0.9 on every penalized run; if that leak is at most 0.10, build it and let rule I decide (a miss that small, on a few dozen test problems, is within the noise of the measurement; rule I overrules it); larger: the next opener candidate that passed rule M (a new pass); none left: not used |
| G4 openers per 1k thinking tokens at test DPs | gated within 1.5x of always-on on penalized runs; >= 0.9x parent on instruction runs | as G3 |
| B1 `unit_check` openers at DPs | penalized runs: HX_B within 1.2x of its always-on twin (`vs_twin` <= 1.2), or both below 1% of the parent's opener mass (a ratio of two near-zero residues is noise); instruction runs: >= 0.85x before. The after / before ratio itself is reported only: dose strength is rule D's matter | penalized runs: the gate passes too little of the penalty: the next looser variant of the gate order; instruction runs: the next stricter one (both rebuilt from the cached pass under new names) |
| B2 `unit_check` away from DPs | top-1 changed <= 2% and mean KL <= 0.01 nats on instruction runs | the next stricter variant, as B1 |
| L limits (`merge.py --dry-run --allow-parent-overlap`) | per tensor <= 0.45, global <= 0.18, payload <= 4 MiB | gate: `--max-rel 0.2`; merged: drop the close step |
| D dose | some dose passes rule D | run the always-on H_4 on dev_hard: no gain either: the lever does not apply here, not used; H_4 gains but no HX_B does: the gate is too strict on calculation (G3), take the next looser variant, rebuild, rerun the best dose |
| I instruction loops | rule I | the next stricter variant of the gate order (rebuilt from the cached pass under new names); rerun step 7 with it, and step 6 for the dose rule D chose; still failing: a smaller dose; still failing at B = 2: not used |
| H holdout | rule H | report the doubt; the upload stays the dev rules' choice |
| E unit build report (step 11) | emulated extra close >= 0.5 on never-closing dev_math and dev_hard test traces, <= 0.2 on closing traces of every calculation set, <= 0.05 on dev_ifsynth | drop the re-check close step |
| U unit check of XL (step 11) | close-tag `after_vs_before` > 1 on dev_math, dev_hard and dev_gsm8k, <= 1.30 on dev_ifsynth and auxtest; B2 | drop the re-check close step (its runs are skipped) |
| C close step | unit check of the merged edit (close-tag `after_vs_before` > 1 on dev_math, dev_hard and dev_gsm8k, <= 1.30 on dev_ifsynth and auxtest; B2), then rule C | drop it; a failed unit check skips the close step's runs |
| K re-check close step | rule K (step 11) | drop it; rule C's choice stands |

A dose whose own build fails G4 or whose unit check fails B1 or B2, while another dose passes, is simply dropped:
rule D then chooses among the doses that passed every check (the fallbacks above are for a check that fails at
every dose).

## Commands
Placeholders in angle brackets (`<candidate>`, `<W1>`, `<B>`) are filled from this run's own measurements; nothing
below is a value to copy. `<set>`, `<src>` stand for each of the listed sets / sources. Decisions go into small
files that the env file reads (`$W/W1` the opener, `$W/V` the gate variant, `$W/X` the chosen edit, `$W/XC` the
merged edit of step 10, `$W/XL` the merged edit of step 11, `$W/HF` the GPU for teacher-forced steps).
```bash
# 0. once. From the agent machine: the entry's code and data, task.json and the parent edit into /workspace/work
#    ssh -F ~/.axonry/ssh/config sandbox 'mkdir -p /workspace/work/entry'
#    scp -F ~/.axonry/ssh/config -r submission/derive submission/data sandbox:/workspace/work/entry/
#    scp -F ~/.axonry/ssh/config -r task.json parent sandbox:/workspace/work/   (as BRIEF.md says: copy task.json there)
#    In the sandbox, one env file; start EVERY later command with `source /workspace/work/run/env.sh` (each ssh call
#    is a new shell: variables and functions do not survive it)
mkdir -p /workspace/work/run/logs; cat > /workspace/work/run/env.sh <<'EOF'
cd /workspace/work
E=/workspace/work/entry; W=/workspace/work/run; C=hx; AUX=$E/data/aux_if.jsonl
export WSP_CASES=${WSP_CASES:-/workspace/work/cases} WSP_WORK=${WSP_WORK:-/workspace/work/wsp}
RI="python $E/derive/run_items.py --case $C --items $AUX"
get() { cat $W/$1 2>/dev/null || true; }
W1=$(get W1); V=$(get V); X=$(get X); XC=$(get XC); XL=$(get XL); HF=$(get HF); HF=${HF:-0}
PEN="dev_math dev_hard dev_gsm8k dev_mmlupro"
GATE="--words $W1 --tau 0.01 --pos-runs $PEN --neg-runs dev_ifsynth auxfit --test-runs auxtest --items $AUX --traces 48 --traces-for dev_ifsynth=64 auxfit=160 --cache $W/gate_pass.npz"
ORDER="0.98,0.995 0.99,0.999 0.98,0.995,per-run 0.99,0.999,per-run 0.995,0.999,per-run 0.95,0.99"
job() { g=$1; n=$2; shift 2; CUDA_VISIBLE_DEVICES=$g nohup bash -c "$*" > $W/logs/$n.log 2>&1 & echo "$n: GPU $g"; }
pre() { python -c "import sys; from wsp.case import model_path; [print(model_path('$C', m)) for m in sys.argv[1:]]" "$@"; }
runf() { echo $WSP_WORK/runs/$C/$1__$2.jsonl; }
waitfor() { for i in $(seq $((${2:-10} * 6))); do [ -e "$1" ] && { echo "ready: $1"; return 0; }; sleep 10; done; echo "not yet: $1"; return 1; }
EOF
source /workspace/work/run/env.sh
python -m wsp.case init $C --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512
python $E/derive/sets.py --case $C                # dev_/holdout_<src>, dev_hard, holdout_hard (64 each)
pre base                                   # the parent model, built once (CPU) before parallel jobs use it

# 1. base runs (step 1); wait on run files (`waitfor $(runf base dev_hard) 10`), never on a PID
job 0 b-hard "python -m wsp.case run $C base dev_hard --n 4"
job 1 b-math "python -m wsp.case run $C base dev_math --n 4"
job 2 b-gsm  "python -m wsp.case run $C base dev_gsm8k --n 4 && python -m wsp.case run $C base dev_mmlupro --n 4"
job 3 b-if   "python -m wsp.case run $C base dev_ifsynth --n 4 && $RI --model base --split fit --tag auxfit --n 1 && $RI --model base --split test --tag auxtest --n 2"
#    A/A draws and the base holdout runs fill the GPUs that free up (see the time plan):
#    python $E/derive/aa_run.py --case $C --set dev_hard --n 4 --salt s1   (and --set dev_math, same way)
#    $RI --model base --split test --tag auxtest --n 2 --salt s1
#    python -m wsp.case run $C base holdout_<src> --n 2      (and holdout_hard)

# 2. diagnosis and opener candidates (CPU), then the decoding test (one GPU per candidate) and rule M
python -c "import json,sys; [print(s, {k: json.load(open(f'$WSP_CASES/$C/results/base__{s}.json'))['summary'][k] for k in ('acc','cut_share','acc_when_finished','kinds')}) for s in sys.argv[1:]]" $PEN dev_ifsynth
python $E/derive/markers.py --case $C --runs base__dev_math base__dev_hard base__dev_gsm8k base__dev_mmlupro --out $W/markers.json
CANDS=$(python -c "import json; print(' '.join(c['word'] for c in json.load(open('$W/markers.json'))['candidates']))")
job 1 dec-1 "python -m wsp.case run $C base@penalize=<candidate>,bias=-3 dev_hard --n 2"   # one such job per candidate
python $E/derive/compare.py --case $C --edit $(for w in $CANDS; do echo base@penalize=$w,bias=-3; done) \
    --sets dev_hard --json $W/cmp_markers.json                                     # rule M
echo '<W1>' > $W/W1; source $W/env.sh                                             # the candidate rule M chose

# 3. always-on units, on the GPU without a vLLM job ($HF), about a minute each
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_unit.py --case $C --words $W1 --beta $B \
    --name H_$B --device cuda; done

# 4. the gate: one pass (cached) with a dry report over the gate order (about 6-8 min), then one build per dose
CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_gate.py --case $C --name HXdry --ref-spec H_3 --dry $GATE \
    --variants $ORDER                                                          # G1-G4 per variant in HXdry__dry.json
python -c "import json; print(json.load(open('$WSP_CASES/$C/results/HXdry__dry.json'))['first_passing'])" > $W/V
source $W/env.sh; echo "variant: $V"                                           # None: see G3 in the checks
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_gate.py --case $C --name HX_$B \
    --ref-spec H_$B $GATE --variants $ORDER --choose $V; done

# 5. built-model checks (the always-on twins first, then each gated edit against its twin) and limits
pre H_2 H_3 H_4 HX_2 HX_3 HX_4
for M in H_2 H_3 H_4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit $M --words $W1 \
    --runs dev_hard dev_math dev_mmlupro dev_ifsynth auxtest --items $AUX; done
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit HX_$B --twin H_$B \
    --words $W1 --runs dev_hard dev_math dev_mmlupro dev_ifsynth auxtest --items $AUX; done      # B1-B2
for B in 2 3 4; do python $E/derive/merge.py --case $C --name HX_${B}lim HX_$B --dry-run --allow-parent-overlap; done   # L

# 6-7. generation: dose series on the penalized dev sets, instruction check with the strongest dose
job 0 x2-hard "python -m wsp.case run $C HX_2 dev_hard --n 4"      # and HX_2 / HX_3 / HX_4 x the four sets in $PEN:
#                                                                    twelve jobs, as GPUs free up
job 3 x4-if "python -m wsp.case run $C HX_4 dev_ifsynth --n 4 && $RI --model HX_4 --split test --tag auxtest --n 2"
#    optional reference: $RI --model H_4 --split test --tag auxtest --n 2
python $E/derive/compare.py --case $C --edit HX_2 HX_3 HX_4 --sets $PEN \
    --pool calc=math,gsm8k,mmlupro --json $W/cmp_dose.json                    # rule D
python $E/derive/compare.py --case $C --edit HX_4 H_4 --sets dev_ifsynth auxtest --pool if=ifsynth,aux_if \
    --json $W/cmp_if.json                                                      # rule I (uncl and accuracy rows)

# 8. first upload: X = the edit chosen by rule D
echo HX_<B> > $W/X; source $W/env.sh
python -m wsp.upload $C $X --report $W/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/${C}__$X/edit --task task.json
python -m wsp.upload $C $X --report $W/REPORT.md --task task.json

# 11a. re-check close step, only when the second switch says ON: pass and unit after the dose series (time plan); CPU parts any time
CUDA_VISIBLE_DEVICES=$HF python $E/derive/sp_collect.py --case $C --model base --runs base__dev_math base__dev_hard \
    base__dev_gsm8k base__dev_mmlupro base__dev_ifsynth --per-run 160 --per-run-for base__dev_math=256 base__dev_ifsynth=96 --opener-words $W1 --out $W/pass11.npz
CUDA_VISIBLE_DEVICES=$HF python $E/derive/dx_build.py --case $C --col $W/pass11.npz --name LT --words $W1 \
    --avoid-specs 'HX*'                                                     # the tool's defaults; check E: results/LT__build.json

# 9. holdout confirmation (rule H) of the chosen edit F ($X, $XC or $XL, whichever rules C and K chose)
job 0 h-math "python -m wsp.case run $C <F> holdout_math --n 2"   # and holdout_hard / holdout_gsm8k / holdout_mmlupro / holdout_ifsynth
python $E/derive/compare.py --case $C --edit <F> --sets holdout_math holdout_hard holdout_gsm8k holdout_mmlupro \
    holdout_ifsynth --pool calc=math,gsm8k,mmlupro --json $W/cmp_holdout.json

# 10. close step, only when the switch at the top says ON (rule C); XC = the merged edit's new name
echo HXC_<B> > $W/XC; source $W/env.sh                  # <B> = the dose of X
CUDA_VISIBLE_DEVICES=$HF python $E/derive/close_unit.py --case $C --name CL --pos-runs dev_math dev_hard dev_gsm8k \
    --neg-runs dev_mmlupro dev_ifsynth --avoid-specs 'HX*' 'LT*' --cache $W/close_pass.npz   # the tool's defaults
python $E/derive/merge.py --case $C --name $XC $X CL --allow-parent-overlap   # refuses unit collisions and over-limit edits
pre $XC; CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit $XC --words '</think>' --tau 0.02 \
    --tokens 0 --traces 8 --runs dev_math dev_hard dev_gsm8k dev_ifsynth auxtest --items $AUX   # whole traces: the close tag's decision positions come late
#    runs of $XC: the four sets in $PEN and dev_ifsynth (--n 4), auxtest (--n 2), then
python $E/derive/compare.py --case $C --edit $XC --ref $X --sets $PEN \
    --pool calc=math,gsm8k,mmlupro --json $W/cmp_close.json             # rule C: against X, dev sets only
python $E/derive/compare.py --case $C --edit $XC --sets dev_ifsynth auxtest --pool if=ifsynth,aux_if \
    --json $W/cmp_close_if.json                                     # rule I for XC
#    only if rule C holds: upload $XC as X was uploaded, then confirm it (step 9)
python -m wsp.upload $C $XC --report $W/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/${C}__$XC/edit --task task.json
python -m wsp.upload $C $XC --report $W/REPORT.md --task task.json
# 11b. re-check close step: merge onto the close step's edit (close step OFF: $X LT), check, runs, rule K
echo HXL_<B> > $W/XL; source $W/env.sh                  # <B> = the dose of X
python $E/derive/merge.py --case $C --name $XL $X CL LT --allow-parent-overlap   # refuses unit collisions and over-limit edits
pre $XL; CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit $XL --words '</think>' --tau 0.02 \
    --tokens 0 --traces 8 --runs dev_math dev_hard dev_gsm8k dev_ifsynth auxtest --items $AUX   # check U
#    runs of $XL: the four sets in $PEN and dev_ifsynth (--n 4), auxtest (--n 2), with the same seeds as $X, then
python $E/derive/compare.py --case $C --edit $XL --ref $XC --sets $PEN \
    --pool calc=math,gsm8k,mmlupro --json $W/cmp_late.json              # rule K: against XC (or X), dev sets only
python $E/derive/compare.py --case $C --edit $XL --sets dev_ifsynth auxtest --pool if=ifsynth,aux_if \
    --json $W/cmp_late_if.json                                      # rule I for XL
#    only if rule K holds: upload $XL as X was uploaded, then confirm it (step 9)
python -m wsp.upload $C $XL --report $W/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/${C}__$XL/edit --task task.json
python -m wsp.upload $C $XL --report $W/REPORT.md --task task.json

# 12. once the final upload is verified and nothing is left worth trying (as the brief says)
axonry-done
```

## Time plan (4 x RTX 4090, 24 GB each; 6 hours)
Trace generation dominates. The parent's answers are much shorter than its own parent's: most calculation answers
now close, so only the hardest problems still run to the limit.
- Prompts: 256 dev problems (64 per source) and 64 `dev_hard` problems at 4 samples; 512 holdout problems (128 per
  source) and 64 `holdout_hard` problems at 2 samples; 280 + 273 self-written prompts at 1 and 2 samples; 64
  `dev_hard` problems at 2 samples per opener candidate.
- Tokens: every answer at the profile's own limit (32,768 new tokens). Expect answers to average about 7k tokens on
  dev_gsm8k and 11k on dev_mmlupro (almost all of them close), about 15k on dev_math, 20k-24k on dev_hard (about
  half of them never close), and 4k-5k on instruction prompts.
- About 9,500 answers in the whole run (12,000 with the close step, about 14,000 with the re-check close step too), roughly
  110-150 M generated tokens. The re-check close step's pass teacher-forces about 830 (about 12 M tokens) of the parent's own traces.
- One vLLM job per GPU at a time (about 3,300-3,500 tokens/s on calculation sets, 4,000-4,700 on instruction sets;
  start-up 1-3 min). Teacher-forced steps go on a GPU with no vLLM job (`$HF`; vLLM reserves 90% of its memory).

The minutes below are estimates for one 4090, start-up included. After the first base runs, replace them by your
own `[gen]` timings (tokens and seconds per job are in the logs) and move the time gates if needed.

| Job (one GPU) | Minutes |
|---|---|
| dev_hard n4 (base) | 30 |
| dev_math n4 | 21 |
| dev_mmlupro n4 | 16 |
| dev_gsm8k n4 | 10 |
| dev_ifsynth n4 | 7 |
| self-written fit n1 | 6 |
| self-written test n2 | 9 (or two `--shard` halves on two GPUs) |
| holdout_<src> n2 | about the dev time of the same source; holdout_hard n2 about 15 |
| decoding test, dev_hard n2 | 15 per candidate |
| edited runs on penalized sets | about the base time (0.8-1.0x: faster only where spirals close) |
| always-on unit, gate build, unit check | about 1 min, 20 s, 30 s each |
| gate pass with its dry report | 6-8 (once; every build after it reuses the cache) |
| close pass | 10-13 |
| re-check close step's pass (step 11) | 12-16 |
| step 11's unit build with its emulation | 2-4 |
| re-check close step runs of XL (four penalized dev sets, dev_ifsynth, self-written test) | about XC's (about 90 GPU-min in all) |

| Clock | G0 | G1 | G2 | G3 | CPU / you |
|---|---|---|---|---|---|
| 0:00-0:12 | | | | | copy the entry, env file, init, sets, `pre base`, PROGRESS.md |
| 0:12-0:45 | base dev_hard | base dev_math, then A/A dev_hard | base dev_gsm8k, base dev_mmlupro | base dev_ifsynth, self-written fit and test, then A/A dev_math | markers once the penalized runs are in |
| 0:45-1:05 | H_2-H_4, gate pass, HX builds, unit checks (no vLLM; with several candidates, after rule M) | (A/A dev_hard), then a decoding test | a decoding test, then base holdout_hard | (A/A dev_math), A/A self-written test, then base holdout_ifsynth | rule M; G1-G4, B1-B2, limits L |
| 1:05-2:20 | dose series (twelve jobs), HX_4 on dev_ifsynth + self-written test, base holdout_math / _gsm8k / _mmlupro, as GPUs free up | | | | |
| 2:20-2:40 | re-check close step: its pass on $HF (12-16 min), then its unit (2-4 min) | | | | compare: rules D and I; REPORT.md; FIRST UPLOAD by 2:40; the re-check close step's checks on its reports |
| 2:40-4:00 | close step if ON: pass, merge, checks; then XC's and XL's merges and unit checks; their dev and instruction runs | holdout runs of X (five sets), as GPUs free up | | | rules C and K; while they are undecided, X's confirmation already runs |
| 4:00-5:00 | holdout runs of XC or XL if rule C or K chose it | | | | rule H on the final edit |
| 5:00-5:30 | | | | | final REPORT.md and final upload; then `axonry-done` |

Time gates:
- Start the close step's and the re-check close step's generation only if their dev and instruction runs (by your own
  timings) end by 4:00, and the holdout runs of the final edit by 5:00. If both do not fit, run XL's runs and leave XC's
  out: XL still holds X, CL and the unit of step 11, and rule K alone decides, against X.
- If the first upload has not happened by 3:15, upload the best edit that passes D, I and L at once.
- Plan the final upload to finish before the first reminder. When a reminder asks for an upload, upload the best
  checked edit with `python -m wsp.upload ... --task task.json`.
- Run all GPU work detached (`job`) and wait on run files or log lines in short waits (`waitfor`), not on PIDs. Find a
  job with `pgrep -f "[p]attern"` (a plain pattern also matches the shell that searches). PROGRESS.md says what runs
  where.

## What to expect (practice data only)
- On the parent, almost no gsm8k-type, multiple-choice or instruction practice answer runs to the limit (multiple-
  choice answers are still long, but they close); about a fifth of math practice answers still never close, and
  about half on the hardest level. With the gate, the never-closing share on the
  hardest level should fall by a quarter to a half.
- markers.py: several openers are over-represented in never-closing traces, most of them without a comma (planning
  and calculation openers); usually one hesitation interjection passes the candidate rule. Its excess is smaller than
  on a model without the finishing edit, so the 1.2 fallback may be needed. Its decoding test should lower the
  never-closing share on dev_hard by several points and raise exact accuracy there.
- The strict-grade gain should sit mostly on the math sets: several points on dev_hard, a few on dev_math, little on
  gsm8k-type and multiple-choice practice, where the parent already closes almost everything (rule D guards them).
  Sets of 64 problems are noisy: two base draws of the same model can differ by several points (instruction draws
  most), which is what the A/A runs show.
- On instruction practice and the self-written test prompts, the never-closing share and the accuracy should stay
  where the parent has them, within noise. The always-on unit should instead show more never-closing instruction
  answers and lower accuracy: that is the hazard the gate removes.
- In the gate report, the held-out AUC at decision positions should be close to 1, and the penalty fraction close to
  1 at calculation decision positions and close to 0 at instruction ones. The opener rate at calculation decision
  positions should fall several-fold, the same for the gated and the always-on unit, and stay unchanged on
  instruction prompts.
- The close step, when ON, should add a little where answers were ready but kept planning, and nothing elsewhere.
  Without a clear gain on the dev sets, it stays out.
- The re-check close step, when ON: its unit fires at re-check starts deep in long calculation traces and almost
  nowhere else, so its emulated extra close is high on the never-closing calculation traces and low on the traces that
  close. It should lower the never-closing share on dev_hard further and leave the instruction sets as XC has them.
- If the confirmation on the hardest holdout problems is not clearly positive, say so in REPORT.md.
