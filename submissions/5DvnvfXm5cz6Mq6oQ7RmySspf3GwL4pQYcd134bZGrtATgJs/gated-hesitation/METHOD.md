# Commit instead of second-guessing, where it is calculation: a context-gated hesitation penalty in the last layer

Close step: ON

(The line above is this method's only switch. OFF: skip step 10 entirely. ON: run step 10 when the time gates of
the time plan allow it. Nothing else depends on it.)

## The problem
This entry stacks on the published edit named in `entry.json` (`parent`). On that model, verbatim loops are rare,
but in calculation problems (and worked multiple-choice questions) a large share of answers still never close their
thinking: they run into the token limit and score zero. Read those traces: after a small slip or an unexpected
intermediate number, the model second-guesses a result it has already checked, sentence after sentence. Each
re-check opens with the same kind of word, a hesitation interjection set off by a comma at the start of a sentence,
and this goes on until the budget is gone. The answers that do close are mostly
right, so every spiral that closes is a likely point.

A penalty on that opener (its logits lowered by a few units at every position, as one always-on unit of the last
layer) removes much of the self-doubt in calculation: fewer answers run out, more are right. But in instruction and
writing tasks the same opener is how the model checks its draft against a constraint. With the opener suppressed
there, the model keeps redrafting without settling, and MORE instruction answers never close. So the penalty must know
where it is: a context gate that applies the full penalty where the model reasons through a calculation and none
where it follows instructions.

Not every opener that is frequent in never-closing traces is a hesitation. Those traces are long and repetitive, so
first-person planning ("I will write ...") and calculation openers ("So", "Then", "Let") are over-represented there
too. They also open ordinary reasoning, and lowering them changes the model everywhere without closing the spirals.
The penalty targets a hesitation interjection only, and a decoding test checks that lowering it closes spirals
(step 2).

Measure first, on the parent, per practice source (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`) and on the
self-written instruction prompts in `data/aux_if.jsonl` (never the evaluation benchmarks):
- accuracy;
- the share of answers that never close (`cut_share`: cut at the token limit, almost always inside the thinking;
  `compare.py`'s `uncl` rows count the ones whose thinking never closed);
- the accuracy of the answers that close;
- which hesitation interjections are over-represented in never-closing traces, compared with finished correct
  traces (`derive/markers.py`).

If the never-closing share of the calculation sources is small (under 10%, pooled over the three), or no hesitation
interjection stands out (step 2), this method does not apply: say so in REPORT.md and upload nothing.
Decide on the toolkit's strict grade: the boxed answer equals the gold after `wsp.grade._norm` (exact match, no
symbolic equivalence). This is the `exact` grade of `derive/compare.py`; report the lenient one (`mv`) beside it.

## Why it happens, and why the edit can be computed exactly
- Each spiral is a chain of decisions to re-open a checked result, and each re-opening starts with a hesitation
  opener. Once the opener is written, the next sentences re-derive the same thing. Making the opener unlikely lets the
  model commit to the result it has and close its thinking.
- The eval sampler (temperature, top-k, top-p, presence penalty of the profile) only draws the opener where the model
  gives it real probability. On the parent's own traces almost all of the opener's sampling mass sits at a small set
  of DECISION POSITIONS: thinking positions where the sampler would draw it with probability at least tau. A gate only
  has to be right at those positions.
- A unit of the last decoder layer's MLP (a design choice of this method, made for the reason that follows; which units,
  directions, thresholds and writes is all found by the run) writes into the final residual, which nothing reads
  except the final norm and the tied output head. It changes the next-token distribution of its own position only. So its effect on the
  parent's traces can be computed exactly from one stored teacher-forced pass, before any generation, with the eval
  sampler emulated (`derive/dplib.py`). A built-model check confirms it.
- At decision positions, the last MLP's input carries which kind of task the model is working on. A linear read (LDA)
  separates calculation reasoning from instruction reasoning there. Two units on that read make a saturating clamp:
  no penalty below a low threshold, the full penalty above a high one. One SiLU unit would not do: its write grows
  with the score, so it is weak at low-scoring calculation positions (the opener leaks through, and a weak penalty
  only delays a spiral) and too strong at high-scoring ones.
- The gated write is exactly the always-on unit's write. The two edits differ only in WHERE the penalty applies, so
  comparing them isolates the gate.


## Approach
Work in the toolkit's case workflow (`python -m wsp.case`), in the sandbox, from `/workspace/work`. Start the case
from the parent (`init ... --parent parent/edit/edit.json`), so "base" below is the parent model.
Use only practice data, the entry's self-written prompts and the model's own generations.
Every number in the edit (opener, unit, read direction, thresholds, write) comes out of this run.

1. **Splits and base runs.** Init the case over the four practice sources: dev 64 and holdout 128 problems per source.
   Write per-source set files (`derive/sets.py`) and build the parent model once (`pre base`) before any parallel
   job. Base runs at the profile's own token limit (never lower it: the never-closing share only shows at the real
   limit):
   - every dev set, 4 samples per problem;
   - the self-written prompts' `fit` split, 1 sample (gate negatives);
   - their `test` split, 2 samples (the held-out instruction check, never in any fit), with `derive/run_items.py`.
2. **Diagnosis, opener candidates, decoding test.** Read each base run's summary (`results/base__<set>.json`: `acc`,
   `cut_share`, `acc_when_finished`, `kinds`). The calculation sources (math, gsm8k, mmlupro) are the PENALIZED
   sources when their pooled `cut_share` is at least 10%. All three stay in even if one of them alone is lower: it is
   the same kind of reasoning, it simply gains less, and rule D guards it. The INSTRUCTION sources are `ifsynth` and
   the self-written prompts.
   - Candidates: `derive/markers.py` on the penalized sources' base runs. Its `candidates` are the hesitation
     interjections: sentence openers set off by a comma in at least half of their uses in never-closing traces, that
     open sentences at least 1.3 times as often there as in finished correct traces, ranked by the excess per 1,000
     sentences (at most three). Read a few never-closing traces at each: it should open re-checks of a result
     already found. No candidate: once more with `--min-ratio 1.2`; still none: the method does not apply.
   - Decoding test (for the choice only, never shipped): each candidate on dev_math with its logits lowered by 3 at
     decoding time (`run <case> base@penalize=<candidate>,bias=-3 dev_math --n 2`, the base run's seeds), one GPU
     each, compared with the pooled base (`compare.py`). Rule M: among the candidates whose test lowers the
     never-closing share (`uncl` d < 0) and keeps exact accuracy within its noise band (d >= -max(2 se, 1 point)),
     W1 is the one with the largest exact gain, except that a candidate ranked higher by markers.py is preferred
     when its gain is within 2 points of that largest gain. None passes: the lever does not apply here, upload
     nothing.
   - With a single candidate, start step 3 with it at once and confirm rule M before step 6. The test also bounds
     what the weight edit can do: the gated edit should come near its gain.
3. **Always-on units, a dose series** (`derive/hesitation_unit.py`). The write direction lowers W1's logits: the mean
   gradient through the parent's own final norm and tied head, scaled by bisection so the mean logit change is -B. It
   goes into the least-used unit of the last layer as an always-on unit. Build H_B for B = 2, 3, 4. Report the realized
   change, the most affected other tokens and the KL on the parent's text.
4. **The gate** (`derive/hesitation_gate.py`). One teacher-forced pass of the parent over base traces of every
   penalized and instruction run (cached, reused by every build). Its parts:
   - decision positions: p(W1) >= tau = 0.01 under the eval sampler;
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
   `derive/merge.py --dry-run` on the edit alone checks the subnet's limits exactly as the manifest computes them:
   per-tensor change vs the parent, global change, payload.
6. **Dose by generation.** Run HX_2, HX_3 and HX_4 on the penalized dev sets (4 samples, the same seeds as the base
   run). Also run a second base draw (another salt) on dev_math, dev_mmlupro and the held-out prompts (`derive/aa_run.py`,
   `run_items.py --salt s1`): it gives the A/A noise of the parent and a pooled base.
   Compare with `derive/compare.py --pool calc=<penalized sources>`. Rule D: pick the dose with the largest pooled
   exact gain whose row is clearly positive (d - 2 se > 0) and whose every penalized source stays within its noise
   band (d >= -max(2 se, 1 point)). A tie within 1 point goes to the smaller dose.
7. **Instruction check** (in parallel with step 6). Run the strongest dose HX_4 (the worst case for any leak of the
   gate) on dev_ifsynth and on the held-out self-written prompts. If a GPU is idle, also run the always-on H_4 on the
   held-out prompts: it shows the hazard the gate removes. Rule I, on `POOL if` and on the held-out prompts alone:
   - the never-closed rows (`uncl`) must not rise: d <= max(2 se, 1 point);
   - accuracy stays within its noise band: d >= -max(2 se, 1 point).
8. **First upload.** Package the edit chosen by D that passes I and the limits. Self-check it with `axonry-check` and
   upload it, with a REPORT.md (what was found, how the spec was derived, controls, what failed).
9. **Holdout confirmation.** Base and the chosen edit on every holdout set (2 samples per problem). Rule H:
   - the pooled penalized-source exact gain is positive (clearly positive: d - 2 se > 0, otherwise say so);
   - no source loses beyond its noise band;
   - holdout_ifsynth `uncl` does not rise (as in rule I).
   Replace the upload only with an edit that is better on this confirmation.
10. **Close step (optional; the switch at the top).** `derive/close_unit.py` builds a gated CLOSE unit:
    - it raises the thinking close tag by about 2 logits where the parent itself already gives it p >= 0.02 (the
      model is ready to answer);
    - it is fitted with the decision positions of the calculation runs (dev_math, dev_gsm8k) as positives, those of
      dev_mmlupro and dev_ifsynth as hard negatives (the self-written prompts stay out of this fit; auxtest is the
      held-out check), and every other stored position (prompts, answers, all
      other thinking) as easy negatives, so it stays silent at ordinary positions (it may also fire at other tasks'
      decision positions, where the model was about to close anyway);
    - it never reuses the gate's units (`--avoid-specs`).

    Where the penalty removed a spiral, thinking often ends in an "answer is ready" planning tail; this unit lets
    that tail close a little earlier. Merge it with the chosen HX_B (`derive/merge.py`, which refuses unit collisions
    and checks the limits) and check the merged edit with the close tag as the word (`unit_check.py`, 8 whole traces per run; read
    `after_vs_before` in `results/<XC>__check.json`): its mass at its decision positions rises on the calculation runs
    (dev_math, dev_gsm8k), rises by at most 30% on the instruction runs (dev_ifsynth, auxtest; the unit may fire at
    instruction decision positions, where the model is about to close anyway; +15-20% is normal), and B2 holds.
    Raise 2, q 0.999 and tau 0.02 are fixed settings of this procedure, never tuned on your runs: the unit, its read
    direction, threshold and write are computed by `close_unit.py` from this run's traces, and the unit stays only if
    rule C holds. Run the
    merged edit on the penalized dev sets, the instruction sets and the holdout calculation sets, with the same seeds
    as HX_B. Rule C:
    - against HX_B, with the dev and the holdout calculation sets in one `compare.py --ref HX_B` call (no set
      'skipped'): the `POOL calc` exact row d >= +1 point with d - se > 0;
    - in the same call: no `ALL math` / `ALL gsm8k` / `ALL mmlupro` exact row with d < -2 se, and the `POOL calc`
      `uncl` row (thinking never closes) d <= max(2 se, 1 point);
    - rule I for the merged edit itself, against the base, on the instruction sets.

    Only then does the merged edit replace the upload. Otherwise drop the close step and report it.
11. **Final upload and report.** The confirmed best edit, uploaded at least 15 minutes before the deadline. Report
    every measurement, including what failed.

## Checks, and what to do when one fails
| Check | Pass | If it fails |
|---|---|---|
| M opener (step 2) | rule M | the next candidate; none left: the lever does not apply here, upload nothing |
| G1 gate report `gate_fit.test_auc_dp` | >= 0.98 (held-out calculation vs instruction decision positions) | more traces (`--traces 64`, new cache file); still below: the context is not separable at this layer, the method does not apply: upload nothing |
| G2 `dp_mass_share` (every run) | >= 0.95 | lower `--tau` to 0.005 (new cache file) |
| G3 penalty fraction `test_dp.mean_frac` | >= 0.9 on every penalized run; <= 0.05 on every instruction run (incl. the held-out prompts) | the dry report walks the gate order and names the first variant that passes (`first_passing`). None passes: take the variant with the lowest instruction leak among those at >= 0.9 on every penalized run; if that leak is at most 0.10, build it and let rule I decide (a miss that small, on a few dozen test problems, is within the noise of the measurement; rule I overrules it); larger: the next opener candidate that passed rule M (a new pass); none left: the method does not apply |
| G4 openers per 1k thinking tokens at test DPs | gated within 1.5x of always-on on penalized runs; >= 0.9x parent on instruction runs | as G3 |
| B1 `unit_check` openers at DPs | penalized runs: HX_B within 1.2x of its always-on twin (`vs_twin` <= 1.2), or both below 1% of the parent's opener mass (a ratio of two near-zero residues is noise); instruction runs: >= 0.85x before. The after / before ratio itself is reported only: dose strength is rule D's matter | penalized runs: the gate passes too little of the penalty: the next looser variant of the gate order; instruction runs: the next stricter one (both rebuilt from the cached pass under new names) |
| B2 `unit_check` away from DPs | top-1 changed <= 2% and mean KL <= 0.01 nats on instruction runs | the next stricter variant, as B1 |
| L limits (`merge.py --dry-run`) | per tensor <= 0.45, global <= 0.18, payload <= 4 MiB | gate: `--max-rel 0.2`; merged: drop the close step |
| D dose | some dose passes rule D | run the always-on H_4 on dev_math: no gain either: the lever does not apply here, upload nothing; H_4 gains but no HX_B does: the gate is too strict on calculation (G3), take the next looser variant, rebuild, rerun the best dose |
| I instruction loops | rule I | the next stricter variant of the gate order (rebuilt from the cached pass under new names); rerun step 7 with it, and step 6 for the dose rule D chose; still failing: a smaller dose; still failing at B = 2: no upload of this lever, report it |
| H holdout | rule H | the next dose that passed D, confirmed the same way if time allows; otherwise keep the first upload and report the doubt |
| C close step | unit check of the merged edit (close-tag `after_vs_before` > 1 on dev_math and dev_gsm8k, <= 1.30 on dev_ifsynth and auxtest; B2), then rule C | drop it (never tune raise / q / tau); a failed unit check skips the close step's runs |

A dose whose own build fails G4 or whose unit check fails B1 or B2, while another dose passes, is simply dropped:
rule D then chooses among the doses that passed every check (the fallbacks above are for a check that fails at
every dose).

## Commands
Placeholders in angle brackets (`<candidate>`, `<W1>`, `<B>`) are filled from this run's own measurements; nothing
below is a value to copy. `<set>`, `<src>` stand for each of the listed sets / sources. Decisions go into small
files that the env file reads (`$W/W1` the opener, `$W/V` the gate variant, `$W/X` the chosen edit, `$W/XC` the merged edit of step 10,
`$W/HF` the GPU for teacher-forced steps).
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
W1=$(get W1); V=$(get V); X=$(get X); XC=$(get XC); HF=$(get HF); HF=${HF:-0}
GATE="--words $W1 --tau 0.01 --pos-runs dev_math dev_gsm8k dev_mmlupro --neg-runs dev_ifsynth auxfit --test-runs auxtest --items $AUX --traces 48 --traces-for dev_ifsynth=64 auxfit=160 --cache $W/gate_pass.npz"
ORDER="0.98,0.995 0.99,0.999 0.98,0.995,per-run 0.99,0.999,per-run 0.995,0.999,per-run 0.95,0.99"
job() { g=$1; n=$2; shift 2; CUDA_VISIBLE_DEVICES=$g nohup bash -c "$*" > $W/logs/$n.log 2>&1 & echo "$n: GPU $g"; }
pre() { python -c "import sys; from wsp.case import model_path; [print(model_path('$C', m)) for m in sys.argv[1:]]" "$@"; }
runf() { echo $WSP_WORK/runs/$C/$1__$2.jsonl; }
waitfor() { for i in $(seq $((${2:-10} * 6))); do [ -e "$1" ] && { echo "ready: $1"; return 0; }; sleep 10; done; echo "not yet: $1"; return 1; }
EOF
source /workspace/work/run/env.sh
python -m wsp.case init $C --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512
python $E/derive/sets.py --case $C
pre base                                   # the parent model, built once (CPU) before parallel jobs use it

# 1. base runs (step 1); wait on run files (`waitfor $(runf base dev_math) 10`), never on a PID
job 0 b-math "python -m wsp.case run $C base dev_math --n 4"
job 1 b-mmlu "python -m wsp.case run $C base dev_mmlupro --n 4"
job 2 b-gsm  "python -m wsp.case run $C base dev_gsm8k --n 4 && $RI --model base --split fit --tag auxfit --n 1"
job 3 b-if   "python -m wsp.case run $C base dev_ifsynth --n 4 && $RI --model base --split test --tag auxtest --n 2"
#    A/A draws and the base holdout runs fill the GPUs that free up (see the time plan):
#    python $E/derive/aa_run.py --case $C --set dev_mmlupro --n 4 --salt s1   (and --set dev_math, same way)
#    $RI --model base --split test --tag auxtest --n 2 --salt s1
#    python -m wsp.case run $C base holdout_<src> --n 2

# 2. diagnosis and opener candidates (CPU), then the decoding test (one GPU per candidate) and rule M
python -c "import json,sys; [print(s, {k: json.load(open(f'$WSP_CASES/$C/results/base__{s}.json'))['summary'][k] for k in ('acc','cut_share','acc_when_finished','kinds')}) for s in sys.argv[1:]]" dev_math dev_gsm8k dev_mmlupro dev_ifsynth
python $E/derive/markers.py --case $C --runs base__dev_math base__dev_gsm8k base__dev_mmlupro --out $W/markers.json
CANDS=$(python -c "import json; print(' '.join(c['word'] for c in json.load(open('$W/markers.json'))['candidates']))")
job 1 dec-1 "python -m wsp.case run $C base@penalize=<candidate>,bias=-3 dev_math --n 2"   # one such job per candidate
python $E/derive/compare.py --case $C --edit $(for w in $CANDS; do echo base@penalize=$w,bias=-3; done) \
    --sets dev_math --json $W/cmp_markers.json                                     # rule M
echo '<W1>' > $W/W1; source $W/env.sh                                             # the candidate rule M chose

# 3. always-on units, on the GPU without a vLLM job ($HF), about a minute each
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_unit.py --case $C --words $W1 --beta $B \
    --name H_$B --device cuda; done

# 4. the gate: one pass (cached) with a dry report over the gate order (about 6 min), then one build per dose
CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_gate.py --case $C --name HXdry --ref-spec H_3 --dry $GATE \
    --variants $ORDER                                                          # G1-G4 per variant in HXdry__dry.json
python -c "import json; print(json.load(open('$WSP_CASES/$C/results/HXdry__dry.json'))['first_passing'])" > $W/V
source $W/env.sh; echo "variant: $V"                                           # None: see G3 in the checks
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/hesitation_gate.py --case $C --name HX_$B \
    --ref-spec H_$B $GATE --variants $ORDER --choose $V; done

# 5. built-model checks (the always-on twins first, then each gated edit against its twin) and limits
pre H_2 H_3 H_4 HX_2 HX_3 HX_4
for M in H_2 H_3 H_4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit $M --words $W1 \
    --runs dev_math dev_mmlupro dev_ifsynth auxtest --items $AUX; done
for B in 2 3 4; do CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit HX_$B --twin H_$B \
    --words $W1 --runs dev_math dev_mmlupro dev_ifsynth auxtest --items $AUX; done      # B1-B2
for B in 2 3 4; do python $E/derive/merge.py --case $C --name HX_${B}lim HX_$B --dry-run; done   # L

# 6-7. generation: dose series on the penalized dev sets, instruction check with the strongest dose
job 0 x2-math "python -m wsp.case run $C HX_2 dev_math --n 4"      # and HX_2 / HX_3 / HX_4 x dev_math / dev_gsm8k /
#                                                                    dev_mmlupro: nine jobs, as GPUs free up
job 3 x4-if "python -m wsp.case run $C HX_4 dev_ifsynth --n 4 && $RI --model HX_4 --split test --tag auxtest --n 2"
#    optional reference: $RI --model H_4 --split test --tag auxtest --n 2
python $E/derive/compare.py --case $C --edit HX_2 HX_3 HX_4 --sets dev_math dev_gsm8k dev_mmlupro \
    --pool calc=math,gsm8k,mmlupro --json $W/cmp_dose.json                    # rule D
python $E/derive/compare.py --case $C --edit HX_4 H_4 --sets dev_ifsynth auxtest --pool if=ifsynth,aux_if \
    --json $W/cmp_if.json                                                      # rule I (uncl and accuracy rows)

# 8. first upload: X = the edit chosen by rule D
echo HX_<B> > $W/X; source $W/env.sh
python -m wsp.upload $C $X --report $W/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/${C}__$X/edit --task task.json
python -m wsp.upload $C $X --report $W/REPORT.md --task task.json

# 9. holdout confirmation (rule H); the base holdout runs were made in idle slots before
job 0 h-math "python -m wsp.case run $C $X holdout_math --n 2"     # and holdout_gsm8k / holdout_mmlupro / holdout_ifsynth
python $E/derive/compare.py --case $C --edit $X --sets holdout_math holdout_gsm8k holdout_mmlupro holdout_ifsynth \
    --pool calc=math,gsm8k,mmlupro --json $W/cmp_holdout.json

# 10. close step, only when the switch at the top says ON (rule C); XC = the merged edit's new name
echo HXC_<B> > $W/XC; source $W/env.sh                  # <B> = the dose of X
CUDA_VISIBLE_DEVICES=$HF python $E/derive/close_unit.py --case $C --name CL --pos-runs dev_math dev_gsm8k \
    --neg-runs dev_mmlupro dev_ifsynth --avoid-specs 'HX*' --cache $W/close_pass.npz   # defaults: tau 0.02, q 0.999, raise 2
python $E/derive/merge.py --case $C --name $XC $X CL            # refuses unit collisions and over-limit edits
pre $XC; CUDA_VISIBLE_DEVICES=$HF python $E/derive/unit_check.py --case $C --edit $XC --words '</think>' --tau 0.02 \
    --tokens 0 --traces 8 --runs dev_math dev_gsm8k dev_ifsynth auxtest --items $AUX   # whole traces: the close tag's decision positions come late
#    runs of $XC: dev_math / dev_gsm8k / dev_mmlupro / dev_ifsynth (--n 4), auxtest (--n 2), and
#    holdout_math / holdout_gsm8k / holdout_mmlupro (--n 2, as in step 9), then
python $E/derive/compare.py --case $C --edit $XC --ref $X --sets dev_math dev_gsm8k dev_mmlupro holdout_math \
    holdout_gsm8k holdout_mmlupro --pool calc=math,gsm8k,mmlupro --json $W/cmp_close.json   # rule C: against X
python $E/derive/compare.py --case $C --edit $XC --sets dev_ifsynth auxtest --pool if=ifsynth,aux_if \
    --json $W/cmp_close_if.json                                     # rule I for XC
#    only if rule C holds: upload $XC as X was uploaded
python -m wsp.upload $C $XC --report $W/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/${C}__$XC/edit --task task.json
python -m wsp.upload $C $XC --report $W/REPORT.md --task task.json
```

## Time plan (4 x RTX 4090, 24 GB each; 6 hours)
Trace generation dominates.
- Prompts: 256 dev problems (64 per source) at 4 samples, 512 holdout problems (128 per source) at 2 samples,
  280 + 273 self-written prompts at 1 and 2 samples, and 64 dev_math problems at 2 samples per opener candidate.
- Tokens: every answer at the profile's own limit (32,768 new tokens): calculation answers average 15k-24k tokens
  because many never close; instruction answers 6k-8k.
- About 9,000 answers in the whole run (11,000 with the close step), roughly 110 M generated tokens.
- One vLLM job per GPU at a time (about 3,300-3,500 tokens/s on calculation sets, 4,000-4,700 on instruction sets;
  start-up 1-3 min). Teacher-forced steps go on a GPU with no vLLM job (`$HF`; vLLM reserves 90% of its memory).

The minutes below are estimates for one 4090, start-up included. After the first base runs, replace them by your
own `[gen]` timings (tokens and seconds per job are in the logs) and move the time gates if needed.

| Job (one GPU) | Minutes |
|---|---|
| dev_math n4 (base) | 33 |
| dev_mmlupro n4 | 27 |
| dev_gsm8k n4 | 21 |
| dev_ifsynth n4 | 13 |
| self-written fit n1 | 9 |
| self-written test n2 | 14 (or two `--shard` halves on two GPUs) |
| holdout_<src> n2 | about the dev time of the same source |
| decoding test, dev_math n2 | 17 per candidate |
| edited runs on penalized sources | about the base time (0.85-1.0x: faster only where spirals close) |
| always-on unit, gate build, unit check | about 1 min, 20 s, 30 s each |
| gate pass with its dry report | 6-8 (once; every build after it reuses the cache) |
| close pass | 10 |

| Clock | G0 | G1 | G2 | G3 | CPU / you |
|---|---|---|---|---|---|
| 0:00-0:12 | | | | | copy the entry, env file, init, sets, `pre base`, PROGRESS.md |
| 0:12-0:47 | base dev_math | base dev_mmlupro, then A/A dev_mmlupro | base dev_gsm8k, self-written fit, then a decoding test | base dev_ifsynth, self-written test, then base holdout_ifsynth | markers once dev_math is in |
| 0:47-1:10 | H_2-H_4, gate pass, HX builds, unit checks (no vLLM; with several candidates, after rule M) | (A/A), then a decoding test or A/A self-written test | (decoding test) | a decoding test or base holdout_gsm8k | rule M; G1-G4, B1-B2, limits L |
| 1:10-2:45 | dose series (nine jobs), HX_4 on dev_ifsynth + self-written test, base holdout_math / _mmlupro, as GPUs free up | | | | |
| 2:45-3:05 | | | | | compare: rules D and I; REPORT.md; FIRST UPLOAD by 3:05 |
| 3:05-3:45 | holdout runs of the chosen edit X (four sources), one GPU each | | | | rule H; re-upload only if needed |
| 3:45-5:00 | | | | | close step if ON: pass (it may run earlier on any idle GPU once the HX builds exist; was: any idle GPU), merge, checks, dev and holdout runs, rule C |
| 5:00-5:40 | | | | | final decision, REPORT.md, final upload by 5:40 at the latest |

Time gates:
- Start the close step's generation only if its dev and holdout runs (by your own timings) end by 5:00.
- If the first upload has not happened by 3:30, upload the best edit that passes D, I and L at once.
- Never start an upload in the last 10 minutes, unless no complete upload exists yet.
- Never block one command for more than about 15 minutes (the run is stopped after 2 x 20 minutes without
  activity): all GPU work runs detached (`job`); wait on run files or log lines in short waits (`waitfor`), never
  on a PID (the sandbox's PID 1 does not reap finished jobs, so `kill -0` on one still succeeds); find a job with
  `pgrep -f "[p]attern"` (a plain pattern also matches the shell that searches, and `pkill` would kill it).
  PROGRESS.md says what runs where.

## What to expect (practice data only)
- On the parent's math practice, a third or more of the answers never close; on gsm8k and multiple-choice practice,
  one in ten to one in four. With the gate, these shares should fall by a quarter to a half.
- markers.py: several openers are over-represented in never-closing traces, most of them without a comma (planning
  and calculation openers); usually one hesitation interjection passes the candidate rule, sometimes two. Its
  decoding test should lower the never-closing share on dev_math by several points and raise exact accuracy.
- The strict-grade gain on the penalized sources should be clearly positive. Practice ranges: about +5 to +12 points
  on math, +2 to +7 on gsm8k, +2 to +6 on multiple choice. Sets of 64 problems are noisy: two base draws of the same
  model can differ by several points (instruction draws most), which is what the A/A runs show.
- On instruction practice and the held-out self-written prompts, the never-closing share and the accuracy should stay
  where the parent has them, within noise. The always-on unit should instead show clearly more never-closing
  instruction answers and lower accuracy: that is the hazard the gate removes.
- In the gate report, the held-out AUC at decision positions should be close to 1, and the penalty fraction close to
  1 at calculation decision positions and close to 0 at instruction ones. The opener rate at calculation decision
  positions should fall several-fold, the same for the gated and the always-on unit, and stay unchanged on
  instruction prompts.
- The close step, when ON, should add a little on gsm8k-like problems (answers that were ready but kept planning) and
  nothing elsewhere. Without a clear gain, it stays out.
- If the strict-grade gain on the penalized holdout sets is not clearly positive, keep the upload small or upload
  nothing, and say so.
