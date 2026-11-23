# Early decisions: credit the stretch of the thinking where a maths answer is won or lost

This entry starts from `0009554615-0007` (`entry.json` "parent") and builds three layers in the run, then uploads them as
one edit on that start. Two layers re-derive published methods with their own scripts; the third is this entry's idea.

| layer | what | source |
|---|---|---|
| L1 | reversed shortcut: the policy gradient of the start's own maths answers, reversed (thinks longer, finishes more) | method and scripts of `0009554110-0010` (`derive/mkdata.py`, `shard_gen.py`, `svf_len.py`, `lendose.py`, `start.py`, unchanged) |
| BR | answer-branch preference gains for instruction requests, from 6 answers per request (1.5 times the published 4: 1.5 times the branch pairs), 4 fits | method and scripts of `0009552831-0015` (`derive/sets.py`, `fit_gains.py` unchanged; `branch2.py`, `pref_gains2.py` are its `branch.py`, `pref_gains.py` with the additions described below) |
| EC | **early-decision credit** for maths: preference gains fitted on same-prefix continuation pairs taken from early points of the thinking, with the loss restricted to the stretch right after the branch | this entry (`derive/run.sh` steps `ec_*`, `merge_gains.py`, the `--window` / `--bad-cut` / `--points` options of `pref_gains2.py`, the `right` kind of `branch2.py`) |

## 1. The failure: a maths answer is decided during the first half of the thinking

On practice maths problems that the model solves only some of the time, take one finished wrong and one finished correct
trace of the same problem and let the model continue each of them, four times, from a point inside its thinking (the
text up to that point kept, the rest sampled again at the profile's decoding, 24k new tokens). Measured on 200 mixed
`math_train` problems (from scratch the model solves them 52.6 % of the time; the model was this entry's L1 + BR stack):

| continued from | prefix of the correct trace | prefix of the wrong trace | gap |
|---|---|---|---|
| 10 % of the thinking | 43.9 % | 33.9 % | 10.0 |
| 25 % | 51.6 % | 33.1 % | 18.5 |
| 50 % | 58.0 % | 26.9 % | 31.1 |

The same measurement on another build of the stack (other seeds and samples in both layers, 194 problems): gaps 10.3 /
15.2 / 23.7. So:
- **The opening alone does not decide it.** At 10 % the two prefixes are only 10 points apart; the gap triples by the
  middle of the thinking.
- **The end is too late.** By the middle of the thinking a wrong trace is mostly lost: continuations from 50 % of it
  pass 26.9 %, half the from-scratch rate. Credit that acts at the end (closing, boxing, wrap-up, answer-stage credit,
  preference pairs branched late in a wrong trace) arrives after the outcome is set.
- **The decision builds up over the first half**, over a stretch of a few thousand tokens, not in one token.
- **Whole-trace credit is diluted.** One right/wrong bit per 14k-token trace spreads its gradient over everything that
  follows the decision: length, hesitation, style.

About half of all early branch points give a mixed outcome (some of the 4 continuations right, some wrong), so the same
prefix yields pairs that differ only in what was decided after the branch.

## 2. The edit (layer EC)

1. **Bank.** The current model (L1 + BR) answers 600 `math_train` problems of levels 3-5 (outside every set the layers
   use), 4 times each at the profile's decoding and token limit.
2. **Mixed problems.** Problems with at least one finished wrong and one finished correct answer, both longer than 3000
   tokens; up to 200 of them, one wrong and one correct trace each.
3. **Early branches.** Each trace is cut at a line break near 10 %, 25 % and 50 % of its thinking and continued 4 times
   (`branch2.py --kinds wrong|right --points 0.1 0.25 0.5 --k 4 --max-new 24000`, presence penalty on the prefix as in
   the published `branch.py`).
4. **Pairs.** At each branch point, a continuation that ends with a correct answer is paired with one that does not
   (an unfinished continuation counts as not correct: `--bad-cut`), at most 2 pairs per point.
5. **Fit.** Per-unit MLP gains and per-head attention gains of layers 4-21 (the parameterisation of the published
   `pref_gains.py`), DPO-style preference (beta 0.1) plus a small NLL term on the correct continuation, **with the loss
   restricted to the first 2048 tokens after the branch point** (`--window 2048`). Four fits, seeds 1-4, 2 epochs.
6. **Strength, chosen in the run.** The four fits' log gains are averaged. How far the same gains move the thinking
   differs from build to build, so the strength is chosen on fresh practice maths (`run.sh` step `ec_dose`): the
   gains are merged at 0.5, 0.75 and 1.0, and the L1 + BR model and each candidate answer the same 256 fresh
   `math_train` problems of levels 3-5 (the next 256 after the bank, outside every set; one answer each at the
   profile's decoding and limit). The largest strength whose mean thinking length is at most 1.18 times the L1 + BR
   model's and whose share cut at the limit is at most 2 points higher (or at most 6 %, if that is more) is used; if none
   passes, 0.5. The reason: at a
   strength that lengthens the thinking too far, more answers run into the token limit (held-out practice at full
   strength: share cut 2.2 % -> 5.2 %), while on a build where the gains barely change the length, half strength is
   too weak to matter.

Why 2048 tokens: the gap between the two prefixes grows over a few thousand tokens after the branch points (section 1);
a window of that order holds the decision, and a much longer one would bring back the dilution of whole-trace credit.

## 3. What it does on practice

Held-out practice problems (never used for the bank, the branches or the fits; drawn from `math_train` and
`gsm8k_train`), 4 samples each at the profile limit, paired by problem, the L1 + BR stack with and without EC at full
strength: `math_train` +1.87 ± 1.04 (375 problems), `gsm8k_train` +0.83 ± 1.56 (150 problems). The accuracy of the
answers that finish rises clearly; the share cut at the limit rises too, which the strength screen (step 6) keeps in check.

## 4. Merging the layers into one upload

BR's gains (strength 1.0) and EC's gains (the strength of step 6) act on the same MLP units and attention heads, and gains
on the same columns compose by multiplication, so the two are merged in log space into **one** gain set
(`merge_gains.py`) and exported once on top of L1 (`fit_gains.py export --k 1.0`). The upload is L1's ops plus that one
gain set (`python -m wsp.upload l1 EC`: the chain start -> L1 -> EC, local layers carried into one edit). This keeps the
payload at about 3.8 MB (L1 3.4 MB + one gain set 0.4 MB) instead of 4.2 MB for two separate gain sets.

## 5. Steps (all scripted: `bash submission/derive/run.sh all` from the task directory)

`run.sh` runs each step in order, writes `run_ec/<step>.done`, and resumes after an interruption (run it again; a single
step: `run.sh <step>`). Log: `run_ec/logs/run.log`.

| step | what | GPUs | time (4 x RTX 4090, measured in our rehearsal) |
|---|---|---|---|
| `start` | the start's edit (`start.py --task task.json`) | - | - |
| `l1_pool`, `l1_fit` | L1 as published: pool (gsm8k 400, math 200, ifsynth 80, mmlupro 80, x4, 12k), 4 fits (seeds 1-4; a seed that fails is run once more), **mean of the 4 seeds** reversed at L2 norm 1.37 (`lendose.py --fit L1 L2 L3 L4 --out Lm --norms -1.37`) | 4 | 26 min |
| `br_init` ... `br_fits` | BR as published, on L1's model, with **6 answers per instruction request** (tuluif A, ifsynth x6, calc x1, 6144 tokens; split over the GPUs), branches of every IF answer (k 4), **4 fits, each on 1.5 times the pairs** (one round on 4 GPUs), export strength 1.0 | 4 | ~60 min |
| `up1` | **upload 1**: L1 + BR (a complete, valid edit in case the rest does not finish) | - | 1 min |
| `ec_bank` ... `ec_fits` | EC (section 2 steps 1-5) on the L1 + BR model | 4 | ~185 min (bank 50, branches ~90, fits ~45) |
| `ec_dose` | EC's strength on fresh practice maths (section 2 step 6) | 4 | ~30 min |
| `up2` | merge at the chosen strength, export, payload check, **upload 2** (replaces upload 1) | - | 4 min |

Total about 5.1 hours on 4 x RTX 4090, leaving time for setup and checks within the 6 hours.

Time guard: `run.sh` counts the 6 hours from its first start. **If time was used before it starts, pass the minutes
already used**, e.g. `T_OFFSET_MIN=20 bash submission/derive/run.sh all`. EC starts only if at least 150 minutes remain
(`EC_MIN_LEFT`), otherwise upload 1 stands; the strength screen runs only if at least 50 minutes remain
(`EC_DOSE_MIN_LEFT`), otherwise strength 0.5. If a step fails, read its log under `run_ec/logs/`, fix the cause (paths, a
GPU that did not start: rerun the step) and rerun `run.sh all`; done steps are not repeated. Each parallel job uses its
own kernel cache (`TRITON_CACHE_DIR`), so jobs that start together on a fresh machine do not race in compilation.

## 6. What to expect, and what not to do

- Upload 2's maths should be clearly above upload 1's on held-out practice maths (paired), with lengths somewhat up.
- Do not override EC's strength chosen by the screen (step 6), and do not drop the 50 % branch point: in our tests the
  version without it neither lengthened the thinking nor helped.
- Do not shrink the EC data (`EC_BANK`, `EC_ROWS`, `EC_POINTS`) to save time: in our tests a smaller version (130
  problems, branch points 10 % and 25 % only, about 550 pairs instead of about 1250) gave a much smaller effect.
- Do not add length-matching to the EC pairs: the window already removes the continuation's length from the loss.

## 7. Notes from our practice runs

- Preference pairs branched late in finished wrong traces with the loss over the whole continuation (the published
  branch method applied to maths): few usable pairs, because most continuations from that far in do not finish within
  the token limit, and no effect.
- Correctness credit over the whole trace on fixed samples mostly changes the length of the thinking (correct practice
  answers are shorter than wrong ones), which is why EC credits a window after a shared prefix instead.
