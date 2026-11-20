# Two layers that make the model finish: refit its own exits from loops, then close stuck thinking and commit its answer

One method in two layers: layer 1 refits the model's own exits from loops, layer 2 is fitted in the same run on the
model layer 1 produces, and the two are uploaded as one edit.

## 1. The problem, and why two layers
`entry.json` names `"champion"` as the parent, so the starting model is the champion when the run starts. The
published champions so far stack on the loop-removal edit, which already took out the main loop direction. The
starting model still fails to reach an answer on a share of problems:
- part of its math and multiple-choice answers are cut at the generation limit (`max_tokens` of the profile), in a
  loop or in long reasoning that does not converge, with `<think>` still open: they score 0;
- finished answers are far more accurate than answers overall;
- most problems cut in one sample are solved, much more briefly, in another.

The ability is there; the model does not reliably conclude. Two causes, one layer each:
- **The model does not take its own way out of a loop** (layer 1). When its thinking is closed where it starts to
  repeat itself, it often still answers correctly: those continuations are its own exits. A small set of MLP units
  carries the choice between going on and concluding; refitting their output weights to the model's own exits and
  concise answers shifts that choice for all answers.
- **What layer 1 leaves stuck is not stopped, and a late close does not commit** (layer 2). Even after layer 1
  some math and gsm8k traces still run to the limit. Layer 2 is measured on layer 1's own traces, not on the
  starting model's: at the newlines of the traces that stay stuck (the only places `</think>` can follow), the MLP
  input of an upper layer separates them from early thinking and from traces that close on their own, yet nothing
  there writes the model's own closing direction (the gradient of log P(`</think>`) at its natural closes). And
  when a stuck trace is closed late, the answer often loops at once and never states a boxed result; committing
  depends on two things at once, the first answer position and lateness, which one linear read cannot express. A
  few gated neurons fitted on these measurements add the missing stop rule and the commitment.

The layers act on different things: layer 1 changes how often the model gets stuck; layer 2 acts only once a trace
is stuck, and is silent before. Layer 2 is fitted on layer 1's generations, so it targets what layer 1 leaves.

**Where the layers come from.** Layer 1 is the exit refit of submission 0009538688-0020 (finish-exits), with its
recipe and rule. Layer 2 consists of the wrap-up and commit units of submission 0009538688-0062 (wrapup-commit),
with its scripts, rules and thresholds (`derive/wu/`), adapted only so that its parent is layer 1's model, its units
take no neuron that layer 1 writes, and it uploads nothing itself. The method is the pairing: layer 2 is measured
and fitted on layer 1's own generations (its stuck traces, its closes, its answers), so it targets what layer 1
leaves and stays silent where layer 1 already concludes; the two are uploaded as one stacked edit, rebuilt and
checked tensor by tensor against the measured chain, under the decision rule and fallbacks of section 8.

Everything is derived in the run, from the toolkit's practice sets and the model's own generations: every layer,
neuron, direction and value comes from the rules below, applied to this run's measurements.

## 2. Layer 1: teach the model to finish from its own exits
- **Pool.** 1200 practice problems outside DEV and HOLDOUT (gsm8k, math, mmlupro, ifsynth), 4 samples each.
- **Exits.** Every pool answer cut in a loop is cut at the sentence before the point where it starts repeating
  itself, and the thinking is closed there (`</think>`, the only inserted text). The model answers twice; the
  continuations the toolkit's grader marks correct become records. Exits come only from answers cut in a loop, at
  the onset of the repetition, and each correct continuation is a record of its own. Instruction-following problems
  get no exits
  (exits taken from them taught the model to close its thinking before checking the constraints).
- **Records.** The exits, learned from the exit on; and each problem's shortest correct answer: its answer part only
  for gsm8k, math and mmlupro, in full (thinking included) for ifsynth.
- **Units and fit** (`derive/train.py`). The gradient of the objective with respect to every down_proj column of
  every decoder layer except the last selects 1024 columns; only those change, each replaced as a whole column
  (`set_cols`, no gains and no direction removal), fitted to raise the
  likelihood of the records under the model's own sampling distribution (the profile's temperature and presence
  penalty), at a low learning rate with an average over the last steps. `--alphas` exports strengths 1, 1.5 and 2
  of one solution.
- **Rule** (`derive/rank.py`), from the DEV run paired with the starting model: r = cut share / the starting
  model's; g = overall DEV gain. GOOD: 0.25 <= r <= 0.80, g >= 3, no change on gsm8k, math or mmlupro below -9,
  ifsynth not below -12. OVER (r < 0.25): the strength-1 export is evaluated too. WEAK with 0.80 < r <= 1: strength 2
  too. SAFE: not GOOD or OVER, g > 0, the same floors. FINAL = the best GOOD by g, else the best SAFE, else the best g;
  with two GOOD seeds, the IF check (`rank.py --final`) chooses between them.

## 3. Layer 2: close stuck thinking at a newline, then commit the answer
Layer 2 is the wrap-up and commit procedure of submission 0009538688-0062 (wrapup-commit), fitted here on layer 1's
model (section 1). Each unit is one gated MLP neuron written in place of the least-used neuron of its layer (ops `set_rows` on
`gate_proj` and `up_proj`, `set_cols` on `down_proj`, the form of `wsp/gate.py`), not a neuron that layer 1 or
another unit writes:
- gate row = g (d - theta m): `d` a read direction on the layer's MLP input, `theta` a threshold, `m` the constant
  direction (`m . x ~ 1`); the unit is silent while `d . x < theta`;
- up row = k m; down column = dose x w / a: `w` the residual write that raises the target token, `a` the unit's
  median activation where it fires.

**Wrap-up units.** `late_nl` reads late thinking at newline positions only: `</think>` can only follow a newline,
and a unit that fires mid-line acts only through later layers' attention, so its write is mostly lost. Its negatives
are early thinking, answers and every non-newline thinking position. `late` reads late thinking at any position
(its complement and fallback). Both write the mean gradient of log P(`</think>`). Their reads are chosen by how many
stuck traces they reach (`stuck_cov`), not by per-position recall: a stuck trace offers thousands of positions, so a
read with low recall still reaches most of them.

**Commit units.** After a late close the answer often loops and never writes `\boxed{}`; it scores 0 like the cut
trace. BF acts at the first answer position after a LATE close and raises the starter tokens of `\boxed{` (found by
tokenizing `\boxed{`, `$\boxed{`, ` $\boxed{`, ` \boxed{`); BX reads a starter in the answer and raises `boxed`.
"First answer position AND late" is a conjunction one linear read cannot express, so BF uses both MLP reads: its
gate reads "first answer position", its up row reads "late", and the SiLU-gated product is large only where both
hold. A commitment is only worth something there: commitments pushed over the whole answer section are mostly made
inside answer loops and are wrong, and early-closed answers, which box their result anyway, get disturbed. Both
units are off once `boxed` is written.

The number of units is the number of distinct reads (up to two wrap-up reads, BF and BX), not a setting to tune;
strength comes from calibration and the dose rule.

## 4. What is uploaded
One edit on the published parent: layer 1's ops followed by layer 2's frozen ops (`derive/stack.py`), only
`set_rows` / `set_cols`, no neuron written by both. It must be bit-identical, tensor by tensor, to the chain that
layer 2 was measured on (published parent, then layer 1, then layer 2). Layer 1 alone is uploaded first, as soon as
it is chosen, and stays the upload unless the stack passes every rule of section 8.

## 5. Running it
Code in `derive/`, built on the toolkit (`wsp.case`, `wsp.gen`, `wsp.grade`, `wsp.edits`, `wsp.gate`, `wsp.upload`).
All GPU work runs in one detached chain, `derive/chain.py`, which runs layer 1, then layer 2 (`derive/stage2.py`,
which runs the scripts in `derive/wu/`), then the upload rule. For set_cols specs the package must also hold the
set_cols `values` arrays, which `wsp.upload`'s export leaves out: `derive/deliver.py` packages and uploads with
`wsp.upload`'s own `package`, `target` and `upload`, and includes those arrays (Uploads, below: why, and how to check
it). `python submission/derive/chain.py ax upload` runs it.

The skill's ground rules hold: practice data only, choices on DEV, HOLDOUT only after the freeze.

Work in `/workspace/work`. From the unpacked bundle, copy the three parts there:
```
scp -F ~/.axonry/ssh/config -r submission parent task.json sandbox:/workspace/work/
```
so that `/workspace/work/parent/edit/edit.json` and `/workspace/work/task.json` exist.

**Step 0, the case**, stacked on the parent, with four practice sources. Run it:
```
cd /workspace/work && python -m wsp.case init ax --profile qwen35-0.8b-think --parent parent/edit/edit.json \
  --practice gsm8k_train math_train mmlupro ifsynth --subset holdout256=holdout:256
```
It prints `{'dev': 256, 'holdout': 512, 'holdout256': 256}`. `mmlupro` holds the toolkit's MMLU-Pro practice items;
`ifsynth` holds GSM8K train questions with added instruction constraints.

**Start the chain**, right after step 0:
```
python submission/derive/chain.py ax start
```
- It returns at once; the chain runs in its own session and writes one line per event to `logs/status.log`.
- It reads the deadline from `/etc/axonry-deadline`. If `start` says the deadline is not usable, start it as
  `DL=$(date -d '<deadline date and time from the brief> UTC' +%s) python submission/derive/chain.py ax start`
  (the real deadline; the chain itself keeps 250 minutes of it for layer 2).
- `start` again resumes where the chain stopped (every output is kept). While a chain runs, `start` changes nothing
  and starts no second chain.

**What the chain runs, and its files.** The chain runs every GPU step and the layer-2 scripts itself
(`derive/stage2.py`, the scripts of `derive/wu/`, `derive/stack.py`), with the GPU split of section 9. Before each job
it waits, up to 5 minutes, until the GPUs that job uses are free; a GPU job started next to it competes for the same
GPU memory. Its settings (the recipe and the schedule at the top of `chain.py` and `stage2.py`) are fixed before the
run. The run's state files (`final.txt`, `final2.txt`, `stage1.txt`, `uploaded.txt`, `deadline*.txt`,
`stage2_state.json`, the locks), `stage1/` and the case `cases/wrapup` are the chain's own: it reads them to resume,
and layer 2 is measured against `stage1/edit` exactly as uploaded. HOLDOUT runs for the report only (section 9).

**Check on the chain** at least every 10 minutes, so that a failed step is seen early (the run's stall rule:
`stall_minutes` = 20), with one command of at most 9 minutes:
```
cd /workspace/work && python submission/derive/chain.py ax status --wait 8
```
It returns as soon as the chain reports something new, after 8 minutes at most (if a command is stopped earlier, use
a smaller `--wait`). It prints the last status lines (layer 2's start with `L2`), layer 1's table, `final.txt`
(layer 1's choice), `final2.txt` (layer 2's decision), what is uploaded, `uploading: <edit>` while any upload is in
progress (the chain's or one started with `upload`), and `L` (real minutes left). Between checks, work on REPORT.md
and PROGRESS.md.

**If `status` says `chain NOT running` before `DONE`** (and it was not stopped at the end), read the last lines of
`logs/status.log` and the log it names. If the cause is an environment problem (for example a full disk), free space
or fix it inside `/workspace/work` and run `start` again; done steps are kept.

**PROGRESS.md**: keep it current; its first line is the status. Note in it how the chain was started.

**task.json** holds the upload credentials: keep its contents out of logs, PROGRESS.md and the report.

**The chain's uploads.** The chain uploads by itself (CPU only, in the background, one upload at a time, each
starting with at least 50 minutes left for layer 1's uploads or 40 for layer 2's, except a re-delivery of layer 1's
FINAL after a failed layer-2 step or upload, which needs at least 15): layer 1's first GOOD or SAFE pick, then its
FINAL, then the stack if it passes. If an upload fails after its files began to go up, the chain at once delivers the
previous upload again, so a complete one is always in place. Each upload shows in `status` (`uploading:`, then
`uploaded:`) and in `logs/deliver.log`. The final upload, with the finished report, is yours (at the end, below).
`python submission/derive/chain.py ax stop` waits for an upload in progress to end, then stops the chain.

**Uploads: `derive/deliver.py`** (the chain's uploads and `chain.py ax upload` run it). It writes the axonry.upload/1
package (the three files, then manifest.json last) and differs from `python -m wsp.upload` only in packaging, for
set_cols specs (layer 1's and the stack keep their `values` arrays in the spec's own npz):
- It calls `wsp.upload`'s own `package`, `target` and `upload` unchanged, and uploads only `edit/edit.json`,
  `edit/edit.npz` and `edit/REPORT.md`, then `manifest.json`.
- The packaging difference: it copies the `values` arrays of the set_cols ops into `edit/edit.npz` when the
  toolkit's export has not already done so (toolkits before commit cdae10e left them in the case's own npz, which is
  not uploaded); with a newer toolkit its package equals `wsp.upload`'s. You can check a package with the toolkit's
  own commands once seed 0 is trained, for example on `S0_a1.5`. The first command only packages, into a separate
  folder, and uploads nothing:
  ```
  cd /workspace/work && WSP_WORK=/workspace/work/wspcheck python -m wsp.upload ax S0_a1.5 --dry-run
  axonry-check /workspace/work/wspcheck/deliver/ax__S0_a1.5/edit --task task.json
  ```
- It runs `axonry-check --task task.json` on the package (the brief's self-check; the sandbox has it) and uploads
  only if it passes; without `axonry-check` it uploads nothing. The check's output is in `logs/deliver.log`.
- One upload runs at a time (`upload.lock`), and none starts with less than 8 minutes left: an upload still in
  progress at the deadline does not count and may spoil the one before it.
- `python submission/derive/chain.py ax upload [<edit>]`: without a name it uploads `final2.txt`, else `final.txt`.
  The upload runs detached (a command stopped by the shell does not cut it short) and the command waits up to
  8 minutes for its end, so a command timeout of at least 10 minutes shows its result. If it returns before the end,
  `status` shows `uploading:` until the upload has ended.

**At the end.** The chain's last upload starts with at least 40 minutes left. After `DONE`, HOLDOUT of the uploaded
edit (report only) follows when at least 55 minutes are left (the `DONE` line says so); its status line
`HOLDOUT (report only)` comes before 32 minutes are left. Once the chain says `DONE` (and, when the `DONE` line says
that HOLDOUT follows, that HOLDOUT line), or 30 minutes before the deadline if it has not:
1. Finish REPORT.md (section 10).
2. If `status` does not show `DONE`: `python submission/derive/chain.py ax stop` waits for an upload in progress to
   end, then stops the chain. With fewer than 30 minutes left, `start` starts no chain again (it prints the upload
   choice).
3. Upload the chain's choice (`final2.txt`, else `final.txt`) with the report, no later than 20 minutes before the
   deadline. If the check above shows the package of `python -m wsp.upload` failing `axonry-check`, upload with
   `python submission/derive/chain.py ax upload` (`deliver.py` above: the same three files and manifest, with the
   `values` arrays included, after the `axonry-check` run described there). An upload still in progress at the
   deadline does not count.
4. If the upload passed (`status` shows no `uploading:` once it has ended) and nothing is left worth trying, run
   `axonry-done`.

## 6. Layer 1's steps (run by the chain)
1. `mkdata.py ids --per-source 300`, then `shard_gen.py --model base --n 4 --max-tokens 16384` (the pool, all GPUs);
   `evaluate.py ax base dev --n 2` (the starting model's DEV run, which pairs every comparison); 200 further ifsynth
   problems as `ifcheck`.
2. `exits.py --per-source 300 --n 2 --sources gsm8k math mmlupro`: the exits (section 2).
3. `mkdata.py data --extra exits.jsonl --answer-only-sources gsm8k math mmlupro --loop-ratio 0`: the records.
4. For seed s = 0, 1, ... while the time allows, one job at a time on all GPUs:
   ```
   torchrun --nproc_per_node 4 submission/derive/train.py --case ax --data train.jsonl --name S<s> --seed <s> \
     --alphas 1,1.5,2 --lr 5e-5 --epochs 3 --batch 64 --ema 0.95 --time-limit 20
   python submission/derive/evaluate.py ax S<s>_a1.5 dev --n 2
   ```
   each followed by the rule of section 2 (`rank.py`; the table is in `logs/select.log`).
5. FINAL (`rank.py --final`, `final.txt`), uploaded at once; `python -m wsp.case freeze ax <FINAL>`.

**Timing in the two-layer run.** Layer 1 is timed against the deadline minus 250 minutes, the time layer 2 needs
on its commit path (section 9):
- the first seed runs if it ends no more than 20 minutes past that point; a further seed only if it ends before it
  (with one GOOD seed, 39 minutes before it), so normally one seed is trained;
- the IF check runs only with two GOOD seeds (with one finalist it cannot change FINAL); HOLDOUT waits until after
  layer 2.

**One-layer fallback.** If layer 1's FINAL is neither GOOD nor SAFE (or there is none), or fewer than 190 minutes
are left when layer 2 would start, layer 2 is skipped for good: the chain goes on as layer 1 alone against the real
deadline (more seeds while time allows, the IF check of the two best GOOD seeds, HOLDOUT for the report), as the
one-layer exit method, and `DONE` follows its FINAL.

## 7. Layer 2's steps (run by the chain, `derive/stage2.py`)
**Parent.** Layer 1's FINAL exactly as uploaded: its uploaded package's `edit/` (checked against the package's
manifest sha256) goes to `stage1/edit`, with the published parent under `stage1/edit/parent`. From then on
(`stage1.txt`) layer 1 uploads nothing more. The case:
```
python submission/derive/wu/gen_traces.py init --case wrapup --parent stage1/edit/edit.json --dev 320 --eval 128 --seed 17
```
creates the case `wrapup` (practice gsm8k and math) on that parent, with `eval128` (the paired choice) and
`traces` (what the units are fitted on) disjoint. Every problem of layer 1's pool, DEV and HOLDOUT is then removed
from its sets, and `init` runs again to rewrite the trace set.

Steps (all GPUs unless stated; every script prints its table and rule, kept for the report):

**a. Traces.** `gen_traces.py run --model base --set traces --n 2`: layer 1 on the trace set, the profile's
sampling, per-problem seeds (so every later run pairs with it). Its measured time rescales every later step's time.

**b. Capture.** `capture.py run`: each trace teacher-forced through layer 1; at every 16th position (32 below 40 GB
of free disk) the MLP input of every layer but the last, labelled with the generated index, the phase (prompt,
thinking, `</think>`, answer), a loop flag and a newline flag.

**a'.** `gen_traces.py run --model base --set eval128 --n 4` on GPUs 1-3, during c-e.

**c. The read and the layer, per unit type** (`fit_gate.py --unit late_nl`, then `--unit late`; CPU). Trace problems
are split by problem into a fit and a tune half. Per layer: `d` = `wsp.gate.lda_direction` on the fit half
(positives: `late` from 0.85 x `max_tokens` on, `late_nl` newline positions from 0.6 x `max_tokens` on; negatives:
early thinking up to 0.45 x `max_tokens`, answers, and for `late_nl` every non-newline thinking position).
`theta` on the tune half from the early false-fire rate on traces that close on their own: 1e-4 per position
(primary) and 3e-5. A layer is eligible in the upper half of the network with at least 4 decoder layers after it,
an answer fire rate at most 1e-2 per answer position (absolute: a late read fires a little in answers wherever it
reaches stuck traces; whether that is harmless is measured in step e) and a prompt fire rate at most 2x its median
over layers. The highest `stuck_cov` wins (ties within 0.05: lower answer fire, then higher AUC).
A type with no eligible layer (exit 1) is left out; with neither type, layer 2 stops and layer 1 stays.

**d. The write and its scale** (`write_vec.py --unit U`, GPU 0, per type with a layer). The neuron: the least-used
of the layer (smallest mean |down_proj input| on fit traces), not one layer 1 or another unit writes. `w_hat`: the
mean unit-normalised gradient of log P(`</think>`) at and shortly before natural closes. The scale: the first norm
of an always-on write of c x `w_hat` (generated positions only) at which mean P(`</think>`) at thinking newlines
reaches 0.45, so a dose means the same at every layer. Control: a random orthogonal write at the same norm.

**e. The edit, its checks** (`build_unit.py`, GPU 0). All units in one spec, at doses 1, 1.5, 2, 3, 4 of the
calibrated write, checked teacher-forced, exactly as built, on every tune trace: fire rates per region, P(`</think>`)
at fired newlines, `hazard10` (closing within the first 10 fired newlines of a stuck trace), the silence check (real
tokens before the first fire unchanged) and `dlogp_answer`. Drop a dose if a unit's early fire exceeds 3x its rate,
its answer or prompt fire exceeds 2x step c's value plus the rate, or `dlogp_answer` < -0.01 nats/token. Per rate,
the smallest dose with `hazard10` >= 0.5 and the next one up; the first two are the candidates, admitted against
rules.json's limits. Only the first (WU) is generated. A random-write control is checked too.

**g1-g5. Commit stage**, on WU (only if its projected end leaves 40 minutes, see section 9):
- g1 `commit_unit.py gen --wu WU`: every trace in which a wrap-up unit fires is continued with WU from just before
  that first fire, 2 samples, to `</think>` plus 256 answer tokens (a vLLM logits processor, `presence_lp.py`,
  applies the profile's presence penalty to the prefix's tokens as in one run);
- g2 `commit_unit.py capture`: those closes and layer 1's natural closes teacher-forced through WU (first answer
  positions, close neighbourhoods, re-closes, starters, answer newlines, random positions; gradients at late ones);
- g3 `commit_unit.py fit` (CPU): BF's two reads per layer (gate: weighted LDA of every first answer position against
  all others, switch point 0.6; up: ridge logistic regression of late (>= 0.45 x `max_tokens`) against early
  (< 0.35 x `max_tokens`) first answer positions, switch point 0.4; gain 8). Eligible: upper half, 4 layers after,
  >= 0.5 a at >= 25% of late first positions; ranked by (1 - lateness AUC) + early share. BX: starter tokens of the
  answer against every other position, none of the thinking starters firing;
- g4 `commit_unit.py write` (GPU 0): free neurons, writes from mean gradients, a scale ladder per ranked layer,
  scored `share - 2 x (early P rise) - 100 x |dlogp elsewhere|` within loose sanity bars; up to 2 candidates;
- g5 `commit_unit.py build --wu WU` (GPU 0): doses 1 and 1.25 checked as built (BF must act as its reads: early
  first positions by a one-sided binomial test, elsewhere 3x its rate); specs `<WU>_C<layer>x<dose>` = WU + BF + BX.
Any of g1-g5 failing or finding nothing: wrap-up only.

**f. Paired choice.** `wu_eval.py run --n 4 --names WU` (GPUs 1-3 during g3-g5, or all GPUs on the wrap-up-only
path), paired with layer 1's eval128 run. `wu_eval.py decide --names WU --no-commit`: WU is eligible when its gain
is above 0 and no source is below layer 1.

**g6.** `wu_eval.py run --commit --wu WU --names <best commit candidate> --n 4`, then `wu_eval.py decide --names WU`:
the commit candidate is kept when its gain over WU (same seeds) is > 0 and no source is below WU by more than 1 SE.
`decide` freezes X (the kept commit candidate, else WU) and writes the case's NOTES.md (every step's tables).
g6 runs only when WU is eligible; a WU that is not is frozen with `decide --accept-negative` and goes on through
rule 2b, wrap-up only.

Fixed rules of this method (the chain applies them): a type without an eligible layer is left out (the unit is not
redefined and `--ans-bar` is not changed); only the first wrap-up candidate and the best commit candidate are generated; the random-write
controls are not generated; nothing in `derive/wu/` uploads. The admission check (`axonry-check`, where it is
installed) runs once, on the flattened edit, in `deliver.py` before it is uploaded (a per-candidate check in
`derive/wu/` would measure layer 2 against the published parent, while its parent is layer 1); layer 2's own
admission applies the rules.json limits to every candidate.

## 8. Decision rule and fallbacks
The stack STK (layer 1's ops + X's ops, one spec on the published parent) replaces layer 1's FINAL only if all four
hold (`logs/s2_rule.log` keeps every number):
1. `decide` froze X through its eligible branch (gain > 0, no source below layer 1); a commit unit only if the g6
   rule kept it. A wrap-up edit that is not eligible is frozen as it stands (`decide --accept-negative`, used here as
   a fixed rule of this method, not as the script's last resort near the deadline), wrap-up only, and can go on only
   through rule 2b.
2. X against layer 1 on eval128 x 4: gain >= max(2.0 points, 1.5 SE), and every source >= -1 SE (its own SE,
   clustered by problem). Or rule 2b instead:
   - **2b. Not worse, instead of a gain.** eval128 is layer 2's practice choice set (practice gsm8k and math, section
     7). A layer-2 unit acts only on traces still thinking late, past its threshold (section 3 and step c), mostly
     ones that would run on to the limit, so its gain on eval128 is at most the share of eval128 answers cut at the
     limit (measured in step a') times the share a close rescues; in the author's practice runs the paired SE on
     eval128 was 1.5 to 2 points, about that size. A rule that requires a clear gain therefore refuses, by chance,
     many units whose real gain is smaller than 1.5 SE. Below its threshold a unit changes almost nothing: it replaces
     the least-used neuron of its layer, and step e's silence check measures the real tokens before its first fire.
     So an X (wrap-up or commit) that misses rule 2's gain is kept if it is not worse on eval128 (gain >= -max(1
     point, 1 SE), every source >= -max(2 points, 2 SE)) and then passes the guard (rule 3) and rule 4 like any other.
3. Guard: STK on the DEV problems of `ax` for mmlupro and ifsynth, 2 samples, paired by (problem, sample) with
   layer 1's own DEV run (same seeds): both changes >= -max(4 points, 3 SE), with 4 points the IF check's floor in
   `rank.py --final` and SE the change's paired SE, clustered by problem. 64 problems x 2 samples per source gave an
   SE of about 4 to 5 points in the author's practice runs, so a fixed floor, or one at 2 SE (crossed by chance about
   once in 40 sources), would refuse a harmless stack too often. 3 SE (about 12 to 15 points at that SE) keeps a false
   refusal under 1% per stack; the guard then catches large losses on these two kinds of problem, such as a unit that
   fires where it was not fitted (the 4-point floor binds only at a smaller SE). Layer 2 is fitted on gsm8k and math
   only; this checks that it does not cost the other two kinds of problem.
4. `stack.py` passes (only set ops, no shared neuron, payload within rules.json, every tensor bit-identical to the
   chained build, relative change within the limits), `axonry-check` passes inside `deliver.py` (where it is
   installed, section 5), and the upload starts with at least 40 minutes left (so it ends more than 30 minutes
   before the deadline). The chain holds the upload lock from this time check through `final2.txt` and the upload,
   so a `stop` (which waits for the lock) does not fall between them.

Then `final2.txt` = STK and it is uploaded. Otherwise `final2.txt` = layer 1's FINAL, uploaded first and kept:
after a crash or timeout of any layer-2 step, too few stuck traces (the minimums in
`derive/wu/` stop it), no wrap-up edit frozen, rule 2 and rule 2b both missed, the guard or rule 4 failing, or no
complete run. If an STK
upload fails after its files began to go up, FINAL is delivered again at once (with at least 15 minutes left).

## 9. Time plan (6 h, 4 GPUs, E = elapsed)
Layer 1's times are those of a full rehearsal of layer 1 on 4 RTX 4090 (pool 52 min, the starting model's DEV 11,
exits 2, seed 0's training 7 and DEV 9); layer 2's are the nominal step times of `derive/stage2.py` (scale 1).
| E | step | GPUs |
|---|---|---|
| 0:00-0:10 | step 0, `chain.py ax start` | - |
| 0:10-1:15 | layer 1: pool (52), starting model's DEV (11), exits (2), records | 4 |
| 1:15-1:31 | seed 0: train (7), DEV (9), rule: FINAL at about 1:21-1:31 (by how long step 0 took) | 4 |
| 1:31-1:38 | FINAL uploaded (background, CPU) and frozen; stage1/, layer 2's case and sets (about 262 min left) | CPU |
| 1:38-2:25 | a (traces x 2, 35), b (capture, 12) | 4 |
| 2:25-3:21 | c (2 x 8), d (2 x 10), e (20) on GPU 0 / CPU, alongside a' (eval128 x 4, 40) on GPUs 1-3 | 1 + 3 |
| 3:21-3:43 | g1 (12), g2 (10) | 4 |
| 3:43-4:25 | g3-g5 (5 + 25 + 12 = 42) on GPU 0, alongside f (WU on eval128 x 4, 38) on GPUs 1-3 | 1 + 3 |
| 4:25-5:00 | g6 (WU + commit on eval128 x 4, 28), then the guard (7) | 4 |
| 5:00-5:08 | decide, stack, checks, upload (8) | CPU |
| 5:08-5:40 | REPORT.md, the final upload; then `axonry-done` if nothing is left worth trying | - |

Gates use the real minutes left, with step times rescaled from step a's measured time (at least 1.0 when step a
resumed on kept outputs after a restart): layer 2 starts only with at least 190 minutes left (the floor of its
wrap-up-only path); every step's time limit leaves the time the steps after it need (f's, the time of the guard and
the upload only: a slow f costs g6, not layer 2). After step e the commit path needs g1 + g2 + max(g3-g5, f) + g6 +
guard + upload = 12 + 10 + 42 + 28 + 7 + 8 = 107 minutes plus the 40 kept at the end, 147 in all. At the nominal
times above, step e ends at about 3:21 with about 159 minutes left, so **the commit path runs**. It runs as long as
step a's scale stays below about 1.05 (layer 2 starting at 1:38; about 1.1 when it starts at 1:28). A slower step a
takes the wrap-up-only path: f on 4 GPUs (28), the guard and the upload, done at about 4:05 at nominal times (about
4:35 at scale 1.2). Since layer 1 cuts fewer answers, its generations are shorter than the nominal ones, which were
measured on the published parent. After `DONE`, HOLDOUT of the uploaded edit (`holdout256`, 2 samples, frozen first)
runs when at least 55 minutes are left, for the report only, and ends with at least 32 left: normally after the
wrap-up-only path, not after the commit path (`DONE` at about 5:08, 52 minutes left).

## 10. REPORT.md
Write `/workspace/work/REPORT.md` while the chain runs (to a temporary file, then `mv` it into place: the chain
uploads it). It covers both layers:
- the failure targeted: the starting model's DEV (cut share, loops, accuracy of finished answers) and layer 1's
  trace set (step a: cut share, accuracy when finished);
- layer 1: records and exit accuracy per source (`logs/exits.log`, `logs/data.log`), units per layer
  (`logs/train_S0.log`), the seed table and rule (`logs/select.log`, `logs/final.log`);
- layer 2: the tables of steps c-g and the decision (`cases/wrapup/NOTES.md`, written by `decide`), the path taken
  (status lines `L2 path`), the four rules with their numbers (`logs/s2_rule.log`), and the flattened edit's checks;
- what was dropped and why: the second wrap-up candidate, the commit runner-up and the generated random-write
  controls (time); the commit stage if the gate or a step stopped it; HOLDOUT before the decision;
- HOLDOUT, if it ran;
- the controls (section 11) and what did not work.
Until REPORT.md holds a report of its own, the chain's uploads carry an automatic one built from these files.

## 11. Controls
- Layer 1: the seeds or strengths that do not pass the rule, next to FINAL; the starting model's paired DEV run.
- Layer 2: random-direction writes of the same norms with the same gates, teacher-forced in steps d, e and g4-g5
  (P(`</think>`) and the real tokens' log-prob must stay at layer 1's level); the silence check before the first
  fire; the always-on write is used only for calibration; `nl_share` and P(`</think>`) at fired newlines show
  whether the units fire where they can act; the commit stage is kept only if it beats WU itself on eval128.
- The upload: the guard on mmlupro and ifsynth, and the flattened edit rebuilt and compared tensor by tensor with
  the chain that was measured.
