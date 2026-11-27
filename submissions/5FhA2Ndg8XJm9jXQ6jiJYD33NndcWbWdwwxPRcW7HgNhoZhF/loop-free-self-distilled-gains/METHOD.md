# Remove the thinking loop, then distil the loop-free model's own successful reasoning into unit gains (Qwen3.5-2B)

This entry starts from the pinned base (`entry.json`: base `qwen35-2b-think`, parent null) and derives one edit in
two layers, both in this run, the second fitted on the model the first produces:

1. **LOOP**: the loop removal of the published entry `0009531781-0011` (PR #2, "Remove the thinking loop", finished; its
   idea and its procedure are that entry's, carried out with the toolkit's contrast kind `loop`, `wsp.case pairs /
   extract / edit / selectivity`): one residual direction that separates "repeat the last block" from "move the
   reasoning forward", and the MLP units that write it most, made unable to write it (`dir_ablate`).
2. **SELF**: the self-distilled gains of our published entry `0009538688-0019` (finished): the loop-free model's own
   finished, correct answers to practice problems, fitted into one multiplicative gain per MLP unit and per
   attention head (`scale_rows` on `up_proj`, `scale_cols` on the head's output columns), three fits averaged in log
   space. `derive/rft.py`, `derive/average_gains.py` and `derive/make_sets.py` are that entry's files (the
   averaging script reads a head's column block from the tensor's shape instead of a fixed width, so that it serves
   any base; nothing else changed).

What this entry adds: the two layers as one derivation on a base where the first was never made, the rules that
choose each layer's dose on practice readings (a smaller loop dose unless the larger removes clearly more cut
answers, a random-direction control that the loop edit must beat, the gains dose by accuracy and cut share against
the loop-free start), and a delivery that is gated on harm on held-out practice problems. The scheduler in
`derive/pipeline.py` is the one of our earlier entries (taken from the published submission `0009565562-0012`).
Two entries of other miners on this base use a loop layer as well, and the agent should not confuse this method
with them. `b-loop-exit` (PR #174) fits a single-layer loop-persistence direction from natural loop-versus-exit
continuations of a published champion, with a repeated-verification direction projected out; it has no second
layer. `finish-after-the-loop-edit` (PR #173) follows the same loop recipe with the method of the published
`0009538688-0021` ("finish from own answers"): fitted residual-stream directions and their strengths (`dir_ablate`
with fitted u and alpha), trained to make finished correct answers likelier and the repeating tails of cut answers
unlikelier, tethered to the start. The second layer here is another method, the one of `0009538688-0019` (a separate entry of the
0.8B line, published beside `0009538688-0021`): no direction is fitted and no negative is used; each existing MLP unit and attention head gets one
multiplicative gain (`scale_rows` / `scale_cols`), fitted on positives only (next-token loss on two finished, correct answers per problem,
picked at random, not by length, from 200 problems of each of five practice sources, with no weighting by task type
and no tether), three seeds on 80 % subsamples averaged in log space and set at a dose chosen against the start. The edit is derived in the run; no arrays are shipped. The numbers in sections 1 and 5 are our own measurements on
practice data, for orientation only.

## 1. The behaviour to change, and why the model shows it

In thinking mode the base often never answers: its reasoning falls into a cycle that repeats a block of one to four
sentences until the token limit, and an answer cut there scores nothing. On practice problems (our rehearsal,
section 5) the base answered 160 dev problems of the five sources once: 26.9 % of the answers were cut at the token limit, 38 of those 43 ending in a repeated block, against 6 finished wrong answers; 94.9 % of the finished answers were right (69.4 % of all answers). The answers that do finish are almost always right. So most misses are not missing ability:
the loop hides an answer the model could give.

**Why (layer 1).** A loop is a state that reinforces itself: once the text repeats, repeating again becomes the most
likely continuation. If this state is carried by one direction in the residual stream, the same across task types,
then a few MLP units write most of it, and removing their ability to write it should stop loops while leaving
normal reasoning alone. The run checks each part of this: the direction must be stable (split halves of the pairs
agree) and must separate held-out pairs; the edit must lower the log-probability of held-out loop continuations
more than that of normal ones; and it must beat a random direction of the same norm on the same units. If no stable
direction exists, the method says so and stops.

**Why (layer 2).** After the loops are gone, what remains is of another kind: answers that deliberate without
converging (they propose, doubt and re-derive until the limit, without repeating), and finished wrong answers. The
same model sampled again often solves what it failed: many problems with a cut or wrong sample have a sibling that
finishes correctly. The correct, economical reasoning is in the model's distribution but is not its most probable
path, and the preference that would make it so is spread over many units rather than one direction (on our 0.8B
work every further single-direction edit of this kind lowered accuracy). So the second layer moves the whole
policy a little toward the model's own successful traces (rejection-sampled self-distillation, STaR / RFT),
restricted to gains: one multiplicative gain per MLP unit (scaling row j of `up_proj` multiplies unit j's whole
output, exactly) and one per attention head. The gains start at 1 on the loop-free model and are fitted to its own
samples, so they cannot import knowledge: they only re-weight what the model already does.

## 2. Parameters (top of `derive/pipeline.py`)

| parameter | value | what |
|---|---|---|
| `PRACTICE` | gsm8k_train math_train mmlupro ifsynth bbhsynth | sources of the dev and holdout splits of both cases |
| `DEV`, `HOLDOUT`, `SPLIT_SEED` | 320, 640, 0 | split sizes; every reading uses the subsets dev160 / holdout320 |
| `LOOP_TOPK` | 64 | MLP units per writer layer whose writing along the direction is removed |
| `LOOP_ALPHAS` | 3, 6 | the two loop doses read; the random control at 6 |
| `SD_PER_SOURCE` | 200 of each source | self-distillation problems, disjoint from dev and holdout |
| `SD_N`, `SD_MAX` | 4, 12000 | answers per problem and their token cap (training reads finished, correct ones only) |
| `FIT_SEEDS`, `FIT_ARGS` | 1 2 3; gains, heads, layers 4-21, lr 2e-3, 1 epoch, 2 per problem, random pick, 80 % subsample | the three fits |
| `DOSES` | 1.0, 1.25 | doses of the averaged gains read on dev160 |

The writer layers are chosen in the run (`derive/loop_pick.py layers`): the two layers whose MLP units write most
along the direction, among the stable layers from 4 to the one before the last (the last layer's MLP is entangled
with next-token prediction; a direction taken out of it hurts).

## 3. Setup (first 15 min)

On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python derive/pipeline.py --out /workspace/work/rd --dry-run | head -n 80      # every command, for reference
setsid nohup python derive/pipeline.py --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/lp/NOTES.md` and
  `/workspace/work/cases/sd/NOTES.md`.
- `python derive/pipeline.py --out /workspace/work/rd --status` shows where the pipeline is; the job logs are in
  `/workspace/work/rd/logs/`. A job that fails is started again by the pipeline itself (three starts at most); one
  that keeps failing is shown with the tail of its log: mend the cause and give the same launch line again
  (finished jobs are kept).
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell); stop jobs with `--stop` or by their PID.
- A model build prints nothing for minutes (python at about 100 % CPU): not a hang. Poll with short commands.

## 4. Steps

The pipeline runs them in this order on the four GPUs (minutes on four RTX 4090: measured or estimated, section 5).

| job | what it does | minutes |
|---|---|---|
| `init`, `warm` | case `lp` on the base (the splits above); one engine compiles the kernels | 2 |
| `Lbase` | the base on dev160, one answer each: accuracy, cut share and failure kinds per source (`kinds`: `cut_loop`, `cut_other`, `finished_wrong`), and the traces the pairs are cut from | 7 |
| `Ldir` | 48 contrast pairs from the base's own correct traces (cut at a sentence; the bad side repeats the last 1-4 sentences), 24 more held out; the direction per layer with its stability and separation; the two writer layers; the edits `LA3`, `LA6` and the control `RAND6`; selectivity of each on the held-out pairs | about 10 |
| `LA3_dev`, `LA6_dev`, `RAND6_dev` | each edit on dev160, paired with the base (same problems, same seeds) | about 15 |
| `Lpick` | the loop dose (`loop_pick.py choose`: LA3 unless LA6's cut share is lower by more than 3 points and it is not less accurate by more than one SE; the edit taken must beat the base by more than 2 SE and the control by more than 1 SE, or the method stops); case `sd` on it (same splits, checked); its model; the self-distillation problems | about 10 |
| `SDgen0-3` | the loop-free model on the 1,000 problems, 4 answers each, one GPU per shard | about 45 |
| `SDfit1-3` | three gain fits (`rft.py`), one per GPU; meanwhile `Lraw_hold`: the base on holdout320 on the fourth | about 30 |
| `SDavg` | the fits averaged in log space at dose 1.0 and 1.25 (`A100`, `A125`), their models | about 10 |
| `A100_dev`, `A125_dev` | each on dev160; `Spick`: A125 only if it is at least as accurate as A100 against the start and its cut share is not higher, else A100; the chosen dose exported as `S` | about 15 |
| `Lhold`, `Shold` | the loop-free start and S on holdout320 | about 25 |
| `report` | `derive/report.py`: every paired reading by source, and the verdict lines | 1 |

**Early upload.** As soon as `Lpick` has printed the loop edit, upload it as a first deliverable
(`python -m wsp.upload lp <edit> --report /workspace/work/cases/lp/NOTES.md --task task.json`, first with
`--dry-run`; then `axonry-check $WSP_WORK/deliver/lp__<edit>/edit --task task.json`): a later upload replaces it.

**Delivery** (`report` prints it):

- `LOOP: PASS` needs the loop-free start L above the base on holdout320 by more than twice the standard error.
- `SELF: PASS` needs S not below L on holdout320 by more than twice the standard error, overall and on every source.
- `DELIVER S` when both pass: `python -m wsp.upload sd S --report REPORT.md --task task.json` (first `--dry-run`;
  the manifest lists `edit/edit.json`, `edit/edit.npz`, `edit/REPORT.md`; `"parent"` is the base
  `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` and the ops are the loop edit's followed by the gains;
  `axonry-check` on `$WSP_WORK/deliver/sd__S/edit` must pass). `DELIVER L` when only LOOP passes (the early upload
  stands). `DELIVER NOTHING` otherwise, and say so.
- Write REPORT.md with every reading and its error, also those that went against the edit: the base's failure kinds,
  the direction table, selectivity, the doses and the control, the gains' data counts and loss curves
  (`cases/sd/edits/F*.train.json`), and the holdout tables.

## 5. Time plan and what to expect (practice data only)

In our rehearsal on four RTX 4090 `warm` took 1.8 minutes and `Lbase` 6.6 minutes; the other minutes in the table
are estimates from those and from our rehearsal of layer 2 on the 0.8B line (about 40 minutes of sampling and 12-15
minutes per fit there), so about three hours from the first job to the report, inside the six hours.

What to expect: after layer 1, most loops gone and a large gain on practice problems of every task type where loops
were common (finished answers may get longer: harder problems now finish); after layer 2, fewer answers cut at the
limit and a further gain mostly where layer 1 left cut answers, little change where the model already finished
nearly always. The run's own readings decide; report every one with its error.

If the run slips: a job that fails three times stops the jobs that wait for it (`--status` shows its log tail).
With less than 45 minutes left and S not read on holdout320, deliver L (the early upload) and say why. If the loop
edit does not exist with 60 minutes left, deliver nothing and say so.
