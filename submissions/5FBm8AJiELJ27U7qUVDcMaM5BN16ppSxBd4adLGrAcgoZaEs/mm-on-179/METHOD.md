# Two focused gain layers on the workings-and-requests model: its own better maths and multiple-choice answers, with its puzzle and instruction layers held

## The problem
A model that answers a problem several times often gets it right in some answers and not in others. On maths the
usual loss is an answer that is still thinking when the token budget ends, on a problem the same model finishes in
another answer. On multiple choice it is a wrong option on a question the same model answers rightly in another
answer. In both cases no new knowledge is missing: the model's own better answer exists, and the model does not give
it reliably.

The start of this entry (`entry.json`: parent `0009599139-0015`) is a model whose chain has removed reasoning loops,
been fitted on its own finished answers, been taught to write short workings on puzzles, and been given answer
preferences on instruction requests. None of those layers was fitted on maths alone or on multiple choice alone, and
the last two are the newest and the easiest to disturb: a fit on other answers can undo them without anyone looking.

The run measures the problem before anything is fitted, on practice problems (never the benchmarks): a pool of 384
maths problems with two answers each, and a pool of 320 multiple-choice problems, 160 of each family the toolkit
ships (`mmlupro`, graduate-level questions with ten options; `mmlucf`, exam questions with four options), with three
answers each, all with the profile's sampling and the full token budget. `DIAG.md` gives, per family:

- accuracy, the share of answers cut at the limit, and accuracy when an answer is finished;
- the share of problems the start answers right in every answer, right in some and wrong in others, and wrong in
  every answer.

The middle group is what this method can move. Our hypothesis: fitting the start on its own finished right answers
of one task type moves it toward them; fitted for that one type, with the start's behaviour on every other type
held, the fit cannot pay for its own type with another; and with the puzzle and instruction families held on many
problems each, the parent's own two layers stay what they are. The run checks each part: each fit's own counts, an
edit's op list before anything is generated with it, and paired answers on problems no fit ever saw.

If the fits come out with too few right answers to learn from, or no edit passes on dev, the method has nothing to
deliver on this start: the chain says so and stops. Report that, and end the run.

## The edit: two layers of one lever
One gain for each MLP unit and each attention head of the output projections of **layers 4 to 21** (ops `scale_cols`
on `mlp.down_proj`, `linear_attn.out_proj` and `self_attn.o_proj`), fitted twice by the toolkit's own fitter
(`python -m wsp.fit run --objective sft --params gains`):

- **Ma**, the maths layer: fitted on the start's finished right maths answers, at most two a problem, with word
  problems, both multiple-choice families, both instruction families, the puzzle family and `musr` held;
- **Mc**, the multiple-choice layer: fitted on the start's finished right multiple-choice answers, at most two a
  problem, with maths, word problems, both instruction families, the puzzle family and `musr` held.

Each layer is three fits with seeds 1, 2 and 3, each on a random 85% of the problems. "Held" means: on the start's
own finished right answers to that family, the fit pays for any change of the start's predictions (a KL term over
the whole vocabulary, the fitter's anchors). `python -m wsp.fit spec` averages the three fits of a layer in log
space. The two layers are then joined into one edit, Ma's ops first and Mc's after them; gains of the two layers on
the same unit multiply. Two candidates are written: `MM`, both layers at full dose, and `MMH`, both at half dose.
One of them is delivered, by rules fixed here in advance.

What the edit does not do: it fits and rewrites no unit, removes or adds no direction, writes no matrix of layers 0
to 3 or 22 and 23 and no MLP input matrix, and is fitted on no answer to word problems, puzzles or instruction
requests.

## Credits
- **The start** is `0009599139-0015`, PR #179 "workings-and-requests-on-173", a finished entry
  (`submissions/5FeeiHS6sVCNF6vAsR3QBE7KHwkDqjgLzy7jejbzwweJbXMu/workings-and-requests-on-173/`). It is PR #173
  "finish-after-the-loop-edit" (`0009598520-0002`) with two layers on top, each credited by #179 to its source:
  the puzzle layer of written workings of PR #156 "workings-and-maths-recall-on-130" (`0009580621-0022`), in the
  published edit 960 `dir_ablate` ops on the MLP gate, up and down matrices of layers 0 to 20; and the instruction
  stage of PR #168 "workings-and-requests-on-164" (`0009588957-0032`), `scale_rows` on the MLP up matrices and
  `scale_cols` on the attention output matrices. The run starts from #179's published edit. Nothing of it is
  derived or delivered again: the workings are #156's, the preferences #168's, their combination #179's.
- **The layer** is PR #180 "focused-finishing-on-173", a finished entry (submission `0009600331-0008`,
  `submissions/5F6Rnqd5fsjmA4inMnpUZAN9fWHNMLKutXMQDd92UuUBGNeL/focused-finishing-on-173/`), re-run here in full on
  another parent. **The same as #180:** the idea of two focused layers, one on the start's own right maths answers
  and one on its own right multiple-choice answers, each with every other task type held by the fitter's anchors;
  the pools (384 maths problems with two answers, 320 multiple-choice problems with three); the fitter's settings
  (`--layers 4-21 --max-len 16000 --frac 0.85`, seeds 1, 2 and 3); the average by `wsp.fit spec`; the composition
  in the order maths then multiple choice; the half dose of both layers as the fallback; and the bar of 60 examples
  of a fit's own type. Its `derive/join.py` is adapted into `derive/run.py`, with credit there. Its
  `derive/choose.py` is not used. The self-distilled gains themselves are, as #180 credits them, the method of
  `0009538688-0019` and of its transfer to this base, `0009598598-0002`.
- **The frame.** `derive/run.py` and `derive/gen.py` are adapted from the `derive/run.py` and `derive/gen.py` of PR
  #178 "finish-the-thought" (submission `0009599014-0016`,
  `submissions/5FyCs7tdcqPLn8p5LJManbqzZes6JsrbNji2gfCgY5fkNFFP/finish-the-thought/`), through our own earlier
  bundle on that code; each file's header names its source and says what differs. The case and its splits by
  problem, the generation on all GPUs, the paired tables, the call of the toolkit's fitter on the start's own
  answers, the report, the packaging with the admission self-check and the two structural bars are #178's. Nothing
  of #178's units is used.
- **The toolkit** gives the prompts, seeds, sampling, grading, the fitter and its averaged spec, the shuffled
  control, building and uploading.
- **Ours, beside #180's method:**
  1. **The parent.** Both layers are fitted on a start that already writes workings on puzzles and carries
     instruction preferences, on that model's own answers, and are read against that model.
  2. **A guaranteed hold on what the parent has.** The anchors' pool holds 100 problems of each instruction family,
     of the puzzle family and of maths, 60 of each multiple-choice family, and 40 of word problems and of `musr`.
     The fitter's anchor count is one total, drawn at random over all held families together, so it is given as
     the number of problems of that pool: every held problem with a finished right answer is held.
     A fit is used only if its own counts show at least 60 held problems of each instruction family and of the
     puzzle family, no answer of another family fitted, and its own type never held. The counts are read per fit.
  3. **The check before generation.** Before anything is generated with an edit, `check` reads its op list (every
     op a gain of the fitter's three kinds of matrix in layers 4 to 21, nothing else) and prints payload and change
     against the limits.
  4. **The rule between the candidates.** The full composition first; the half dose is read only if the full one
     fails its check or the no-harm rule. A passing `MM` is delivered and is not replaced.
  5. **The no-harm rule.** Our wide rule in which every bar is a bar against harm, with two groups read for these
     two layers: maths and word problems together, and the two multiple-choice families together.
  6. **The splits, the early upload and the holdout read once.** Five splits by problem, a holdout among them. An
     edit is uploaded as soon as it passes on dev. The holdout is read once and stops a delivery only on clear
     harm.
  7. **The chain** with its time plan and restarts.

**Against the start's chain and the other entries of this line.** Each entry is described by what it does.
- PR #173 "finish-after-the-loop-edit": the toolkit's loop edit with a fit on the model's own finished answers of
  all task types on top. It is in the start's chain. This bundle adds no loop edit and no fit over all task types:
  each of its layers is fitted on one task type with every other type held.
- PR #179: the parent. A puzzle layer of written workings and an instruction layer of answer preferences, side by
  side on PR #173. This bundle starts from its model, fits no workings and no preference, runs no step of that
  entry, and holds its puzzle and instruction behaviour by anchors with a counted quota.
- PR #180: the source of the layer, as credited above. It fits the two layers on PR #173 and chooses among the
  composition and the single layers by its own rule. This bundle runs the two layers on PR #179's model and
  differs in the seven points listed; it delivers the composition or its half dose, never a single layer.
- PR #176 "loop-free-self-distilled-gains": the loop removal, then self-distilled gains fitted on the loop-free
  model's own right answers of all task types together. This bundle fits two focused layers, each on one task
  type, with the parent's own layers held; it fits nothing over all types.
- The entries on PR #173, or on the line's champion, that close the thinking near the end of the budget, alone or
  beside a layer of their own. This bundle closes nothing: it fits no answer preference, writes no unit and adds
  no closing direction:
  - PR #181 "wrap-up-units-on-the-champion": 96 MLP units in the last four layers rewritten to close the thinking
    late, fitted on the start's own answers and delivered alone;
  - PR #182 "finish-on-the-champion": self-distillation on the start's own right answers together with such
    units;
  - PR #183 "base-fitted-wrap-up-units-on-the-champion": the same kind of units fitted on the unedited base's
    answers and carried onto the start;
  - PR #184 "wrap-up-to-answer-mode": one rewritten MLP unit that pushes long thinking toward its end early enough
    for the answer to solve the problem again;
  - PR #185 "instruction-kinds-then-close-in-time" (submission `0009602170-0009`,
    `submissions/5En7mQG9i7PqQG5Uj8gUFqdei8fVCWubf6QtPNc8KzkJKJzF/instruction-kinds-then-close-in-time/`): on the
    champion, an answer-preference fit for instruction kinds the practice sets lack (the second stage of the 0.8B
    entry `0009591920-0008`, run on 2B in `0009598608-0007`), then PR #178's wrap-up units. The instruction layer
    of this bundle's start is PR #179's (credited above), not a new one;
  - PR #186 "finish-the-thought-layer-on-173" (submission `0009601430-0008`,
    `submissions/5DPd3evA4mvubpX2efsepKCVjVK76oLYx7TgaMHZupVH4FLV/finish-the-thought-layer-on-173/`): PR #178's
    whole layer, its self-distilled gains fitted on all task types together and its wrap-up units, derived on the
    unedited base and delivered onto PR #173. This bundle's gains are fitted on PR #179's own model, in two layers
    of one task type each with every other type held, and it carries no unit.

## Which model to start from
`entry.json` names `0009599139-0015` as the parent. The task lists it in `task.json` (`starts`), with its published
edit in the bundle as `0009599139-0015/edit/`. Copy that directory next to `task.json` in the sandbox.
`python submission/derive/run.py start` prints the start and records its id, the sha256 of its `edit.json` and a
digest of its whole chain; `run.py setup` builds the case from it. From then on every command compares the recorded
start with the task and with the case's own copy, and refuses to run if either has changed.

**This method is defined on that parent only.** If the task does not offer `0009599139-0015`, every command stops
and says so. Do not run the method on the champion or on the base in its place: the anchors, the quota and the dev
rule are written for a start that holds those puzzle and instruction layers. Report it, and end the run.

## Steps
All steps are commands of `derive/run.py` (`-h` lists them, `<command> -h` their options). Work in the sandbox under
`/workspace/work`, with `task.json`, the start's `0009599139-0015/edit/` and `submission/` there, and run every
command from that directory, or give `--task <path to task.json>`. Set `WSP_CASES` and `WSP_WORK` to directories
under the work directory. Run long jobs detached (`tmux`, or `nohup` with the output in a log file) and follow
`$WSP_WORK/mmgains/LOG.txt`, where every command writes what it prints. A command that stops with an error says why
on its own output (`chain.log` for the chain).

**Steps 1 to 8 as one job.** `nohup python submission/derive/run.py chain > chain.log 2>&1 &` runs them in the order
below with no wait between them. **As soon as an edit has passed on dev it uploads it**, with the report draft
(`report`, then `deliver`); a later upload replaces an earlier one. The half dose is checked and read only if `MM`
failed its check or the no-harm rule. Then come the control and the holdout. The half dose, the control and the
holdout are each started only if they can end before the run's last 40 minutes, and no step is started inside
them. If a check or an evaluation is started and leaves nothing behind, the chain stops with an error and no
`CHAIN DONE` line: mend the cause and start it again. Each step writes `chain: ...` lines with its time into
LOG.txt. The last line is `CHAIN DONE: deliver <name>: uploaded ...`, which names the edit and says what was
uploaded, or `CHAIN DONE: no edit
passed ...` with the reason. A step whose output exists is not run again, so after a stop start the chain again.
`chain --no-upload` packages and checks wherever it would upload, and uploads nothing: use it for rehearsals.

1. **Start, splits and starting model.** `python submission/derive/run.py setup` takes the start from the task,
   creates the toolkit case `mmgains`, builds the starting model, and splits the practice problems of every family
   the toolkit ships (`gsm8k`, `math`, `mmlupro`, `mmlucf`, `musr`, `bbhsynth`, `ifsynth`, `tuluif`), by problem:
   - `mafit`, the maths fit pool: 384 maths problems;
   - `mcfit`, the multiple-choice fit pool: 160 problems of each multiple-choice family (320);
   - `keep`, the anchors' pool: 100 problems of each instruction family, of the puzzle family and of maths, 60 of
     each multiple-choice family, 40 of word problems and of `musr` (600);
   - `dev`: 192 problems of maths, of each multiple-choice family, of each instruction family and of the puzzle
     family, 96 of word problems and of `musr` (1,344);
   - `holdout`: 150 a family (1,200).
   No problem is in two splits. The same five lists are kept in the toolkit case, so the fitter itself drops any
   answer on a dev or holdout problem.
2. **The start's own answers.** `run.py traces --split mafit --gpus 0,1,2,3` (two answers a problem), `run.py traces
   --split mcfit --gpus 0,1,2,3` (three) and `run.py traces --split keep --gpus 0,1,2,3` (two), with the profile's
   sampling and the full budget. Everything is fitted from these answers. Note how long the three take together:
   the time plan scales from it.
3. **Diagnose.** `run.py diagnose` writes `DIAG.md`: the measurements of "The problem" for each family of the two
   fit pools, and the same summary for the families of the anchors' pool.
4. **The fits and the two candidates.** `run.py gains --gpus 0,1,2,3` writes the answers where the toolkit's fitter
   reads them and runs six fits, one a GPU, side by side:
   ```
   python -m wsp.fit run mmgains Ma_s<seed> --runs base__mafit base__keep --objective sft --params gains \
       --layers 4-21 --max-len 16000 --frac 0.85 --seed <seed> \
       --anchor bbhsynth gsm8k ifsynth mmlucf mmlupro musr tuluif --anchor-n <problems of the anchors' pool>
   python -m wsp.fit run mmgains Mc_s<seed> --runs base__mcfit base__keep --objective sft --params gains \
       --layers 4-21 --max-len 16000 --frac 0.85 --seed <seed> \
       --anchor bbhsynth gsm8k ifsynth math musr tuluif --anchor-n <problems of the anchors' pool>   # seed = 1 2 3
   ```
   - These are #180's lines with two differences: `musr` is anchored too, because the anchors' pool here holds
     it, and the anchor count is given. The fitter fits every family of its runs that is not named under
     `--anchor`: so `Ma` also learns from the maths answers of the anchors' pool, and `Mc` from its multiple-choice
     answers. Nothing else is fitted.
   - Each fit reports its examples per family. `gains` stops, and no edit is made, unless every fit has at least 60
     examples of its own type (#180's bar), fits no answer of another family, holds no answer of its own type, and
     holds **at least 60 problems of each instruction family and of the puzzle family**.
   - Then `python -m wsp.fit spec mmgains Ma10 --fits Ma_s1 Ma_s2 Ma_s3 --dose 1` and the same for `Mc10` from
     `Mc_s1 Mc_s2 Mc_s3`: the three fits of a layer averaged in log space. `Ma05` and `Mc05` are the same with
     `--dose 0.5`.
   - Then the join, inside `run.py` (adapted from #180's `derive/join.py`): one edit with the ops of the maths layer
     first and those of the multiple-choice layer after them, on their common start. `MM` is `Ma10` then `Mc10`;
     `MMH` is `Ma05` then `Mc05`.
5. **The check, before anything is generated.** `run.py check --name MM`.
   - The op list: every op is a `scale_cols` on `mlp.down_proj`, `linear_attn.out_proj` or `self_attn.o_proj` of
     layers 4 to 21. Any other kind of op, any other matrix, or any other layer fails the check.
   - The limits, printed and recorded, each against the protocol's limit as the toolkit's `rules.json` gives it:
     the payload in bytes, the largest relative change of a matrix against the start, and the global relative
     change.
   - An edit that meets all of it has its model built. One that does not is recorded as failed and is not used:
     nothing is generated with it. `MMH` is checked only if `MM` fails its check or the no-harm rule.
6. **The starting model on dev.** `run.py traces --split dev --gpus 0,1,2,3` is the draw that every edit is compared
   with, on the same problems and seeds (two answers a problem; see Time for when the chain takes one). `evaluate`
   reads an edit with as many answers a problem as this draw has.
7. **The composition on real answers, the upload, then the control.**
   `run.py evaluate --name MM --split dev --no-control --gpus 0,1,2,3` generates the answers of `MM` on dev and pairs
   them with the start's, problem by problem and seed by seed. It prints every table and ends with `No harm on dev:
   PASS` or `FAIL (...)`; see "The no-harm rule".
   - **Upload at once.** As soon as an edit has passed on dev, run `report` and `deliver` for it (step 9). A later
     upload replaces it, so an early one is never wasted.
   - **The half dose, only after a failure.** If `MM` fails its check or says `FAIL` on dev, run `run.py check --name
     MMH` and `run.py evaluate --name MMH --split dev --no-control --gpus 0,1,2,3`, and on a `PASS` `report` and
     `deliver` for `MMH`. `check`, `evaluate` and `deliver` refuse `MMH` in any other case; see "The rule between the
     two candidates".
   - **Then the control.** `run.py control --name <chosen>` writes the edit `<chosen>_control`: each layer's
     averaged fits at the edit's dose with the toolkit's `spec --shuffle` (each matrix's gains put on other units
     or heads of that matrix), joined the same way. `run.py check --name <chosen>_control` checks it like any edit
     and builds its model. Then run `run.py evaluate --name <chosen> --split dev --gpus 0,1,2,3` again: it finds the
     answers already generated, adds the control's, and rewrites the table with the control line; the `PASS` or
     `FAIL` does not change. The control is recorded and reported, and it is not a condition. If it cannot be made
     or read, say so in the report and go on: the dev table and the upload stand.
8. **The holdout, once, when time is left.** `run.py evaluate --name <chosen> --split holdout --gpus 0,1,2,3`, for
   the one edit chosen on dev. The command first records that edit as the case's frozen edit, then generates the
   start's and the edit's answers on the holdout and prints the same table. It refuses an edit that has not passed
   on dev, and any second edit.
   - The table is reported whatever it shows. It is not a second condition.
   - It changes the delivery only on **clear harm**: the pooled result against the start is below −1 standard
     error, or the instruction families together are below −2 standard errors. An edit not yet uploaded is then
     not delivered. An edit already uploaded stays live with the report draft it was uploaded with: `deliver`
     refuses that edit from then on, so no final report goes up with it. Write the holdout table and this outcome
     into `REPORT.md` in the work directory, and end the run.
   - Choose nothing else on the holdout, and evaluate no second edit there.
9. **Deliver.** Only an edit whose check stands and whose dev table says `PASS`.
   - `run.py report --name <chosen>` assembles `$WSP_WORK/mmgains/REPORT.md` with every table measured. Add in your
     own words what was targeted, the start, the diagnosis, each fit's counts, which candidate was read and which
     rule decided, the control, the holdout table or why it was not read, and what did not work.
   - `run.py deliver --name <chosen> --dry-run` packages the edit by the upload protocol and runs `axonry-check` on
     the package; nothing is uploaded. `run.py deliver --name <chosen>` does the same and then uploads.
   - The chain has already uploaded the edit with the report draft. Deliver again once the report is complete: the
     last complete upload counts. (After clear harm on the holdout this is refused; see step 8.)
   - `deliver` refuses an edit without a dev `PASS`, `MMH` when `MM` has not failed, an edit whose holdout table
     says clear harm, tables measured on another version of the edit or of its control, an edit written again
     since its check, and a recorded start that no longer matches the task.
10. **End.** The method is complete when an edit has passed on dev and been uploaded with its report, or when the
    chain has ended with `no edit passed`. Then end the run with `axonry-done`. If no edit passed, nothing is
    uploaded.

## The no-harm rule
One rule decides on dev. The table prints every number, and `PASS` needs every condition:

| Condition | Bar |
|---|---|
| Pooled paired result against the start, all families | not below −1 standard error |
| `math` and `gsm8k` pooled together, against the start (the maths group) | not below −1 standard error; `math` must have been measured |
| `mmlupro` and `mmlucf` pooled together, against the start (the multiple-choice group) | not below −1 standard error; each must have been measured |
| `ifsynth` and `tuluif` pooled together, against the start | not below −2 standard errors; each must have been measured |
| Every single family against the start | fails only below −max(2 points, 3 standard errors); `bbhsynth` must have been measured |
| Answers that end inside the thinking (#178's bar) | at most 1 in 250 |
| Answers that close the thinking twice, start against edit | at most the start's share plus 1 point |
| Share of answers cut at the limit with the thinking open, start against edit, all families together | up by at most 1 point |
| The same share, family by family | in no family up by more than 3 points |
| On dev: pooled paired result against the shuffled control | recorded, not a condition |

**Every bar is a bar against harm.** A gain is not required: on a few hundred answers the run cannot tell a small
gain from noise, so the edit is uploaded unless it measurably harms. The bars are wide for the same reason; the
anchors, the counted quota and the check are the fine guard. Results are paired on the same problems and seeds, with
standard errors clustered by problem.

On the holdout the same table is printed and reported, without the control line. It stops a delivery only on clear
harm: the pooled result against the start below −1 standard error, or the instruction families together below −2
standard errors.

## The rule between the two candidates
Fixed in advance; the chain, `report` and `deliver` apply the same function, and `check` and `evaluate` refuse `MMH`
until it is due.

- `MM`, both layers at full dose, comes first. If it meets its check and passes the no-harm rule it is uploaded.
  Nothing replaces it: `MMH` is then neither checked nor read.
- `MMH`, both layers at half dose (#180's own fallback), is read only if `MM` fails its check or fails the no-harm
  rule on dev. It is delivered if it meets its check and passes the no-harm rule itself.
- If both fail, no edit passed. No single layer is delivered, and no third dose is made.

## Time
Generation takes the time. PR #180's METHOD gives these times for its own steps on four GPUs: about 75 minutes for
the start's answers (about 3,000 answers in all), about 20 minutes for six fits (one GPU each, four at a time),
about 30 minutes for three candidates on its dev split (448 problems, one answer each). **Planned from those and
from the numbers of problems here, not measured for this bundle:** step 2 (768 answers on `mafit`, 960 on `mcfit`,
1,200 on `keep`) about 75 minutes; the six fits and the candidates about 30 (a fit here holds up to 500 anchor
problems where #180's held at most 200, so each is planned longer); a check a minute or two; one model on dev
(1,344 problems, two answers each) about 60, so the start and `MM` together about 120; the report draft, packaging
and `axonry-check` about 8. That is about 3 hours 55 minutes to the first upload. `MMH` on dev takes about 60
minutes more if it is needed, the control about 60, and the holdout (1,200 problems) about 55 for each answer a
problem, the start and the edit together.

The chain plans from the time step 2 took, a little above those numbers: one model's answers on dev at 0.45 times
step 2 for each answer a problem, the holdout at 0.4 times step 2 for each model and each answer a problem.
- **Two answers a problem on dev**, unless the start and one edit on dev, with the first upload, would then not end
  before the last 40 minutes; then one. Standard errors grow by about two fifths.
- **`MMH` (when it is due), the control and the holdout are read only if each can end before the last 40
  minutes.** The holdout is read with one answer a problem when two would not end in time.

The chain writes its choices to LOG.txt. Correct the planned numbers from the times the first steps really take,
and keep the last 40 minutes for the report and the upload.

## When something fails
- **First rule: an edit that has passed on dev is uploaded before anything else is tried.** If an edit has a dev
  `PASS` and is not uploaded yet, run `report` and `deliver` for it now, whatever else is running or has failed.
- **"the task does not offer the start 0009599139-0015".** The method does not apply to this task. Run nothing on
  another start; report it and end the run.
- **A stopped chain or step.** Check with `nvidia-smi` that nothing still holds the GPUs, then start the chain again.
  Finished steps are skipped.
- **The chain stops with an error, or the same step fails twice.** Do the remaining steps by hand, in the order of
  "Steps": each command is independent, reads what the earlier ones wrote, and says what it needs. From a finished
  step 2 that is: `diagnose`; `gains --gpus 0,1,2,3`; `check --name MM`; `traces --split dev --gpus 0,1,2,3`;
  `evaluate --name MM --split dev --no-control --gpus 0,1,2,3`; on a `PASS`, `report --name MM` and `deliver --name
  MM`. The control and the holdout come after the upload: if one cannot be read, say so in the report. Change no
  file of `derive/`.
- **"the recorded start no longer matches".** The task's start, its `edit.json` or the case's copy of it has changed
  since `setup`. Run nothing on that work directory. If the task really offers the start anew, begin again with a
  new work directory and case.
- **"not placed next to task.json".** Copy the start's `0009599139-0015/edit/` from the bundle next to `task.json`.
- **Generation fails on a GPU.** The message names the log (`runs/<name>.parts/gpu*.log`). Run the same command again.
- **A fit fails.** Its log is `$WSP_WORK/mmgains/logs/Ma_s<seed>.log` or `Mc_s<seed>.log`. Free the GPU it ran on and
  run `gains` again; no edit is made from fewer than three fits a layer, and none from one layer alone.
- **A fit misses a bar** (fewer than 60 examples of its own type, fewer than 60 anchor problems of an instruction
  family or of the puzzle family, or a family on the wrong side). No edit is made. Do not lower a bar and do not
  change the fitter's options. Say what was counted and end the run with `no edit passed`.
- **`MM` fails its check.** It is not used and nothing is generated with it. `MMH` is checked, and read on dev if
  it meets its check; the chain does this by itself. If `MMH` fails its check too, no edit passed.
- **`MM` says `FAIL` on dev.** Do not change a bar and do not re-sample the table. `MMH` is checked and read; the
  chain does this by itself when the time allows it. If it passes, it is delivered.
- **No edit passes on dev.** Nothing is delivered. Say what was measured and end the run.
- **`MMH` is due and cannot be read in time.** No edit passed. Say so in the report.
- **The control cannot be made or read.** Say so in the report and go on. The edit keeps its dev table and its
  upload, and the chain goes on to the holdout.
- **The holdout table says clear harm.** The uploaded edit stays live with its report draft, and `deliver` refuses
  it from then on: write the holdout table and this outcome into `REPORT.md` in the work directory. Do not evaluate
  a second edit on the holdout.
- **The chain's delivery fails.** The `CHAIN DONE` line says so. Read LOG.txt and `chain.log`, mend the cause and
  run `deliver` by hand.
- **`axonry-check` fails.** Nothing is uploaded. Its output is in LOG.txt.
- **Time runs short.** The chain reads dev with one answer a problem, and leaves out the control or the holdout
  when one cannot end in time. The dev table stands; say in the report what was not read.

## What to expect
Each layer can move only problems the start already answers right in some of its answers. On maths, expect fewer
answers cut at the limit and a small gain; on multiple choice, a small gain whose share between the two families is
for the run to measure. `DIAG.md` shows beforehand where the room is. Either gain may lie inside the noise of dev.
Expect little change on puzzles and instruction requests: their answers are held on a counted number of problems,
which is the point of this entry's anchors. The composition may cost what one layer alone would not, since both
layers scale some of the same units; the two groups of the rule read that, and the half dose is the answer to it.
`no edit passed` is a valid result of this method. Expect the result on the holdout to be smaller than on dev.

## Conduct
Practice data and the run's own generations only. No evaluation items, answers or results enter the derivation or the
report. All edit arrays and fitted parameters are computed in the run; the start is the entry's parent, named by
its id and used as published.
