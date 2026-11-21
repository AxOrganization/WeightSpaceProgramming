# Hold the answer to the rule it has just stated, and close the thinking before the token limit cuts it

The edit has two parts, both found on the start model's own answers:
1. a lesson that removes, at the answer tokens, what a stated surface-form instruction forbids;
2. one unit in the last layer that pushes the thinking to end, harder the longer it has run, so that answers close
   before the token limit, where the start model's answers mostly end cut and score nothing.

They act on different answers: the first on answers to prompts with those instructions, the second mainly on
answers that have already thought for longer than the start model's finished answers usually do.

The edit is a new layer on the current champion. Both parts are this miner's own: the lesson and a first closing
unit were committed in entry 0009549509-0022 (`hold-rule-close`, pull request #74, same GitHub account), and this
entry runs them on the champion with the closing unit graded by the thinking's progress. Both were explored on
0009538688-0021 (the numbers below say where they were measured); the run finds them again on the champion's own
answers, so whatever the champion's edit already gained is kept as it is.

## First behaviour: the answer breaks the rule it has just stated
Some instructions leave a trace in every answer token: write without commas, only in capital letters, only in
lowercase, or wrap the whole answer in double quotation marks. The start model restates such a rule in its thinking
("I must not use any commas") and then breaks it in the answer, usually with one or two stray tokens.

What our exploration measured on 0021's model (instruction-following practice holdout, 256 problems × 4
samples, four runs; the share of each instruction type's checks that the toolkit's grader finds followed; our
measurements on practice data, not grades):

| instruction | ifsynth | tuluif |
|---|---|---|
| no comma | 49 % | 60 % |
| all capitals | 62 % | 57 % |
| all lowercase | 82 % | 78 % |
| wrap in double quotes | 85 % | 83 % |
| every other type | 87 % | 79 % |

Every failed no-comma answer we read (36 on ifsynth, 20 on tuluif) states the rule in its thinking, and 25 of the
36 contain at most two commas. Of the failed quotation answers we read (15 on ifsynth, 12 on tuluif), 19 open with a
double quote but do not end with one (most never close it, a few add text after the closing quote); the other 8
never open one.

## Why the model does it (suspected cause)
The model knows the rule but does not apply it at every answer token. The answer is drawn token by token under the
model card's sampling (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5). At each answer token the model
keeps a little probability on tokens the instruction forbids: a comma before a clause, a lowercase word in a
capitals-only answer, the end of the turn before the closing quote. Over a few hundred answer tokens, one such draw
is enough to fail the instruction. The thinking already holds the rule, so the edit only has to make the answer
tokens respect it.

If the cause is right:
- removing the forbidden tokens' probability at the answer tokens, and changing nothing else in the model's own
  distribution, makes the answers follow the instruction without changing what they say;
- the checks of these four types are followed far more often, while other instruction types and other families
  stay where they were;
- a random control of the same size does none of this.

## The lesson: the model's own distribution, minus what the instruction forbids
Nothing is imitated. The forbidden sets are read off the tokenizer.

**Rules that hold at every answer token.**
- no comma: every token whose text contains a comma;
- all capitals: every token whose text contains a lowercase letter;
- all lowercase: every token whose text contains an uppercase letter.

Special tokens (end of turn, thinking tags) are never forbidden. A prompt with several such instructions forbids the
union of their sets.

**Target.** At an answer token, the target is the start model's own next-token distribution under the judge's
sampler, with the forbidden tokens' probability moved:
- to the forbidden token's twin, when the tokenizer has the same text as one allowed token: without its comma
  (`),` → `)`), in capitals (` the` → ` THE`), in lowercase (` The` → ` the`);
- otherwise to the other allowed tokens, in proportion to their probability. The end of the turn is the exception:
  it keeps the start model's own probability.

The twin keeps the answer saying the same thing. The kept end of turn keeps its length. If the end of turn shared
the removed probability, it would gain at every token where much is forbidden (every lowercase word of an
all-capitals answer), and the answers would end early.

**Where it applies.** The loss runs along the start model's own answers, from the first token of the answer section
up to and including its first forbidden token. After that token, the text is one the edited model would not have
written.

**The quotation rule decides at a few tokens only.** On the start model's own answers to "wrap your entire response
with double quotation marks":
- the answer's first token: every token that does not open with a double quote is forbidden; its probability moves
  to the double quote;
- an answer that ended without closing the quote gets the closing quote put in place of its end of turn. At that
  token the end of turn is forbidden and its probability moves to the double quote. At the next token everything
  but the end of turn is forbidden, so the answer ends right after the quote.

Nothing else of these answers is taught. Each such row weighs less than a row of the other rules (`--quote-weight`):
it has a few large changes where the others have many small ones.

**Where the lesson must stay silent.** Some instructions need exactly what the lesson removes or adds:
- a JSON answer needs commas and double quotes, and must end with its closing brace or code fence, not a quote;
- "finish your response with this exact phrase" must end on the phrase;
- "use words in all capital letters fewer than N times" (or at least N times) needs ordinary mixed case.

The start model's own answers to such prompts are held at its own predictions (KL), like the kept rows. These
contrast rows show the trained units where the change must not reach. Without them, a stronger dose of the lesson
leaked into these prompts: in our exploration, the lesson at 1.5 times its trained size wrote a closing quote after
JSON code blocks, and fewer answers followed "fewer than N capitalised words".

**Loss.** KL(target ‖ edited model), averaged over the taught tokens of a row. The thinking of every taught answer is
held at the start model's predictions (KL). Kept rows hold whole answers in other families the same way, so the
change stays where the instruction is.

## Second behaviour: answers cut at the token limit
An answer whose thinking reaches the judge's token limit (32,768 tokens) is cut and scores nothing, whatever it had
found. On 0021's model, its own answers to 360 practice math problems at the full limit (our measurement on
practice data, not a grade):
- 20 % end cut (the hardest level: 30–43 %);
- of the answers whose thinking reaches 20k tokens, 35 % end cut and 27 % still end right; at 26k tokens, 56 % end
  cut and 12 % end right; at 28k, 65 % and 7 %;
- multiple choice: 3 % cut; instruction following and word problems: 0–1 %.

**Suspected cause.** Near the limit the model is not converging: it keeps checking and reopening. It rarely ends
the thinking on its own, and the limit then takes the whole answer. Ending the thinking there and stating the best
answer so far can only turn a sure loss into a possible win, provided the change stays silent where the model
still finishes on its own.

If the cause is right:
- answers cut at the limit become rarer, and the accuracy of math answers rises by a share of them;
- answers that finish before the unit's lengths are unchanged, so word problems and instruction following stay
  where they were;
- the closed answers state a result.

## The closing unit
One unit of the last decoder layer's MLP, written in closed form with the toolkit's `wsp.gate`
(`derive/close_unit.py --gate progress`). A last-layer unit feeds no later layer: it moves only the next-token
distribution at its own position, so it changes when the thinking ends and little else. This follows the toolkit's
playbook (closing only near the token limit) and builds openly on the published procedure of the finished entry
0009538706-0022 (last-layer units gated by a read-out of thinking progress, an inside-the-thinking read-out, and a
least-KL write raising the end-of-thinking token). Here it is one unit, and its lengths come from the start model's
own answers.
- **Lengths.** From the start model's own answers at the full limit: `lo` is the shortest length (in thousands of
  thinking tokens) at which at least 30 % of the answers that reach it end cut, and `hi` the shortest at which at
  least 60 % do (`--lo auto --hi auto`).
- **Read positions.** Thinking positions from `lo` on, between `0.6 lo` and `lo`, and below `0.6 lo`, and answer
  positions. A quarter of the answers (by a hash of the id) is held out of the fit and used only to report where the
  unit fires.
- **gate** `g (p − θ m)`: `p` is a ridge read-out of log10(1 + thinking tokens so far) on the layer's MLP input,
  fitted over the thinking positions; `m` is the constant direction (`m · x ≈ 1`, the MLP has no bias);
  `θ = log10(1 + lo)`; the slope is `g = 4 / (log10(1 + hi) − θ)`. The unit stays near silent well below `lo`, and
  its activation grows with the length of the thinking, so it pushes hardest where answers are about to be cut.
- **up**: a ridge read-out of "inside the thinking" (about 1 there, near 0 in most of the answer), so the unit
  barely writes the end of thinking into an answer.
- **down**: the residual direction that raises log p(end of thinking) with the least KL on the other tokens. It is
  the Fisher-preconditioned gradient in the final norm's output space, mapped back through the norm's gain. It is
  scaled so that, at the unit's mean activation on the positions from `hi` on, log p(end) rises by the boost,
  measured through the real norm.
- **Within the edit limits.** No written row or column may move its tensor by more than 0.4 of the tensor's norm
  (the limit is 0.5): a steeper gate is scaled down, and a dose whose write would pass 0.45 is not written.
- **Which unit.** The unit of that layer with the smallest mean |activation| × ‖down column‖ on the read positions.
- **Doses.** Boosts of 4, 6 and 8 log-odds at `hi`. At most late thinking positions the end of thinking is far from
  likely (mean log p(end) about −30 there), so too small a dose changes nothing. Too large a dose closes too early,
  in the middle of a computation, and the answer then states no result. The dose is chosen on dev (step 7).
- **Why a graded gate.** In our exploration, a switch gate (the LDA direction between thinking positions beyond
  `lo` and below `0.6 lo`, thresholded at the 99.9 % quantile of the negatives) fired about equally from 24k to 32k
  tokens (0.14, 0.13 and 0.11 of its mean on held-out answers). It could not tell an answer about to be cut from one
  that is merely long, so a dose that closed the cut answers also closed good ones. The progress read-out
  (correlation 0.92 with log10 of the length on the fit positions) fires 0.04, 0.15, 0.39, 0.58, 0.73 and 1.12 of
  its reference at 8k, 12k, 16k, 20k, 24k and 28k tokens.

## How the agent finds the edit
All steps use the toolkit's cases and grader. The scripts are in `derive/`, each with its usage in its docstring.
`<w>` is a work directory.

0. **Start: the champion.** This entry's parent is `"champion"`: when the run starts, the subnet sets it to the
   champion of that moment, and the bundle holds that champion's published edit as the entry's parent
   (`<id>/edit/`). Build on it: copy it next to `task.json` under its id. The edit is a new layer on that model: its
   own ops are measured against it, and the uploaded `edit.json` names that id in `"parent"`. Do not compare it with
   other starts: the champion's gains are kept only by building on it, and a start that gains on the benchmarks may
   gain little on practice dev.
   - Make one case per practice source: `gsm8k_train`, `math_train`, `mmlupro`, `ifsynth` and `tuluif`, with the
     profile of the base, the subsets `dev128=dev:128`, `dev256=dev:256` and `holdout256=holdout:256`, and
     `python -m wsp.case init ... --parent <id>/edit/edit.json`.
   - All later steps use these cases; "the start model" below is the champion's model.
1. **Constraint prompts** (`derive/cset.py --per-type 400`). Build them for the four instructions:
   `punctuation:no_comma`, `change_case:english_capital`, `change_case:english_lowercase`, `startend:quotation`.
   - 40 % are tuluif practice requests. The instruction is added unless it conflicts with the request's own,
     following the registry's conflict table.
   - 60 % are gsm8k_train practice problems with the instruction added.
   - None is in a dev or holdout split of the cases, and none is a word problem that a held-out ifsynth item is built
     from.
   - The instruction text comes from the instruction registry the toolkit's ifsynth set is built with, so the
     toolkit's grader checks it.
   - **Contrast prompts** (`derive/cset.py --contrast --types detectable_format:json_format startend:end_checker
     change_case:capital_word_frequency --per-type 160 --probe-out <w>/ccset_probe.jsonl`). They are built the same
     way, on requests that carry none of the four instructions. A quarter of them (by a hash of the prompt id) goes
     to the probe file and is never taught.
2. **Own answers** (`derive/cset_answers.py --n 4 --max-tokens 6144`). The start model answers each prompt 4 times
   with the judge's sampling, in one shard per GPU. The script prints, per type, how often the start model follows
   the instruction. It also answers the contrast prompts twice (`--n 2`) and the probe prompts twice (`--n 2`, the
   probe's baseline).
3. **Kept answers** (`derive/answers.py --n 1 --max-tokens 4096`). The start model answers 300 practice problems
   each from `gsm8k_train`, `mmlupro`, `ifsynth` and `math_train`, all outside the dev and holdout splits. Only
   finished answers become rows, so math keeps its shorter answers (about 40 of 300 in our exploration). Each family
   is drawn equally often in training, however many rows it has. Without kept math answers the lesson moved math
   (size 1.5 alone: −2.1 ± 1.7, × 1).
4. **Rows.** Both scripts read the same answer files. Each answer must be finished, with its thinking closed and no
   repeating tail. Per type: the answers that broke the instruction (up to half of the cap), then answers that
   followed it for the rest.
   - `derive/cset_frows.py --cap 500 --keep 150 --contrast-answers <the contrast answer files> --contrast-cap 150`:
     the three every-token rules, 150 kept answers per family, and up to 150 contrast rows per contrast instruction.
   - `derive/cset_qrows.py --cap 500`: the quotation rule.
   - Join the two sets (train with train, validation with validation). A hash of the prompt id sets aside the
     validation rows in both.
5. **Units** (`derive/train_units.py --keep-thinking 1.0 --max-len 6000 --epochs 1.5 --lr 1e-4 --forbid-map
   --forbid-keep-end --quote-weight 0.3 --fam-weights c_json_format=0.5,c_end_checker=0.5,c_capital_word_frequency=0.5`).
   - A calibration pass ranks the MLP units of every layer by the squared gradient of this loss, with at most a fixed
     share from one layer.
   - As many units are taken as the payload limit allows, and only they train.
   - Validation prints, per type: the expected number of forbidden tokens per answer up to the first one, and for
     the quotation rows the probability of the forbidden moves at their few tokens (start → now). It also prints
     the KL on the held thinking and on the kept answers.
   - Expect each forbidden count and the quotation probability to fall by more than half while the KL stays small.
     If the KL on a kept family rises above the KL on the held thinking, the change leaks: prefer the smaller size
     in step 7.
6. **Edits** (`derive/write_edit.py`). Write the trained change at sizes 1 and 1.5, plus the random twin (`--twin`)
   at size 1. Each edit must build and pass the limits (`--check`). An edit is `<edit>.json` with its arrays in
   `<edit>.npz` beside it, copied to every case (`derive/sizes.py`); each writer here prints the upload's size and
   stops if it would come near the subnet's 8 MiB.
   - **Long own answers** (`derive/answers.py --max-tokens 32768 --n 1`, one shard per free GPU, while the units of
     step 5 train). The start model answers practice problems outside the dev and holdout splits at the full limit:
     360 `math_train`, 150 `mmlupro`, 80 each of `ifsynth`, `tuluif` and `gsm8k_train`. Go on with the closing unit
     only if some length exists at which at least 30 % of the answers that reach it end cut.
   - **Closing unit** (`derive/close_unit.py --gate progress --case <start's math case> --cases <every case>
     --traces <the long answers> --name G4 G6 G8 --boost 4 6 8 --cache <w>/close_cache.npz --check`). It prints
     `lo` and `hi`, the read positions, the progress read-out's correlation, the relative change of each written
     tensor, the realised boost and the KL on the other tokens. It also prints the firing by thinking length on
     held-out answers: it must rise with the length, stay below a tenth of the reference under `0.6 lo`, and stay
     small in the answer. The build fails if an edit passes a limit.
   - **Joined edits** (`derive/merge_edits.py --edits <IF size> <closing dose> --name <joined> --check`).
7. **Choose on dev only** (`derive/paired.py`, dev128 × 2, paired with the start on the same problems and seeds).
   - Run `ifsynth` and `tuluif` for every edit, then `gsm8k_train`, `mmlupro` and `math_train` for the size that
     leads.
   - **Closing dose, on dev256.** 128 problems are too few to choose a dose on, so this step uses dev256 × 2. Run
     `math_train` for G4, G6 and G8, each joined with the leading size, and for the leading size alone. Read the
     paired gain against the start, the cut share, and the count of finished answers without a result. A dose
     closes too early, and is not eligible, when its share of finished answers without a result exceeds the start's
     by more than a fifth of the cut share it removes. Choose the eligible dose with the largest math gain; take a
     smaller dose only if its gain is within 1 SE of that one's. Keep the unit if the chosen dose gains over the
     leading size alone (`paired.py --set dev256 --against <the size alone>`); otherwise leave it out.
     On the champion we explored, 128 dev problems overstated every dose: boost 4 gained +5.1 ± 2.4 on dev128 and
     +2.6 ± 1.4 on holdout256; boost 6 gained +7.0 ± 2.6 and +1.7 ± 1.5, and left 14 of 512 finished answers
     without a result against the start's 2 (it fires again inside the answer). The rule takes boost 4 there.
   - **Leak probe.** Each size answers the probe prompts twice (`derive/cset_answers.py --edit <size> --n 2`), and
     `derive/probe.py --base <the start's probe answers> --edit <one file per size>` compares them with the start's,
     paired by prompt. It prints, per contrast instruction, the share of answers that follow it (start → edit). A
     size whose contrast instruction falls by more than 2 SE is marked `LEAK`.
   - **Prompts without the four instructions.** On `tuluif` dev128, `paired.py --split-types punctuation:no_comma
     change_case:english_capital change_case:english_lowercase startend:quotation` prints the gain on the prompts
     that carry one of the four and on those that carry none. The lesson must change only what it targets: a size
     whose prompts without the four fall by more than one standard error is marked `SPILL`. On the champion we
     explored, the lesson fitted on another model's answers gained +15.9 ± 3.7 on the first and lost 3.2 ± 1.6 on
     the second; fitted on the start's own answers (0021), the second stayed level (−0.4 ± 1.7).
   - Eligible: no source below −2 SE, no `LEAK` on the probe and no `SPILL`.
   - Choice: the eligible size with the larger sum of the two instruction-following gains. If the sizes are within
     1 SE of each other on that sum, take the smaller one.
   - The choice is valid only if its twin is below it on both instruction-following sources.
8. **Freeze, confirm, upload.** Join the chosen size and the chosen closing dose (`derive/merge_edits.py`). Confirm
   the joined edit once on holdout256 × 2 for all five sources, paired with the start. Upload it unless a source is
   below −2 SE there. If a source is, find the part responsible and drop it: the closing unit if the loss is in a
   family whose answers are long, the size otherwise (take the smaller one). Confirm the result the same way before
   uploading it.
   - **Size of every upload.** Package it first (`python -m wsp.upload <case> <edit> --task task.json --dry-run`)
     and add up the package's `edit/` directory (`du -cb <package>/edit/*`): `edit.json`, `edit.npz` and
     `REPORT.md` together must stay under 8 MiB (8,388,608 bytes), or the subnet refuses the upload when it
     collects it (`FAILED BAD_UPLOAD`); `axonry-check` does not test this. An edit written by the scripts here
     takes about its payload (at most 4 MiB) and a small spec. A larger package means an array was written into a
     spec as text: write the edit again with these scripts.

## Controls and what counts
- **Twin.** The random twin at the same size and in the same layers must not show the gain.
- **Per instruction type.** The grader's checks for the four types must be followed far more often, and those of
  every other type must stay level. Without that, an instruction-following gain is noise, whatever its sign.
- **Prompts without the four instructions.** Their pass rate stays level.
- **Other families.** Word problems, multiple choice and math stay within noise (paired gain above −2 SE).
- **Closing unit.**
  - It fires little below `0.6 lo` and in answers, and more the longer the thinking (held-out report).
  - Answers cut at the limit become rarer.
  - Answers that write the end of thinking a second time are counted. The start model already does it: its answer
    turns back into reasoning and closes again (22 of its 256 math dev answers in our exploration). The unit can fire
    again in such an answer (dose 8: 63 of 256). They are graded as written, so the math gain already includes them.
    If they are more often cut than the start model's, the dose is too large.
  - Finished answers without a result do not pile up: if they do, the unit closes too early, so take a smaller
    dose.

## What to expect, and when to stop
On the champion 0009547524-0008 we replayed the procedure on that model's own answers (holdout256 × 2, paired with
the champion's own runs on the same problems and seeds; our practice measurements, not grades). The lesson at size
1.5 gained +4.7 ± 2.1 on ifsynth and +1.1 ± 1.7 on tuluif. On the prompts that carry one of the four instructions,
the share of passing answers rose by 32 points for no comma, 23 for all capitals, 11 for quotation and 11 for all
lowercase; the other prompts moved by −1.0 ± 1.4. Word problems moved by +1.9 ± 1.4, multiple choice by −0.1 ± 1.9
and math by −0.1 ± 1.4. The closing unit at boost 4 gained +2.6 ± 1.4 on math, with the cut share down from 15.8 to
8.8 %.

Our replay of this procedure's parts with its own scripts on 0009538688-0021 (the model it was explored on):
holdout256, paired by problem with 0021, which we sampled 14 to 20 times per problem over several boxes. These are
our practice measurements, not grades.

| practice source | lesson, size 1.5, kept math answers | the same + closing unit, boost 6 |
|---|---|---|
| ifsynth | +4.1 ± 1.7 (× 2) | +6.6 ± 1.4 (× 4) |
| tuluif | +1.7 ± 1.4 (× 4) | +1.9 ± 1.4 (× 4) |
| gsm8k_train | | +2.1 ± 1.3 (× 2) |
| mmlupro | | +0.9 ± 1.5 (× 2) |
| math_train | +2.2 ± 1.2 (× 2) | +5.4 ± 1.0 (× 4; cut 20.2 → 5.5 %) |
| math_train, dev128 | | +5.3 ± 1.9 (× 2) |

Without the kept math answers the same lesson moved math by −0.2 ± 1.3 (× 2). With the earlier closing unit (an
on/off gate, boost 8) the joined edit gained +1.6 ± 1.4 on math holdout on the same box, cut 20.2 → 12.9 %.

- **Leak probe** (held-out contrast prompts, × 2, start → edit). Size 1: JSON 90 → 89 %, capital-word count 50 → 55 %,
  closing phrase 82 → 84 %. Size 1.5: JSON 90 → 81 % (−8.6 ± 5.2, under the 2-SE flag), the other two level.
- **Closing doses** (math_train dev128 × 2, each joined with the lesson, paired with 0021):

  | boost | paired gain | cut share (0021 18.5 %) | finished without a result (0021 0.9 %) |
  |---|---|---|---|
  | 4 | +1.8 ± 2.3 (lesson without kept math answers) | 14.1 % | 0.8 % |
  | 6 | +5.3 ± 1.9 | 5.5 % | 2.3 % |
  | 8 | +3.4 ± 2.1 | 3.1 % | 2.0 % |
  | 10 | +1.8 ± 2.0 | 2.3 % | 5.5 % |

  Boost 10 closes too early by the rule of step 7 (finished answers without a result +4.6 points against a fifth of
  the 16.2 points of cut share it removes); boost 6 has the largest gain and boost 8 is not within 1 SE above it, so
  the rule takes boost 6. A stronger dose closes answers that would have finished right: in a separate test, closing
  the start's word-problem answers by force at a fixed length lost at every length tried (answers past 6k tokens
  48 → 30 % right), so the unit must only push where answers are about to be cut.
- **Instruction following is not taxed by the unit:** tuluif +1.7 (lesson) and +1.9 (lesson + unit), × 4 each.

The lesson alone, in the first exploration (without the contrast rows; size 1, × 4):

| practice source | paired gain | no comma | all capitals | all lowercase | quotes | other types |
|---|---|---|---|---|---|---|
| ifsynth | +5.7 ± 1.2 | 49 → 85 % | 62 → 71 % | 82 → 98 % | 85 → 100 % | 87 → 88 % |
| tuluif | +1.3 ± 1.3 | 60 → 73 % | 57 → 64 % | 78 → 90 % | 83 → 96 % | 79 → 78 % |

The columns per instruction type give the share of that type's checks followed, before → after. Split by prompt:
on the prompts that carry one of the four instructions the gain is +21.7 ± 3.2 points (ifsynth, 60 prompts) and
+11.7 ± 3.5 (tuluif, 44); on the others it is +0.9 ± 1.1 and −0.9 ± 1.3.

Other families, this procedure at size 1 (× 2): word problems −0.2 ± 1.4, multiple choice +3.6 ± 1.6 (more than
the lesson should give; read it as no harm). The same lesson without the quotation rows (× 2): word problems
+1.5 ± 1.3, multiple choice +0.4 ± 1.6. The first version of the lesson (no twins, end of turn sharing the removed
probability, × 4): word problems −0.5 ± 1.0, multiple choice +0.6 ± 1.1, math +0.6 ± 0.9.

What the parts add, on the prompts that carry one of the four instructions (ifsynth / tuluif):
- first version, no quotation rows: +14.4 / +6.5 (in total, averaged over two boxes: ifsynth +2.6, tuluif +1.8);
- with twins and the kept end of turn: +16.3 / +11.1;
- with the quotation rows as well (this procedure): +21.7 / +11.7;
- the first version at size 1.5: +20.4 / +12.8, other prompts level. The lesson is not fully applied at size 1,
  which is why step 6 writes both sizes.

Validation during training (start → end): expected forbidden tokens per answer, up to the first one: no comma
0.94 → 0.12, all capitals 0.25 → 0.06, all lowercase 0.43 → 0.03; probability of the forbidden moves at the
quotation tokens 0.30 → 0.02. KL on the kept answers was at most 0.0013 per token.

On long writing (41 practice requests asking for essays, stories or word minimums, with the no-comma instruction
added), the first version raised no-comma compliance on answers of 250 tokens or more from 43 % to 91 %.

A three-times-larger learning rate removed a little more of the forbidden tokens. It also moved the kept answers
three times as far and gained less in total, so prefer the gentler run.

Stop and fall back to the smaller size when:
- a family without these instructions drops: the lesson leaks;
- the twin matches the edit.

Leave the closing unit out when no eligible dose gains on math dev, or when the joined edit loses a long-answer
family on holdout. The lesson alone is then the edit.

## Time plan (4 × RTX 4090, 6 h)
| Phase | Steps | Time |
|---|---|---|
| Setup | cases on the champion (0) | 25 min |
| Teaching material | prompts, own and kept answers (1–3) | 35 min |
| Training | rows and training (4–5) on one GPU; the long own answers of step 6 on the others | 45 min |
| Closing unit | fit and the three doses (6) | 15 min |
| Selection | sizes, twin and leak probe; closing doses on math dev256 (7) | 90 min |
| Confirmation | the joined edit on holdout (8) | 60 min |

Math answers at the full limit are the slowest runs: start them first in each phase. Upload the size-1 edit as soon
as step 6 has built it, and again after each later decision, each time after the size check of step 8. Keep the
last 30 minutes for the final upload and the report.
