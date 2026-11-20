# State the answer: in the reference's form, and before the budget runs out

## The problem
The published edits so far make the model finish more of its answers. Two things still cost it answers that it has.

**The form of the answer.** The prompt asks for the final answer in `\boxed{}` and says nothing about how to write it.
The references of the word problems are bare numbers, and the model often puts more than the number into the box:
`35\%` for 35, `\$694` for 694, `16.00` for 16, `8 \text{ cm}` for 8. Such an answer has the right value and is not
the reference as written. The toolkit's grader compares the two strings first, after a light normalisation, and only
when they differ falls back on a check of mathematical equivalence: an answer that is identical to its reference is
right however answers are compared, and one that is not is right only as long as that check accepts it. In our runs on
a model that had been taught to finish, three in a hundred of the right answers to word problems taken at random were
written differently from their reference, and far more where the question invites it: half of the right answers when a
percentage is asked for (a percent sign), a fifth when an amount of money is asked for and the problem states prices
with cents (decimal zeros, a currency sign). Measure it first, on the model's own practice answers: among the answers
that the grader counts right, how many are identical to their reference after the grader's own light normalisation,
per family and per kind of difference.

**The token limit.** What is still lost to the token limit is lost in the same way as before the published edits. The
thinking goes on, circling the problem, until the budget is spent; an answer cut at the limit has no final answer and
is scored wrong; and nothing in the model's behaviour changes as the budget runs down: at 90% of it the model writes
as if it had just started. On a model that already finishes most answers this is concentrated where the problems are
hardest (in our runs math: one answer in five to seven), and it is no longer a matter of speed: making such a model
brisker once more gained nothing in our runs. Measure it first, on practice problems of every family the toolkit ships
(arithmetic, other math, multiple choice, instruction following on math questions and on varied requests; never the
benchmarks): the share of answers cut at the limit, with the thinking still open and after it had closed; the accuracy
of the answers that finish; how long the thinking of finished answers is; and, for each depth of thinking, how many
answers reach it and how many of those still end correct.

## Why we think it happens, and what should happen instead

### The form
The model ends its reasoning in prose ("so the total is $16.00", "that is 35% of the flowers"), and the box repeats
the last line as it stands. This is a habit of writing, not a lack of ability, and the choice is close: at the token
where its answer and the reference part (a brace that closes the box where the model went on with a percent sign or a
decimal point; a brace that opens it where the model put a currency sign first), the model gave the reference's token
a probability of 0.19 to 0.30 on average in our runs, and more than a half in one case in five to eight. A choice this
local does not need the model to be retrained. A few MLP units that read the state inside the box can tip it, and
nothing outside the box has to change.

So the units are fitted on the starting model's own answers. An answer that is right and written differently is
rewritten inside its last box when deleting the decoration around its value (a currency sign, a percent sign, a unit,
decimal zeros of a whole number, a degree sign, "x =") leaves exactly the reference. The tokens of the rewritten box
are the targets. An answer that is identical to its reference teaches its own box as it is, so that what is already
right stays right, also where a decimal, a unit or a degree sign is the right form. Everywhere else, in the thinking
and in the rest of the answer, the starting model's distribution is the target and the units are held silent. Every
other difference is left alone, and in the other math family that is most of them: a fraction written with a slash or
as a decimal where the reference writes `\frac`, `\dfrac` against `\frac`, a sign or a unit that only the reference
carries, the order of terms. Nothing in the problem says which way such a reference goes, and where we let the units
learn the one regular case among them (`\frac{a}{b}` for `a/b`) they began to disturb boxes that had been right: one
in twenty-five on word problems, and on problems with a fraction for an answer five right answers in 120 turned wrong
while five gained the reference's form.

Three predictions, each checked before anything is delivered:

1. on answers the units were not fitted on, the reference's token becomes the likely one at the decisions that a
   rewrite changes;
2. boxes that were identical to their reference stay as likely as they were, in every family;
3. on real answers to problems kept apart, more answers are identical to their reference, no fewer are right by the
   grader, and nothing else about the answers changes (their length, the share cut at the limit).

If the starting model writes hardly any right answer differently from its reference (fewer than one in a hundred of
the right answers to the word problems chosen in step 1), there is nothing to fit: say so and go on with the second
part.

### The limit
Very long thinking is a sign that the answer will not come: once the thinking has used most of the budget, only a few
answers in a hundred still end correct. Yet by then the model usually holds a view of the answer. If its thinking is
closed for it at that point and the answer is started for it, what it writes next is right far more often than what
thinking on would have produced. Closing the thinking alone is not enough where a boxed answer is asked: left to write
the answer, the model starts the problem over inside the answer and runs out of budget again. So two things are
missing, not ability: a sense that the budget is nearly spent, and the habit of then stating the answer at once.
Teaching the model to finish sooner does not supply either: it shortens the answers that can be shortened, and the
rest still meet the limit unprepared. Each part of this can be checked on the starting model before anything is
edited:

1. past some depth, continuing ends correct less often than stopping and answering at once would (replay its own long
   traces: closed at that depth, with the answer left to it, and with the answer started by a short stem);
2. the stem matters where a boxed answer is asked: with a bare close most of those replays never reach a final answer.
   Where the answer is written out in full (instruction following), the bare close is enough: the model then writes
   its answer as it would have;
3. the model's state does say how long it has been thinking: units reading it can be fitted to act late and stay
   silent early, on traces they were not fitted on.

If 1 fails in every family, there is nothing to gain from this part: say so and deliver the form units alone. If 3
fails (the fitted units also fire early, or the model stops closing its thinking by itself), no wrap-up units are
shipped.

A model that already finishes most of its answers demands one thing more: the edit has to leave it alone until the
late lines. Holding the starting model's output distribution about right along its own traces is not enough. In our
runs on such a model, fits that met every held-out number that way lost on real answers. Their rewritten units were
active everywhere (a fit without the silence term wrote a third of its layers' MLP output at ordinary positions,
arranged so that the predictions along the traces hardly moved), and the closing token had lost more than half of its
probability at the line ends of the first half of the thinking. Writing by itself, the model then concluded later,
more answers reached the late lines and were wrapped up with a guess than the edit rescued, and math lost 8 to 11
points in three fits of four. So the fit here holds the units themselves silent wherever nothing should change, holds
the starting model's distribution five times more firmly, and overwrites only units that never fire strongly
(overwriting the ninety-six units that are merely quiet on average, with nothing written, by itself cost between one
and two points in our runs, within its standard error but in the same direction on two models). With these the edited
model is the starting model up to the late lines.

## Which model to start from
The task holds this entry's parent in `parent/edit/` and, when the champion is another model, the champion in
`champion/edit/`. Both parts of the method are fitted on the starting model's own traces and stack on that model, so
the choice is made once, before step 1. `python submission/derive/run.py start` (run it where those two directories
are) reads them and prints the choice:

- the champion, if the champion is built on this entry's parent: it then has everything the parent has;
- the parent otherwise, also when the champion is another model: the method was developed and measured on the parent,
  and what it adds is added to that model.

The second case is the likely one, and the choice matters. When this was written the champion was `0009538688-0019`,
which is not built on the parent, and on the practice problems the parent was ahead of it: outside math the parent
loses hardly any answer to the limit (that champion: one in eight in multiple choice, one in fifteen on word problems,
one in eighteen on varied requests), and it is at least as accurate in each of those families. Both parts of this
method then add to the model that is already ahead. Added to that champion instead, the form units still work, but no
wrap-up fit became usable on it in our runs (none of thirteen seeds), so half of the method would be lost.

Copy the directory you start from to the sandbox under its own name (`parent/` or `champion/`, next to `task.json`):
the upload refers to it by that name, and `run.py setup` keeps it.

## Approach
All steps are commands of `derive/run.py` (`-h` lists them). It uses the toolkit for prompts, seeds, sampling,
grading, building and uploading, and adds generation on all GPUs at once (`derive/gen.py`), the fit of the form units
(`derive/form.py`) and the fit of the wrap-up units (`derive/units.py`). Depths are shares of the profile's token
budget, counted in tokens of thinking. Work in the sandbox under `/workspace/work` with the bundle's `task.json`,
`submission/` and the directory you start from copied there. Run long jobs detached (`tmux`, or `nohup` with the
output in a log file) and follow them in `$WSP_WORK/wrapup/LOG.txt`, where every command writes what it prints; if a
generation job is stopped, check with `nvidia-smi` that nothing still holds the GPUs before starting the next.

1. **Starting model and splits.** `python submission/derive/run.py setup --parent <start>/edit/edit.json`. It creates
   the toolkit case `wrapup`, builds the starting model and splits the practice problems of every family, by problem,
   into `fit` (to find and fit the edit: 100 problems a family, 300 for math, which loses most answers to the limit
   and so supplies most of the traces the wrap-up fit needs), `dev` (to choose: 96 a family, 192 for math and for the
   two families whose answers are written out, which are short to generate and where an edit that touches answers
   shows first) and `holdout` (150 a family, to confirm once, at the end). For the family whose references are bare
   numbers (the word problems) it sets two more groups of problems apart, none of them in another split and chosen by
   what the question asks for (a percentage; an amount of money, in a problem with prices in cents or without;
   anything else): `form` (420 problems, to fit the form units; the first three kinds are where a bare number is
   decorated most often, so they supply the answers the fit learns from) and `formdev` (200 problems, to check those
   units on real answers).
2. **The model's own traces.** `run.py traces --split fit --gpus 0,1,2,3`, then `run.py traces --split form --gpus
   0,1,2,3`: two answers per problem with the profile's sampling. Both parts of the edit are made from these traces
   and from nothing else.
3. **The form units.** `run.py form --gpu 0` prints the table named under "The problem" (per family: answers, right by
   the grader, identical to the reference, right but written differently, by kind of difference) and fits the units:
   in the last two layers, the 8 MLP units per layer that are used least on the model's finished answers (as defined
   in step 7) are rewritten, as `set_rows` / `set_cols` ops. They start with random read rows and write nothing. One
   round of 1,500 steps, each on one rewritten answer and one other: the tokens of a rewritten box, and of a box that
   was identical to its reference, are the targets; at every other position the starting model's distribution is held
   with five times the weight and what the units write, measured against their layer's MLP output, is driven to zero.
   A fifth of the problems is held out, and the command ends with the numbers on them, starting model next to edited
   model: the probability of the reference's token at the decisions a rewrite changes, overall and per kind of
   difference, and the share of those decisions where it is now above a half (prediction 1); the probability of the
   box tokens of answers that were identical to their reference, per family (word problems, other math, multiple
   choice), and how many such boxes lost a fifth of their probability (prediction 2); the KL divergence of all other
   predictions from the starting model; what the units write where nothing should change. The conditions, checked by
   the command: at least six in ten of the changed decisions favour the reference's token; in no family do more than
   one in twenty of the boxes that were identical (or more than two, where there are few) lose a fifth of their
   probability; KL at most 0.004; the units write at most 0.01. It prints `USABLE` or `NOT USABLE` with the conditions
   that fail, writes the edit `form` and builds its model. The fit takes about three minutes; in our runs it was
   usable at the first seed (`--seed` gives another). While it runs, generate the starting model's answers on the
   problems kept apart: `run.py traces --split formdev --gpus 1,2,3`.
4. **The form units on real answers.** `run.py form-eval --gpus 0,1,2,3` generates the edited model's answers on the
   same problems and seeds and prints, per kind of question and overall: right by the grader and identical to the
   reference, starting model next to edited model, with the paired gain of the second and its standard error; the
   table of step 3 for both; mean tokens, the share cut at the limit and the answers that end inside the thinking; and
   `PASS` or `FAIL`. It passes if more answers are identical to their reference, beyond twice the standard error; no
   fewer are right by the grader, by more than twice theirs; the share cut at the limit does not rise beyond what two
   runs of one model differ by (twice the standard error of that difference, a point at least); and no more answers
   end inside the thinking, by more than one in a hundred (prediction 3). Two runs of one model do not write the same
   answers, so single problems differ in both directions: read the totals. If it passes, deliver the form edit at once
   (`run.py report --name form`, then `run.py deliver --name form --task task.json`, see step 11): it is an edit in
   its own right, and every later upload replaces it. If the fit is not usable or fails here, try one other seed; if
   that fails too, go on without the form units and say so in the report.
5. **Diagnose.** `run.py diagnose` prints and saves the tables named under "The problem" for the token limit.
6. **Replay.** `run.py replay --gpus 0,1,2,3` takes every answer whose thinking reaches a depth (70%, 84%, 90%, 94%
   and 97% of the budget), closes the thinking at the first line end past it and lets the starting model continue,
   once with the answer left to it and, for tasks that ask for a boxed answer, once with the answer started by the
   stem; what the model writes after the stem is its own immediate answer. The table compares, per family and depth,
   ending correct if left alone with ending correct if stopped, and what accuracy would become (predictions 1 and 2 of
   the limit). The stop depth is 90% of the budget: the model's own finishing is left alone as long as possible, and
   the last tenth of the budget is kept for the answer. The fitted units start to close somewhat before the stop
   depth; the replays at 70% and 84% show that closing there costs nothing either (by then few answers still end
   correct), and those from 90% on supply the fit with the model's immediate answers, at three points of the last
   tenth: the units are asked to close somewhere in it, not at its first line, which a weaker and therefore cleaner
   trigger can do. Write down the table. The table is also the ceiling of what the wrap-up units can gain; on a model
   that finishes most answers it is a few points in the families that still lose answers and nothing elsewhere.
7. **Fit the wrap-up units.** `run.py fit --name <name> --seed <s> --gpu <g>`: in the last four layers (they write the
   next token, and only they have to be run during the fit) the 24 MLP units per layer that are used least on the
   model's finished answers are rewritten, as `set_rows` / `set_cols` ops on their `gate_proj` / `up_proj` rows and
   `down_proj` column; the units of the form fit are never taken. Used least means: the smallest largest-activation
   within an answer, averaged over the answers, times the output norm. Overwriting a unit costs what the unit did, and
   a unit that is quiet on average but fires strongly now and then is not free: in our runs the model without the
   twenty-four units a layer chosen by mean activation moved its predictions three to five times as far from the
   starting model's as the model without those chosen this way, which could hardly be told from the starting model. A
   rewritten unit starts with random read rows and writes nothing. The weights are fitted on cut answers and on
   finished ones (360 of them, half the longest, which a trigger on depth could hurt most; a fifth of the problems is
   held out for step 8): at the line ends of cut answers from the stop depth on (sequences ending at 90%, 94% and 97%
   of the budget) the next token should close the thinking; after such a close, for boxed tasks, the stem and the
   model's own immediate answer from step 6 (a target only where that answer was right), then the end of the answer.
   Everywhere else the starting model's next-token distribution is the target, unchanged, with five times the weight
   of the targets: up to 14% of the budget below the stop depth in those answers, and all through finished answers, no
   token the fit promotes may gain logit, wherever the starting model itself closes or might close its thinking its
   distribution is kept with extra weight, and the units themselves are held silent: what they write, measured against
   their layer's MLP output, is driven to zero at these positions. The fit runs one long round (2000 steps, the step
   size falling to a tenth) on these terms: the trigger is found fastest that way. A fit left at that, however,
   withholds the end of answers: it teaches the end token after the stem, and to keep that token out of the thinking
   the units learn to suppress it, also where an answer of the starting model did end. The model then goes on writing
   after its answer, or closes its thinking a second time and repeats the answer, which costs most where the answer is
   written out in full (instruction following). So short rounds follow (1000 steps at a small step size) in which the
   starting model's distribution inside answers is held with extra weight, at their line ends and above all at their
   end. After each round the fit measures the numbers of step 8 on held-out traces; it stops at the first round that
   meets every condition (at most four rounds, usually one or two) and prints `USABLE`; if no round does, it keeps the
   weights of the round that failed the fewest conditions and prints `NOT USABLE` with the conditions that fail. The
   command then writes the edit spec into the case, builds the model and prints the payload and the relative change of
   every matrix. If the form units have passed step 4 they go into the same edit (the command says so; `--no-form`
   leaves them out, and `run.py merge --name <name>` writes the edit of a saved fit again): what is evaluated and
   delivered from here on is the two together. Whether a wrap-up fit gets there depends on its random start and on the
   starting model (in our runs on one model between one seed in five and three in seven did, most of them by the
   second round; on another none of thirteen did, and the fits that came closest went to step 9), so fit several seeds
   side by side: seeds 0 to 2 on three GPUs while the fourth generates the starting model's dev answers (`run.py
   traces --split dev --gpus 3`), then seed 3 on it, and go on with seeds 4 to 7 if none is usable. A fit that is not
   usable after its second round seldom became usable in the third or fourth in our runs: once another seed is usable,
   stop it and give its GPU to step 9. A fit keeps about 55 GB of main memory for the kept layer inputs: run only as
   many at a time as fit into what `free -g` shows as available, with 20 GB to spare.
8. **Selectivity before generation.** Each round ends with numbers on held-out traces, starting model next to edited
   model: the probability of closing at the late line ends, and the chance that a cut answer has closed its thinking
   by 90%, 94% and 97% of the budget (accumulated over all its line ends); the probability of the stem and answer
   tokens after the close; the chance that an answer closes its thinking below 70% of the budget (accumulated
   likewise), for cut and for finished answers; the share of line ends where closing is likely, by depth; the
   probability of closing at the model's own closing points; inside answers, how often closing again is likely and the
   probability of ending the answer where the model ended it; the KL divergence of all other predictions from the
   starting model; and what the units write where nothing should change, as a share of their layers' MLP outputs
   (summed over the four layers). The conditions, checked by the fit itself: a cut answer has closed by 97% of the
   budget with a chance of at least 0.6 (measured along the starting model's own traces this number runs low: a fit
   with 0.54 ended nine in ten of the stuck answers when the edited model wrote by itself, and whether the answers end
   is settled in step 9); stem and answer tokens at least 0.8; KL at most 0.015; the units write at most 0.02 where
   nothing should change; the chance of closing below 70% of the budget no higher than the starting model's by more
   than 0.04 for finished answers (prediction 3 of the limit; for answers that run out anyway an earlier close loses
   nothing, and only a rise by more than 0.5 counts against the fit); the model's own closing at least nine tenths of
   what it was; inside answers, closing again no more likely than in the starting model; and the end of an answer
   (either end token) at least 0.85 times as likely as in the starting model where the answer did end. Generate with
   usable fits only, the one with the lowest KL first (`run.py check --name <name>` measures a saved fit again): an
   edit that stops answers too soon replaces right answers by guesses, however well it closes late. How many seeds are
   usable differs from one set of traces and one starting model to the next (in ours from none in five to three in
   seven). If none of eight seeds is usable, take the fit that fails the fewest conditions by the smallest margins,
   provided that it still closes in time, and let step 9 decide. These numbers are necessary, not sufficient: they are
   measured along the starting model's own traces, and only step 9 shows what the edited model does when it writes by
   itself.
9. **On real answers.** `run.py evaluate --name <name> --split dev --gpus 0,1,2,3` generates the starting model (once,
   if step 7 has not already) and the edited model on the same problems and seeds and prints the paired gain with its
   standard error, answers rescued and lost and the cut share, per family and overall; the same for the families that
   lose at least one answer in twenty to the limit taken together, with the share of their answers cut while the
   thinking was still open, and for the families whose answers are written out taken together; how many answers of
   boxed tasks closed their thinking past 70% of the budget and how many of those go on with the stem; how many
   answers end inside the thinking; how many close their thinking a second time inside the answer; for the tasks that
   ask for a boxed value, the same paired table with "identical to the reference" in place of "right by the grader";
   the families with a gain beyond twice its standard error; and `PASS` or `FAIL` with the reasons. The wrap-up units
   can only gain where answers are lost to the limit, so they are judged there: over all families the gain of an edit
   that works is, on a model that finishes most answers, within the difference between two runs of the same model. A
   fit passes if, in the families that lose answers to the limit taken together, the share cut with the thinking open
   falls to a third of what it was or less and the gain is positive; no family is worse than the starting model by
   more than twice its standard error, and the families whose answers are written out, taken together (where an edit
   that also acts inside answers shows first), by no more than twice theirs; in no family are fewer answers identical
   to their reference, by more than twice the standard error; at least four in ten of the late closes of boxed tasks
   go on with the stem (otherwise the model writes its answer out at length after the close and runs out again); no
   more than one answer in a hundred ends inside the thinking, and no more than one in eight closes its thinking
   twice. A family that reads worse on its 192 or 384 dev answers need not be worse: the starting model's dev answers
   are a single draw that every fit is compared with, and two runs of one model differ by several points on that many
   answers (in our runs multiple choice read 5 points down for two fits on 192 answers, and 2 points up for one of
   them on 768, standard error 1.7). So if the only reasons for a `FAIL` are families worse than the starting model,
   measure those families again before giving the fit up: `run.py recheck --name <name> --family <family> --gpus
   0,1,2,3` generates eight answers a problem for both models and says whether the family is still worse by more than
   twice its standard error; if it is not, the family does not count against the fit. Held-out numbers do not replace
   this step: we have seen usable fits fail it badly. **Upload the first fit that passes at once** (step 11); it
   replaces the form edit of step 4, which it contains. Fits differ a good deal from one another, so do not stop
   there: until 50 minutes before the deadline keep fitting further seeds on GPUs that are free, evaluate every usable
   fit, and replace the upload whenever another fit passes and is better: the larger gain in the families that lose
   answers to the limit, provided that the written-out families together are not lower than for the fit it replaces by
   more than their standard error; among fits within one standard error of each other, the one whose late closes go on
   with the stem more often (an answer stated at once is right far more often than one written out at length after a
   late close). If no usable fit passes, fit more seeds; if time runs short, upload as a last resort the fit with the
   largest gain in those families among those the command allows as such (a gain there, no family worse by more than
   twice its standard error) and say in the report which checks it failed. If there is none, the form edit of step 4
   stays as the delivery; if that did not pass either, upload nothing and say so.
10. **Confirm once**, if at least 80 minutes are left when the comparisons of step 9 end (more candidates are worth
   more than the confirmation): `run.py evaluate --name <chosen> --split holdout --gpus 0,1,2,3`. Choose nothing on
   the holdout; report it whatever it shows.
11. **Deliver.** `run.py report --name <chosen>` assembles `$WSP_WORK/wrapup/REPORT.md` with every table measured; add
   in your own words what was targeted, which model you started from and why, what the tables of steps 3 and 4 and the
   replay of step 6 showed, the selectivity numbers of step 8, what did not work, and the holdout. `run.py deliver
   --name <chosen> --task task.json` packages the edit by the upload protocol (the toolkit's packer), runs
   `axonry-check` on the package and uploads only if it passes (the toolkit's uploader).

## Time
Generation dominates, and a model that finishes most answers generates faster than one that does not. On four GPUs the
fit traces take 20 to 30 minutes and the answers to the form problems 7; the form fit takes 3 minutes on one GPU and
its check on real answers 10 (both models); the replay takes 5 to 25 minutes; a wrap-up fit takes 45 to 75 minutes on
one GPU, depending on the rounds it needs; the starting model's dev answers take 15 to 25 minutes on four GPUs (longer
on fewer), each edited model's about as long. The form edit should be uploaded within the first hour and a first edit
with both parts within three and a half. The time after it goes to further seeds and their comparison (step 9). A
recheck of one family takes 15 to 30 minutes, and the holdout 40 to 70 (starting model and edit). Keep the last 40
minutes for the report and the final upload, and skip the holdout rather than miss the upload.

## What to expect
**The form.** In three runs on a model that had been taught to finish, and one on another, each on its own draw of
problems: before the edit, about 19 in a hundred of the right answers to the chosen word problems were written
differently from their reference (some 120 answers a run: a percent sign in 65 to 76, decimal zeros in 40 to 46, a
currency sign in 8 to 12, a unit in 1 to 3). The fit was usable at the first seed every time. On held-out answers the
reference's token went from a probability of 0.19 to 0.30 at the decisions a rewrite changes to 0.96 to 1.0, for the
percent sign, the decimal zeros and the currency sign alike (a unit after the number, of which a fit sees only one to
three, was deleted in the two runs where one was held out). No box that was identical to its reference lost a fifth of
its probability (42 to 51 of them a run, word problems, other math and multiple choice), KL 0.0002 to 0.0003, and the
units wrote less than a ten-thousandth of their layers' output elsewhere. On real answers to 200 word problems kept
apart, answers identical to their reference rose from 61.8% to 74.0% in one run and from 70.8% to 79.8% in another
(+12.2 and +9.0, standard error 2.4); the answers right but written differently fell from 50 and 26 of 400 to none;
right by the grader stayed where it was (74.2% to 74.0%, 77.2% to 79.8%), as did the length of the answers. On the dev
problems of every family the edit moved the answers right by the grader by -0.5 and +2.9 points in two fits (standard
errors 1.4 and 1.2), which is what two runs of one model differ by. Expect the gain where the references are bare
numbers, about three points on word problems taken at random, and nothing elsewhere. One side effect is known: the
percent sign is deleted in the other math family too, where a reference now and then carries one (4 in a thousand of
the practice references); the units cannot tell those apart.

**The limit.** Answers that used to run out now end with the model's best answer, right in a fair share of cases; the
gain is largest in the families with the most cut answers. On two models that had been taught to finish, between a
seventh and a fifth of the math answers still ran to the limit with the thinking open, and on one of them hardly any
answer of the other families; an immediate answer at 90% of the budget was right for one in six to one in four of the
math answers that got there, and where written-out answers still got there, three in four were right once the thinking
was closed: a ceiling of a few points in math and one or two in multiple choice. On real answers to unseen problems,
fits that kept the model unchanged elsewhere ended nearly every math answer that had run out and gained 3 to 5 points
in math, with arithmetic and multiple choice within the difference between two runs of the same model. In a full run
of both parts on that model (seven seeds, three of them usable) the three fits that were evaluated, two usable ones
and the nearest miss, all passed step 9 with the form units in the same edit: math +7.3, +6.5 and +10.4 by the grader
(standard errors 1.8 to 2.3) and +4.9, +5.2 and +8.3 counted as identical to the reference, the share of math answers
cut with the thinking open down from 22.9% to between 2.3% and 3.4%, and word problems identical to their reference
+5.7, +9.4 and +6.2 (standard errors 3.1 to 3.5). The written-out families came out three to five points down on a dev
set half the size of the one used here, and that was the draw: with eight answers a problem the same fit was level
with the starting model (0.3 points down, standard error 1.0). Two runs of one model on the same written-out problems
and seeds differed by five points. This is why the dev set now holds twice as many problems of these families, why
step 9 judges the wrap-up units where they can gain, and why it asks for twice the standard error before it calls a
family worse. Accuracy among answers that finished by themselves is unchanged, because below the stop depth the model
is the starting model.

Neither part makes the model think better. One makes it write down the answer it has as the reference would; the other
makes it answer when thinking has stopped paying.
