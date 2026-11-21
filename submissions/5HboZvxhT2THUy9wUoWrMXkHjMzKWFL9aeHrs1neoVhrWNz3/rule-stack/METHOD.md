# Rules written into units: close the thinking in time, state the result, keep to the marks a request names; and the keep-verifying direction removed where the start still has it

This method stacks on a published edit, called the start below (section 3 says which). It adds up to fourteen
rewritten MLP units (`set_rows` on `gate_proj` and `up_proj`, `set_cols` on `down_proj`), in pairs, and it removes
one direction from what four attention layers write (`dir_ablate` on `self_attn.o_proj`). Each pair is a rule on the
next-token logits that holds only where the model's own state marks a situation:

- **an exit pair** at layer 6 and **a break pair and a commit pair** at the last layer, for answers whose thinking
  never ends: they reach the token limit and score nothing;
- **up to four guard pairs** at the last layer, for requests with an instruction about a mark: no commas, at least N
  highlighted sections, at least N placeholders in square brackets, the whole response in double quotation marks.

The removed direction is the one along which the model's state differs between going on to verify a result it has
already stated and closing its thinking. It is the edit of the finished entry 0009538689-0017, derived again here on
this method's start and put on top of the pairs (sections 1.4 and 2). A start whose own chain of edits already
removes that direction keeps it as it is: the builder measures this and leaves the step out. The parent named in
`entry.json` is such a start, so on it the run delivers the pairs alone.

Nothing is trained. Computed in the run, from the start's own answers to practice problems: every read (ridge
regressions, discriminant directions, constant directions), every threshold, the write directions (linear solves
through the final norm and the output head), which units are rewritten, and the removed direction (a difference of
two means). Fixed by the method: the layer of the exit pair, the sizes of the pushes, the constants of the gates and
the four attention layers of the direction; sections 1 and 2 give the reason for each.

The figures below are the author's measurements on practice problems, paired by problem and seed. Unless said
otherwise they were measured on the published edit 0009538688-0021, a model of this line whose answers no longer
loop; where a figure belongs to another published edit, it says so. The parent named in `entry.json`, the finished
entry 0009552724-0008, is a later edit of the same line: section 6 has the author's run of these steps on it, and
the steps say what to measure first and when to stop.

Where the parts come from, and how the exit works. The exit pair reads how long the thinking has run, a clock, and
pushes towards the closing tag harder the longer it runs, so that the model closes by itself. It does not read
whether a run is stuck or repeats itself; section 7 says what such a read gave in the author's runs. The break and
commit pairs then act only at the closes the exit caused. The exit, break and commit pairs are the units of this
miner's entry 0009545597-0008,
and the comma guard is the rule of this miner's finished entry 0009543609-0016 (the same GitHub account as this
entry). The guards for the marks that the sampler takes back are new here. The concluding direction is the edit of
the finished entry 0009538689-0017, derived again in this run.

## 1. The behaviours to change, and why they are there

### 1.1 Thinking that never ends

**An answer that reaches the token limit scores nothing, and most such answers never closed their thinking.** 19%
of the mathematics practice answers are cut at the limit: 13% with the thinking still open, the rest in an answer
part that turned into more reasoning. The other practice sources hardly reach the limit.

**Cutting the thinking by hand does not help.** With the thinking closed by hand at a fixed depth and the same model
left to go on (mathematics, answers still thinking at that depth, split by what they did when left alone):

| closed by hand at | never-closing answers: right, answer left to the model / opened with `\boxed{` | answers that would have closed later and been right: the same |
|---|---|---|
| 12,000 tokens | 1% / 24% | 41% / 75% |
| 20,000 tokens | 3% / 21% | 44% / 77% |
| 28,000 tokens | 7% / 26% | 43% / 100% |

After a close by hand the model goes on reasoning in the answer part and reaches the limit again. Made to open
with the box it is right in a fifth to a quarter of the cases, and the answers that would have finished by
themselves lose a quarter of theirs. A hard cut needs to know which answers will never close, and a length read
cannot tell: at 20,000 tokens four in ten of the answers still thinking will finish, almost half of them right.

**A pressure that rises slowly makes the model close by itself.** The exit pair (section 2) adds, at layer 6, a
push towards the closing tag that grows over several thousand tokens once a length read passes a threshold. On 800
mathematics practice problems that no fit had seen (2 samples each, paired by problem; budget 0.16):

| | before | exit pair alone | with break and commit |
|---|---|---|---|
| thinking never closes | 13.2% | 0.1% | 0.1% |
| cut at the limit | 19.2% | 4.8% | 1.0% |
| finished without a result | 0.9% | 6.6% | 0.9% |
| right | 63.5% | 65.6% (+2.1 +- 1.0) | 66.6% (+3.1 +- 1.1) |

With the budget 0.12 the same two builds gain +1.9 +- 1.0 and +2.7 +- 1.0 (cut at the limit 8.4% and 3.1%). On the
hardest level of these problems the right answers go from 29% to 34% with all three pairs.

Under the pressure a good part of the answers close while it is still small: of the closes the pair takes part in,
about four in ten happen while it lets through less than a fifth of its write. There the push only tips a decision
the model was already close to, at the end of a line, and what follows is an answer like any other.

**Where the pressure enters decides whether it works.** The same pair at the same budget (0.12), with its read
fitted at the layer it sits in, on dev problems (mathematics, 2 samples):

| layer of the pair | answers cut at the limit (before: 18.0%) | mean tokens per answer (before: 14,700) |
|---|---|---|
| 6 | 7.8% | 14,800 |
| 8 | 25.0% | 16,300 |
| 10 | 13.3% | 15,000 |
| last | 18.8% | 15,300 |

Deeper in the network the pair still closes the thinking, but the close is one the model did not prepare: the
answer part runs on to the limit as after a cut by hand. Entering early, the push passes through the rest of the
network and the model brings its line to an end first.

**What stays wrong is what follows a close that needed the full pressure.** Those answers are not finished. The
model keeps reasoning inside the answer part, the pressure closes it again, and only the text after the LAST closing
tag counts as the answer: one line without the boxed result, or more reasoning until the limit. With the exit pair
alone, late answers with one closing tag are right 57% of the time, with two 27%, with three or more 13%; over a
quarter of all mathematics answers have several tags. Pushing only once would be worse: continued by the model
without the pair from just before their second tag, the answers with several tags are right 15% of the time (20% as
generated) and more than half of them run to the limit.

**So the result has to come at once after a pushed close.** The break and commit pairs raise the opening of a boxed
answer right after a closing tag that the pressure pushed, and nowhere else. Fitted on the model without the exit
pair, at closes inserted by hand, such a pair reaches about half of the real closes in generation: the real ones
come earlier in the rise and with the exit pair's write in their context. Fitted on answers generated by the model
that carries the exit pair, it brings the answers cut at the limit and those without a result to the figures of the
table above. Where the model closed by itself the commit must stay silent: restarting the model's own answers with
the box cost 2.8 points in mathematics (9 right answers lost, 2 won of 250).

**Suspected cause.** The model closes its thinking at the end of a line that states a result it accepts. How long
the thinking has run takes no part in that decision, although the length is marked in its state (the clock of
section 2 reads it). Where the thinking passes near such a line, a small push is enough. Where it is far from one,
a close leaves the model with an answer part that it treats as more thinking.

### 1.2 Marks the request asks for, and the sampler that takes them back

On a model that already closes its thinking (0009538688-0021 with the three pairs above), with the profile's
sampler, the answers to practice requests (`ifsynth`, `tuluif`) follow the instruction itself (the toolkit's
checker, that instruction alone) this often:

| instruction | answers | followed |
|---|---|---|
| at least N highlighted sections (`*...*`) | 256 | 62% |
| at least N placeholders (`[...]`) | 248 | 73% |
| the whole response in double quotation marks | 270 | 87% |

The model has read the instruction: its thinking restates it. That is where the failure comes from. The profile's
sampler carries a presence penalty of 1.5: before every draw it takes 1.5 logits from each token that has already
occurred in the generated text. The generated text begins with the thinking, and a thinking that restates "i.e.
\*highlighted section\*" or "such as [address]" has written the mark: in every one of the 503 answers to the first
two kinds of request the thinking holds it. So from the first position of the visible answer, every token that
holds the mark is 1.5 logits less likely than the model itself has it. The answer then avoids exactly what the
request asks for.

Two measurements at decoding (the toolkit's rule: measure a decoding change at decoding before writing it into the
weights), same requests, same seeds; share of the answers that follow the instruction:

| drawing | highlighted sections | placeholders | quotation marks |
|---|---|---|---|
| the profile's sampler | 62% | 74% | 87% |
| the same without the presence penalty | 76% | 86% | 90% |
| the profile's sampler with +1.5 on the mark's two tokens | 80% | 92% | 91% |

Giving the mark back its 1.5 logits repairs more than removing the whole penalty does, because the penalty is not
only harmful: without it the answers to "all capital letters", "repeat the request" and "two different responses"
get worse (by 9, 9 and 11 points), since the penalty also keeps the model from repeating itself. So the penalty is
to be undone for the mark, in the answers that are asked to carry it, and left alone everywhere else. A constant
bias on the whole run cannot do that either: at +3 on the asterisk tokens the thinking itself fills with asterisks,
15% of the answers never close it and the other instructions of the same requests are followed 56% of the time
instead of 81%. The weights can: the model marks linearly, at the last layer, that it is writing the answer to a
request with the instruction (section 2), and a unit that fires there can add a constant to the logits of the
mark's tokens.

### 1.3 The comma the request forbids

This is the same kind of slip with the opposite sign. On 0009538688-0021, 44% of the answers to requests that forbid
commas fail, nearly all by one to three commas. The thinking shows that the model has read the instruction; while
it writes, the habit token wins a few draws. The presence penalty already works against the comma here and is not
enough. A published run of this rule alone (the edit 0009543609-0016) measured on 50 requests kept out of the
building, 4 samples each: 61.0% -> 93.5% of the answers pass (+32.5 +- 5.4).

**Why the last layer for the guards, the break and the commit.** Their writes are defined on the final residual
`h`: a solve against the output head as the final norm scales it, with `rms(h)` measured where the gate fires. A
last-layer unit changes the output of its own position only, so what it does is known before any generation. The
playbook warns that the last layer's MLP is entangled with next-token prediction: that entanglement is the lever
here, and steps E and F measure its price where the units should stay silent.

### 1.4 Verifying after the result is found

**Answers state their result and go on checking it.** The finished entry 0009538689-0017 measured this on the first
model of this line and removed it with one direction per layer. The models of this line have changed since (their
thinking no longer loops, and the pairs above act on what is left), so the direction is measured again on this
method's start. The toolkit's `conclude` contrast takes a right, finished answer of the start and cuts it where the
thinking first states the right result. From there one branch closes the thinking and gives the model's own final
answer; the other is the model's own next text, which is more verifying. Both share the prefix. Per layer, the
direction is the mean residual over the verifying branch minus the mean residual over the closing branch.

Measured with this method's builder on 0009538688-0021: of the 880 trace answers to word problems and mathematics
problems, 589 are right and finished, and 128 pairs come from 105 of them. At layers 11, 15, 19 and 23 the
split-half stability of the direction is 0.98 to 0.99, the two branches lie 5.8 to 6.7 standard deviations apart
along it, and it orders every pair correctly. Its size is 0.33 to 0.58 of the residual's own at those layers.

**Why it is added to the pairs.** The exit pair acts late, on thinking that would never end. The direction acts
from the first stated result on, in every kind of problem: with it the model returns to its result less often,
which matters most where the answer has to follow a request to the letter and every further round of checking is a
chance to drift from it.

Measured on 0009538688-0021 with the exit pair and three guard pairs in place: the same stack with the direction
removed, against the stack without it, on 2,000 practice problems outside dev, holdout and every list a builder
read (2 samples, paired by problem and seed):

| source | problems | without | with | gain | mean tokens per answer |
|---|---|---|---|---|---|
| mathematics | 500 | 60.0% | 62.9% | +2.9 +- 1.2 | 16,200 -> 13,900 |
| word problems | 500 | 77.4% | 77.5% | +0.1 +- 1.2 | 6,700 -> 4,800 |
| multiple choice | 400 | 45.8% | 46.5% | +0.8 +- 1.7 | 10,800 -> 8,600 |
| `ifsynth` | 200 | 84.0% | 86.8% | +2.8 +- 2.2 | 4,400 -> 4,100 |
| `tuluif` | 400 | 79.6% | 81.1% | +1.5 +- 1.6 | 3,600 -> 3,200 |
| all | 2,000 | 67.8% | 69.3% | +1.5 +- 0.7 | 9,100 -> 7,400 |

The answers get shorter in every source, by a fifth over all. In mathematics the gain comes from the answers the
exit pair has to close: with the direction removed, fewer of them need several closes (32% -> 21% of the mathematics
answers) and fewer end without a result (10.1% -> 5.5%). Against the start itself the whole stack is at +2.5 +- 0.7
on this set (mathematics +4.2 +- 1.3, multiple choice +4.4 +- 1.6, `ifsynth` +3.0 +- 2.2, word problems +0.7 +- 1.3,
`tuluif` +0.5 +- 1.6), with the cut share going from 6.2% to 0.6%.

## 2. The units

Each part is a pair of units with the same reads; the second has a shifted gate and the opposite write, so that
the pair's write stops growing (a flat top). A unit computes `down * silu(gate . x) * (up . x)` from its layer's
MLP input `x`; `m` is a read with `m . x ~ 1` (the MLP has no bias).

**The clock (`derive/clock.py`).** A ridge regression of log(tokens generated so far) on the MLP input of layer 6.
Position by position it is noisy (error 0.28 in log units). The first position at which a whole answer crosses a
threshold is regular: `clock.py screen` measures, on problems held out from the fit, how many never-closing
answers cross a threshold in time, how many right late answers and how many early-closing ones do, for a series
of thresholds, and chooses one (the budget: the share of the self-closing answers that reach the threshold
before their own close).

**Exit (`derive/exit.py`, layer 6).** Gate: one smooth step, `sharp * (clock + lam * thinking - offset)`, where
`thinking` is the discriminant read of thinking against answer positions, centred on its median in late thinking.
Up: the constant read. Write: at that layer, the vector that at the final residual raises the closing tag and holds
the tokens that end the turn or open a thinking block (a frequency-weighted least-squares solve through the final
norm and the head). Fixed constants:

- `lam = 0.25`: the thinking read lowers the gate in answers without adding much of its own noise to the clock.
- relative change of the layer's gate matrix 0.30 (the limit of an edit is 0.5 per matrix): this sets `sharp`.
  The ridge direction has a large norm, so the step is wide: from open to saturated the length grows by about
  half. That width is what lets the model choose its line.
- push 60 logits where the step has saturated: measured with a hook before any unit was written, of a push meant
  to be 40 logits 27 to 30 arrive at the output, and pushes under 20 arrive at a third.
- flat top at 2 gate units, up scale 4: numerical choices that keep both units' rows of similar size. Below the
  threshold the pair's write is slightly negative (at most 8% of the full push): shortly before the threshold it
  leans against a close.

**Break and commit (`derive/commit.py`, last layer).** A closing tag counts as pushed when the exit pair let through
at least a fifth of its full write somewhere in the 32 positions before it, as the model's own when the pair let
through less than a fiftieth since the previous tag. Break: at a closing tag, raises the `\n\n` token (12 logits;
after its own closes the model writes it anyway, after a pushed one about half of the time). Commit: at that `\n\n`
after a pushed tag and at the backslash after it, where the prompt contains `\boxed`, raises the backslash and
`boxed` (14 logits: the model's own log-probability of the backslash there is about -5). Both are fitted on the
exit model's own answers, read through that model; in answers whose first close was pushed, the tag is also
inserted at up to two earlier positions where the pair let through half of its write, to add positives. The other
positions collected are the negatives: the two positions after the model's own closes, thinking positions, the
answers the model wrote by itself, every backslash, blank line and brace in them. The threshold of each pair leaves
out the 0.1% of them that look most like its own positions, and the builder reports where the pair still fires.

**Guards (`derive/guard.py --kind ...`, last layer).** One builder, four kinds:

| kind | instruction | family of tokens | shift where the pair fires |
|---|---|---|---|
| `comma` | `punctuation:no_comma` | every token whose text holds a comma | -10 |
| `highlight` | `detectable_format:number_highlighted_sections` | every token whose text holds an asterisk | +1.5 |
| `placeholder` | `detectable_content:number_placeholders` | every token whose text holds `[` | +1.5 |
| `quote` | `startend:quotation` | every token whose text holds `"` | +1.5 |

The instruction itself names the family. Nothing is searched or ranked.

- Gate (the toolkit's `wsp.gate.build`): one linear read of the last layer's MLP input. Its positives are the
  answer positions (after the thinking has closed) of requests whose instruction list holds the instruction. Its
  negatives are the thinking of the same requests, where the model may plan with the mark, and all positions of
  requests without the instruction and of the other practice sources; the threshold is above 99% of them.
- Write: the vector that moves the logits of the most frequent tokens of the family by exactly the shift and every
  other token as little as possible (a frequency-weighted least-squares solve with equality constraints, through
  the final norm and the head). For the comma these are the 24 comma tokens that answers in general use most; for
  a mark to give back, the 8 tokens of the family that the answers to requests with the instruction use most (the
  opening and the closing form of the mark).
- Size of the shift. +1.5 is the presence penalty: the pair gives back what the sampler takes and no more. Measured
  with the units built at three sizes, on requests kept out of the building (4 samples): highlighted sections
  followed 59.5% -> 76.5% at +1.5 (+17.0 +- 4.3), 75.0% at +3, 49.0% at +5 with 64% of the answers running to the
  limit in repeated marks; placeholders 68.5% -> 83.3% at +1.5 (+14.9 +- 4.0), 82.7% at +3 with 8% at the limit,
  78.0% at +5 with 58% at the limit. The comma shift of -10 is a ban: the instruction allows no comma at all.

The requests with an instruction are split by a hash of their id: 60% to build its pair, 40% kept out to test it
(`derive/guard_sets.py`).

**The concluding direction (`derive/conclude.py`).** Pairs: the start's own right, finished answers among the
traces of step A to practice problems with a known result (`gsm8k`, `math`), each cut where its thinking first states
that result (`wsp.directions`, contrast `conclude`); at most 128 pairs, one per answer. Direction per layer: the
mean residual over the verifying branch minus the mean residual over the closing branch, normalized. Edit:
`dir_ablate`, side `out`, alpha 1, on `self_attn.o_proj` of layers 11, 15, 19 and 23. These are the full-attention
layers from the middle of the network up, and the layers and strength the finished entry found best on its start
(two of the layers alone, all six attention layers, half and twice the strength were worse there); alpha 1 projects
the direction out of what those layers write and adds nothing. The builder prints, per layer, the split-half
stability of the direction, the separation of the two branches along it and the share of pairs it orders correctly.
It stops with `NO CONCLUDING DIRECTION` when the traces give fewer than 48 pairs or when a layer is under 0.9 on
the stability or on that share.

It also stops when the start carries this edit already. For each of the four layers the builder compares the
direction with every direction that an edit of the start's own chain projects out of the same attention output at
a strength of 0.5 or more, and prints the largest cosine (`largest_cosine_with_a_direction_the_start_already_
removed_there`). A cosine of 0.7 or more at any layer means the direction was removed there before, and it is not
removed a second time. Measured by the author: on 0009538688-0021 and on 0009547524-0008 the largest cosine is
0.20 and 0.22 (the chain's first edit removes another direction at layers 15 and 19), and the direction is built.
On the parent of this entry it is 0.96 at layers 15 and 19, where the parent's own edit removes this direction at
strength 0.75, and the step is left out. The finished entry that made that edit reports the same from its own
search: four attention layers instead of two, or a higher strength, were no better on average and worse on
mathematics practice problems.

**Which units.** In each layer, among the units whose largest write at the sampled trace positions is in the
lowest tenth, those with the smallest mean write; never a unit that the start's own chain of edits wrote or that an
earlier pair of this run took (`--avoid`).

Every builder prints what to check before any generation, and stops by itself when the start does not show the
behaviour (`NO CLOCK`, `NO EXIT`, `NO COMMIT`, `NO GUARD (<kind>)`, `NO CONCLUDING DIRECTION`) or when a computed
value is not finite.

## 3. Setup (first 20 min)

**The start is the entry's parent.** The bundle holds every start under its id, `<id>/edit/`; `task.json` lists
them under `starts`: the entry's parent with the role `parent` and, when the champion is another edit, that one
with the role `champion`. This method starts from the parent in both cases, and the champion's directory is not
copied: the parent is the edit this entry was prepared for (section 6 has the author's run on it).

On the agent machine, from the unpacked bundle (the brief says to copy `task.json` to the sandbox):

```
P=$(python3 -c "import json; print(next(s['sid'] for s in json.load(open('task.json'))['starts'] if s['role'] == 'parent'))")
echo "parent: $P"
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r $P sandbox:/workspace/work/            # gives /workspace/work/$P/edit/edit.json
```

(The start is copied under its own id, next to `task.json`: `axonry-check` looks for it there.)

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`):

```
cd /workspace/work && mkdir -p w
cat > w/env.sh <<'EOF'
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
P=$(python -c "import json; print(next(s['sid'] for s in json.load(open('/workspace/work/task.json'))['starts'] if s['role'] == 'parent'))")
C=sparent                                  # the case of this run: it stands on the parent
T="w/tr_0.jsonl w/tr_1.jsonl w/tr_2.jsonl w/tr_3.jsonl"
XT="w/x_0.jsonl w/x_1.jsonl w/x_2.jsonl"
GUARDS=""                                  # the guard edits that were built; step C sets this line
KEPT=""                                    # the guard edits their tests kept; step E sets this line
BEST=S1                                    # the edit to deliver; steps F and F2 set this line
EOF
. /workspace/work/w/env.sh
python -m wsp.case init $C --profile qwen35-0.8b-think --parent $P/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 320 --holdout 640 --subset holdout320=holdout:320
python -c "from wsp import case; case.model_path('$C','base')"    # build the start ONCE, before any GPU job
```

**Run `. /workspace/work/w/env.sh` first in every new shell, ssh call and tmux window**: the commands below rely on
`$P`, `$C`, `$T`, `$XT`, `$GUARDS`, `$KEPT`, `$BEST` and the working directory (bash; names in these variables
are split on the spaces). The case `sparent` stands on the parent `$P`. Read the `REPORT.md` next to the start's
`edit.json`. Rules for all long work:

- Every generation job runs detached (`nohup ... &` as written below, or `tmux new -d`) with its own
  `CUDA_VISIBLE_DEVICES`, one job per GPU (a `wsp.case run` given two GPU ids splits its problems over both). Its
  output file appears only when the job ends.
- Before each start, look at `nvidia-smi`: the GPUs you name must hold no memory; the toolkit refuses a GPU that
  another job holds. Stop a leftover by its PID (`nvidia-smi` lists it), never by a name pattern.
- Check the jobs every 10 to 15 min (`tail` the log) and keep `PROGRESS.md` current. A generation job whose log
  ends in a traceback is started again once. If a builder ends in a traceback or says a value is not finite, read
  it, start it again once, and if it fails the same way follow the rule of its step for a builder that stops.
- Use only `practice:` data and the generations of the start and of this run's own edits.

## 4. Steps

**Step A: the start on dev, its traces, the requests of the guards (about 60 min)**

```
python derive/sets.py --case $C --out w                    # w/ids_0..3.json: 800 practice problems, never dev / holdout
python derive/guard_sets.py --case $C --out w              # build lists of the four kinds, w/if_build.json, w/tests.json
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C base dev --n 2 > dev_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python derive/draw.py --case $C --ids w/ids_0.json --out w/tr_0.jsonl > t0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case $C --ids w/ids_1.json --out w/tr_1.jsonl > t1.log 2>&1 &
```

When `t0.log` and `t1.log` each end with a `[gen] ...` line (each such line is the generation rate of one GPU, see
section 5):

```
CUDA_VISIBLE_DEVICES=2 nohup python derive/draw.py --case $C --ids w/ids_2.json --out w/tr_2.jsonl > t2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case $C --ids w/ids_3.json --out w/tr_3.jsonl > t3.log 2>&1 &
```

And on GPUs 0 and 1, when the dev run has ended:

```
CUDA_VISIBLE_DEVICES=0 nohup python derive/draw.py --case $C --ids w/if_build.json --out w/if_build.jsonl --n 2 --salt rb --max-tokens 8000 > t4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python derive/draw.py --case $C --ids w/tests.json --out w/tests_base.jsonl --n 4 --salt rt > t6.log 2>&1 &
```

and, when `t4.log` ends with its `[gen] ...` line, the measurement of section 1.2 on this run's start (the building
requests once more, without the presence penalty):

```
CUDA_VISIBLE_DEVICES=0 nohup python derive/draw.py --case $C --ids w/if_build.json --out w/if_build_nopp.jsonl --n 2 --salt rb --max-tokens 8000 --set presence_penalty=0.0 > t5.log 2>&1 &
python derive/pair.py w/if_build.jsonl w/if_build_nopp.jsonl --by instruction          # when both files exist
```

`draw.py` is `python -m wsp.gen` on the case's start or, with `--edit NAME`, on one of its edits; `--max-tokens` only
shortens a drawing and `--set` changes a sampler setting for that drawing alone. The traces `$T` are drawn at the
full token limit (the exit is about the answers that reach it), 800 problems x 2. The last reading goes into
REPORT.md; no step depends on it (the builders measure for themselves how often the start breaks each instruction).

Read `cases/$C/results/base__dev.json`: `summary.acc`, `summary.cut_share`, `summary.by_source`, `summary.kinds`.
**If `summary.cut_share` is under 0.02, this start hardly runs to the limit: there is no exit and no commit in this
run. Skip step B and the exit of step C; everything else stays (with `S1` holding the guards alone).**

**Step B: the clock (about 25 min; needs all four files of `$T`)**

```
CUDA_VISIBLE_DEVICES=0 nohup python derive/clock.py reads --case $C --runs $T --out w/clock > reads.log 2>&1 &
```

When `reads.log` ends with `READS DONE` (forward passes over half of the traces):

```
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup python derive/clock.py series --case $C --runs $T --out w/clock --shard $i/4 > s$i.log 2>&1 & done
```

When each of `s0.log` to `s3.log` ends with `SERIES DONE`:

```
python derive/clock.py screen --out w/clock
```

It prints one line per read and ends with `SCREEN layer 6 read ridge budget B: ...`, the choice the exit will use.
**If `reads.log` or the screen ends with `NO CLOCK`, the length of the thinking is not linearly marked on this start
(or the traces hold too few answers of a kind): there is no exit and no commit in this run; go on with the guards
of step C.** (Start the series jobs only when the GPUs are free; the drawings of step A have priority.)

**Step C: build the exit, the guards and the concluding direction (about 20 min)**

On GPU 1, beside the commands below (it needs the four files of `$T` and nothing else):

```
CUDA_VISIBLE_DEVICES=1 nohup python derive/conclude.py --case $C --rows $T --name CC > build_CC.log 2>&1 &
```

On GPU 0, one command after the other:

```
CUDA_VISIBLE_DEVICES=0 python derive/exit.py --case $C --runs $T --clock w/clock --name X > build_X.log 2>&1
python derive/sets.py --case $C --out w --again $T --shards 3
CUDA_VISIBLE_DEVICES=0 python derive/guard.py --case $C --kind comma --name G_comma --runs w/if_build.jsonl $T --exclude w/tests.json --avoid X
CUDA_VISIBLE_DEVICES=0 python derive/guard.py --case $C --kind highlight --name G_highlight --runs w/if_build.jsonl $T --exclude w/tests.json --avoid X G_comma
CUDA_VISIBLE_DEVICES=0 python derive/guard.py --case $C --kind placeholder --name G_placeholder --runs w/if_build.jsonl $T --exclude w/tests.json --avoid X G_comma G_highlight
CUDA_VISIBLE_DEVICES=0 python derive/guard.py --case $C --kind quote --name G_quote --runs w/if_build.jsonl $T --exclude w/tests.json --avoid X G_comma G_highlight G_placeholder
ls cases/$C/edits/X.json cases/$C/edits/G_*.json
```

Set the `GUARDS=` line of `w/env.sh` to the guard names that `ls` printed (for example
`GUARDS="G_comma G_highlight G_placeholder G_quote"`), source it, and stack what was built:

```
python derive/stack.py --case $C --name S1 --edits X $GUARDS          # without X if there is no exit
```

- `build_X.log` ends with the measurements (also `cases/$C/edits/X.build.json`): `never_closing_answers_reached`
  (expect 0.8 or more), `of_them_closing_before_12k` (expect 0), `exit_gate_relative_size` (about 0.30). If it stops
  with `NO EXIT`, it says why; with "raise --budget", run it once more with `--budget` one step up (0.16, then 0.2).
  If it still stops, there is no exit and no commit in this run. `sets.py --again` lists the problems to draw again
  for the commit (those whose thinking ran long, never closed or closed twice, and 120 others).
- Each guard builder prints how often the start breaks its instruction, `recall` and `neg_fire` of the gate, the
  shift of the family's tokens and the largest shift of any other frequent token (also in
  `cases/$C/edits/<name>.guard.json`). Measured by the author: gate `recall` 0.67 to 0.99, `neg_fire` under 0.01,
  `rel_gate` 0.11 per pair. These values are for comparison only: unless a builder stops by itself, go on.
- A guard builder stops with `NO GUARD (<kind>)` when the traces hold too few answers to requests with the
  instruction, when the start breaks it in fewer than one answer in twenty, or when the gate reaches under half of
  the answer positions. Then that kind has no pair on this start. `--avoid` naming an edit that was not built is
  harmless.
- `build_CC.log` ends with the measurements of the direction (also `cases/$C/edits/CC.build.json`): the number of
  pairs, and per layer `split_half_stability`, `separation_sd` and `pairs_ordered_correctly`. Measured by the author
  on 0009538688-0021: 128 pairs, stability 0.98 to 0.99, separation about 6 SD, every pair ordered correctly. If it
  ends with `NO CONCLUDING DIRECTION`, there is none on this start, or the start carries it already (section 2),
  and step F2 is left out. The direction is not
  part of `S1`: it is added in step F2, on top of what steps E and F have chosen.
- If neither `X` nor any guard was built, the pairs have nothing to add on this start. If `CC` was built, stack
  `S1` from it alone (`--edits CC`), upload it as in step D, then compare it with the start on dev and on the test
  requests (`python -m wsp.case run $C S1 dev --n 2`, `draw.py --edit S1` on `w/tests.json`, `compare.py` and
  `pair.py` against `base__dev` and `w/tests_base.jsonl`), keep it by the rule of step F2 and go to step G. If
  nothing at all was built, the method has nothing to add: say so in REPORT.md and end the run as the brief says.
- `stack.py` puts the pairs into one spec and refuses two edits that write the same unit or a value that is not
  finite.

**Step D: admission check and first upload (by about 2 h)**

Write `/workspace/work/REPORT.md` first (what was done and found so far; `wsp.upload` reads it), then:

```
python -m wsp.upload $C S1 --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/${C}__S1/edit --task task.json
python -c "import json; print(json.load(open('$WSP_WORK/deliver/${C}__S1/edit/edit.json'))['parent'])"
python -m wsp.upload $C S1 --task task.json --report /workspace/work/REPORT.md
```

The third line must print the id of the start, `$P` (the toolkit writes it; do not edit it). The builders write
their values inline in the spec; the upload stores them in `edit.npz` (an older toolkit leaves them inline). Measured by the author: the exit pair changes the gate matrix of layer 6 by 0.30 of its norm; four
guard pairs change the last layer's gate matrix by 0.33, and with the break and commit pairs by about 0.40 (the
limit is 0.5 per matrix). If `axonry-check` reports a tensor above the per-tensor limit, stack again without
`G_quote` (the smallest effect), then without `G_placeholder`, and use that stack as `S1`. This first upload is a
safeguard; later uploads replace it.

**Step E: test S1; draw the exit's own answers (about 45 min)**

```
python -c "from wsp import case; case.model_path('$C','S1')"
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run $C S1 dev --n 2 > dev_S1.log 2>&1 &
python -c "from wsp import case; case.model_path('$C','X')"          # only with an exit, as the next line
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((i + 1)) nohup python derive/draw.py --case $C --edit X --ids w/again_$i.json --out w/x_$i.jsonl > x$i.log 2>&1 & done
```

When a GPU is free (at once on GPU 1 if there is no exit), the test requests on `S1`, with the seeds of
`w/tests_base.jsonl`:

```
CUDA_VISIBLE_DEVICES=1 nohup python derive/draw.py --case $C --edit S1 --ids w/tests.json --out w/tests_S1.jsonl --n 4 --salt rt > t7.log 2>&1 &
```

Readings:

- **The guards**: `python derive/pair.py w/tests_base.jsonl w/tests_S1.jsonl --by instruction`. Each line is the
  gain of `S1` on the test requests that carry one instruction, with its standard error; the lines that count are
  `punctuation:no_comma`, `detectable_format:number_highlighted_sections`, `detectable_content:number_placeholders`
  and `startend:quotation`. **A guard is kept if the gain on its line is above the standard error printed there.**
  Set the `KEPT=` line of `w/env.sh` to the kept guard names and source it.
- **The exit**: `python derive/compare.py $WSP_WORK/runs/$C/base__dev.jsonl $WSP_WORK/runs/$C/S1__dev.jsonl` prints,
  per source and over all, the gain with its standard error, the cut share, and how the answers of the second run
  end (several closing tags, opening with a box, no result). **The exit qualifies if the cut share of `S1` over all
  sources is at most 0.7 times the start's, `S1` is not below the start over all sources by more than twice the
  standard error printed there, and no single source is below it by more than three times its own.** The accuracy
  of one source moves by several points between two drawings at this size (64 problems per source, a standard
  error of 3 to 4.5 points). A rule at twice that error fails a harmless exit in about one run of five: in one of
  the author's two runs of these steps on 0009538688-0021, word problems read -7.0 +- 3.3 and multiple choice -9.4 +-
  4.4 while all sources together read -0.2 +- 1.7 and the cut share went from 5.3% to 0.5%. The cut share is the
  reading that counts here; the larger sets of sections 1.1 and 6 are the reading for the single sources.

**Step F: the commit; choose what to deliver (about 60 min; only with an exit that qualified)**

When the three `w/x_*.jsonl` files are there:

```
CUDA_VISIBLE_DEVICES=2 nohup python derive/commit.py --case $C --exit X --runs $XT --name K --avoid $GUARDS > build_K.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m wsp.case run $C base holdout320 --n 2 > hold_base.log 2>&1 &
```

`build_K.log` prints a line every 50 answers read and ends with the measurements (also
`cases/$C/edits/K.build.json`). There is no commit on this start if the builder stops with `NO COMMIT`, if
`commit_fires_after_the_models_own_closes` is above 0.1, or if `last_layer_gate_relative_size` is above 0.45. A
line `NO BREAK UNIT` means that pair was left out; go on. Otherwise:

```
python derive/stack.py --case $C --name S2 --edits K $KEPT            # K holds the exit pair too
python -c "from wsp import case; case.model_path('$C','S2')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C S2 dev --n 2 > dev_S2.log 2>&1 &
```

Then `python derive/compare.py` for the start against `S2`, and for `S1` against `S2` (the files
`$WSP_WORK/runs/$C/base__dev.jsonl`, `S1__dev.jsonl`, `S2__dev.jsonl`). Choose, write the choice into the `BEST=`
line of `w/env.sh` and source it:

1. **`BEST=S2`** if `S2` qualifies against the start by the rule of step E, no source of `S2` is below `S1` by more
   than twice its standard error, and over all sources the cut share of `S2` plus its share of answers without a
   result is not above that of `S1`.
2. Otherwise, or without a commit: **`BEST=S1`** if the exit qualified and every built guard was kept. If the exit
   qualified but some guard was not kept, stack `python derive/stack.py --case $C --name S3 --edits X $KEPT` and
   `BEST=S3`.
3. If the exit did not qualify (or there is none): `python derive/stack.py --case $C --name S3 --edits $KEPT` and
   `BEST=S3` (the kept guards alone; if every built guard was kept and there was no exit, that is `S1` itself).
4. If neither the exit qualified nor any guard was kept, the method does nothing on this start: leave the first
   upload, say so in REPORT.md and end the run as the brief says.

If `BEST` is not `S1`, upload it now (the four commands of step D with `$BEST` in place of `S1`); if `axonry-check`
refuses it, fall back to the last stack it accepted.

**Step F2: the concluding direction on top of `BEST` (about 30 min; only if `cases/$C/edits/CC.json` exists)**

Stack the direction on the edits `$BEST` is made of (`S1`: `X $GUARDS`; `S2`: `K $KEPT`; `S3`: as stacked in step F):

```
python derive/stack.py --case $C --name S4 --edits <the edits of $BEST> CC
python -c "from wsp import case; case.model_path('$C','S4')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C S4 dev --n 2 > dev_S4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case $C --edit S4 --ids w/tests.json --out w/tests_S4.jsonl --n 4 --salt rt > t8.log 2>&1 &
```

`S4` is compared with `$BEST` on dev and on the test requests. `S1` has both runs already and `S2` has its dev run.
What is missing is drawn on GPU 2, one after the other, with the same seeds:

```
CUDA_VISIBLE_DEVICES=2 python -m wsp.case run $C $BEST dev --n 2 > dev_$BEST.log 2>&1          # only for S3
CUDA_VISIBLE_DEVICES=2 python derive/draw.py --case $C --edit $BEST --ids w/tests.json --out w/tests_$BEST.jsonl --n 4 --salt rt > t9.log 2>&1    # for S2 and S3
```

When all of them have ended:

```
python derive/compare.py $WSP_WORK/runs/$C/${BEST}__dev.jsonl $WSP_WORK/runs/$C/S4__dev.jsonl
python derive/pair.py w/tests_$BEST.jsonl w/tests_S4.jsonl --by instruction
```

**The direction is kept (`BEST=S4`) if, on dev, `S4` is not below `$BEST` over all sources by more than twice the
standard error printed there and on no single source by more than three times its standard error, and if on the
test requests the `ALL` line of `pair.py` is not below `$BEST` by more than twice its standard error.** Otherwise `BEST` stays
as it is. This is a check that the direction does no harm on this start; the size of its gain is in section 1.4 and
cannot be read from sets of this size. If `BEST=S4`, write it into `w/env.sh`, source it, and upload it (the four
commands of step D with `S4`); if `axonry-check` refuses it, the earlier `BEST` stays the delivered edit.

**Step G: confirm on holdout, deliver**

```
python -m wsp.case freeze $C $BEST
python -c "from wsp import case; case.model_path('$C','$BEST')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C $BEST holdout320 --n 2 > hold_best.log 2>&1 &
```

(`freeze` wants a run of the edit: for an `S3` that was never run it stops with a message; go on.) If step F did not
run the start on holdout320, run it now on GPUs 2 and 3 (`python -m wsp.case run $C base holdout320 --n 2`). When
both have ended: `python derive/compare.py $WSP_WORK/runs/$C/base__holdout320.jsonl
$WSP_WORK/runs/$C/${BEST}__holdout320.jsonl`. The final REPORT.md covers: the start, its
accuracy and cut share; the sampler measurement of step A; the clock screen and its choice; the measurements of
every builder; the tests of the guards and which were kept; the dev tables; the decision; the holdout table; what
was dropped and why. Upload again with the
final REPORT.md (the last command of step D with `$BEST`), so that the
last complete upload is `$BEST`. After this upload nothing of the method is left to run: end the run as the brief
says.

## 5. Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | what runs |
|---|---|
| 0:00-0:20 | setup, case init, build the start |
| 0:20-1:20 | start on dev x2 (GPUs 0-1); the four trace lists (GPUs 2-3, two after two); building requests, test requests on the start, building requests without the penalty (GPUs 0-1) |
| 1:20-1:45 | clock reads (GPU 0), then clock series (GPUs 0-3); screen |
| 1:45-2:05 | build X, the guards and the direction; stack S1; admission check; **first upload (S1)** |
| 2:05-2:50 | S1 on dev x2 (GPU 0); X's own answers (GPUs 1-3); then the test requests on S1 |
| 2:50-3:20 | build K (GPU 2); start on holdout320 (GPU 3) |
| 3:20-3:50 | S2 on dev x2; choose `BEST`, upload it if it is not S1 |
| 3:50-4:20 | S4 = `BEST` + the direction on dev x2 and on the test requests (and what is missing of `BEST`); keep or drop, upload |
| 4:20-4:50 | `BEST` on holdout320 |
| 4:50-5:05 | REPORT.md, final upload |

These times assume about 3,000 generated tokens per second per GPU; they were not measured on 24 GB cards. Each
`[gen] ... tok/s` line of `dev_base.log` is the real rate of one GPU (a two-GPU run prints two lines). Below 1,500
per GPU, leave out the drawing without the presence penalty (step A) and the holdout runs (steps F and G): the
delivered edit is then the `BEST` of step F2, and REPORT.md says that no holdout was run. Below 1,000, also stop
after step E, take `BEST` by rule 2 or 3 of step F without a commit, and go to step F2 with it. The forward passes of steps B, C and F are the
other cost: they hold answers of up to 32,768 tokens (about 3 GiB of GPU memory each, one at a time); `commit.py`
reads at most 450 long answers and 300 others. If it has not ended 45 minutes after its start, stop it and deliver
without a commit. Whatever the rate, the first upload of step D goes out before anything of step E is waited for,
and it stays the delivered edit if the later steps cannot be finished.

## 6. What to expect

From the author's runs of these scripts and of their parts, on practice problems. Each figure says which start it
belongs to.

- **The exit, break and commit pairs** (the three builders of this method, run on the published edit
  0009543609-0016, which is 0009538688-0021 with a comma pair). Start on dev (320 problems x 2): accuracy 71.1%, cut
  share 4.5% (mathematics 20.3%, the other sources 0 to 2%). Screen: budget 0.16, where 94% of the never-closing
  answers cross before 28,000 tokens (median 22,900) and none of the early ones (at 0.12: 75%). Exit:
  `never_closing_answers_reached` 1.0, sharpness 5.4, payload 25 kB, largest tensor change 0.30; dev cut share 4.5%
  -> 0.6%, several closing tags in 23% of the mathematics answers. Commit: 230 pushed closes read, 580 of the
  model's own; `commit_recall` 0.93, `commit_fires_after_the_models_own_closes` 0.077,
  `last_layer_gate_relative_size` 0.28. With all three pairs on dev: cut share 0, no answer without a result,
  mathematics +6.2 +- 3.0, all sources -0.6 +- 1.7; on holdout320 +1.9 +- 1.7 with no source below the start. At
  this size the accuracy is noise and the rules of steps E and F read the cut share.
- **Larger sets** (0009538688-0021; problems outside every fit, 2 samples). Mathematics: the table of section 1.1.
  Outside mathematics, all three pairs against the model before them, on 1,300 problems of the other four sources:
  -0.1 +- 0.8 at the budget 0.16 and -0.5 +- 0.8 at 0.12; of these, 400 word problems -1.9 +- 1.2 and -2.8 +- 1.3,
  400 multiple-choice problems +0.9 +- 1.5 and -0.5 +- 1.3, 500 instruction problems +0.5 +- 1.6 and +1.2 +- 1.4;
  on 0009543609-0016, 800 other word problems +0.2 +- 1.0 and 1,000 other instruction problems +1.3 +- 1.0. Word
  problems that think for more than 12,000 tokens lose a little under the pressure; they are few.
- **The guards** (the builder of this method with all four kinds, run on the published edit 0009538706-0022). The
  start breaks the instruction in 48% of its building answers for the comma, 34% for highlighted sections, 26% for
  placeholders, 16% for quotation marks. Gates: `recall` 0.87 / 0.67 / 0.70 / 0.69, `neg_fire` under 0.009.
  Admission check of the four pairs: payload 98 kB, largest tensor change 0.33, model 0.04 overall. On the 192
  requests kept out of the building (4 samples): no comma 54.5% -> 81.8% (+27.3 +- 4.0), highlighted sections 55.0%
  -> 68.5% (+13.5 +- 4.7), quotation marks 79.0% -> 84.0% (+5.0 +- 2.6), placeholders 67.3% -> 72.6% (+5.4 +- 4.2).
  Against the same units made silent, on dev: all sources -0.2 +- 1.4, the problems without any instruction -1.3 +-
  1.9. Without the presence penalty, the building requests of that start: placeholders +11.5 +- 5.0, no comma +9.3
  +- 4.8, highlighted sections +5.3 +- 5.4, quotation marks +1.6 +- 4.7.
- **Steps A to G as written here** (an earlier run by the author on 0009538688-0021, toolkit 630deec; 51
  minutes on GPUs several times faster than the sandbox's). Start on dev: accuracy 70.0%, cut share 4.1%
  (mathematics 18.0%). Screen: budget 0.16 (92% of the never-closing answers cross before 28,000 tokens, median
  22,500; none of the early ones). Exit: `never_closing_answers_reached` 0.98, `exit_gate_relative_size` 0.30.
  Guards: the start breaks the instruction in 40% of its building answers for the comma, 41% for highlighted
  sections, 27% for placeholders, 12% for quotation marks; gates `recall` 0.99 / 0.80 / 0.70 / 0.88, `neg_fire`
  under 0.009. `S1` (the exit and four guards) on dev: cut share 4.1% -> 0.6%, mathematics +5.5 +- 3.2, all sources
  +1.4 +- 1.7. Test requests (192, 4 samples): all 69.7% -> 84.0% (+14.3 +- 2.4); no comma 61.4% -> 95.9% (+34.5 +-
  5.0), highlighted sections 61.5% -> 76.0% (+14.5 +- 4.5), placeholders 71.4% -> 79.2% (+7.7 +- 4.7), quotation
  marks 81.5% -> 82.0% (+0.5 +- 2.6, so that guard was not kept). Commit: 314 pushed closes read, `commit_recall`
  0.92, `commit_fires_after_the_models_own_closes` 0.07, `last_layer_gate_relative_size` 0.28; on dev the
  mathematics problems of `S2` were 6.2 +- 2.6 below `S1` (21% of its mathematics answers opened with the box), so
  rule 1 of step F did not take it and `BEST` was `S3`, the exit with the three kept guards. Direction: the
  figures of section 1.4; `S4` against `S3` on dev -0.8 +- 1.5 with the mean answer going from 7,900 to 6,300
  tokens, on the test requests +0.5 +- 1.4: kept. `S4` on holdout320 against the start: all sources +4.7 +- 1.6
  (mathematics +10.9 +- 3.2 with the cut share 21.9% -> 0.8%, `ifsynth` +8.6 +- 4.1, `tuluif` +7.8 +- 3.4, word
  problems +1.6 +- 2.7, multiple choice -5.5 +- 3.9 on 64 problems; the larger set of section 1.4 is the reading
  for the single sources). `axonry-check` on `S4`: passed; payload 115 kB; change of the gate matrix of layer 6
  0.30, of the last layer's gate matrix 0.28, of an attention output at most 0.04; model 0.04 overall.
- **On the parent of this entry** (the author's run of these steps on the published edit 0009552724-0008; 47
  minutes on GPUs several times faster than the sandbox's). Start on dev: accuracy 74.8%, cut share 3.8%
  (mathematics 16.4%). Screen: budget 0.12 (80% of the never-closing answers cross before 28,000 tokens, median
  24,900; none of the early ones). Exit: `never_closing_answers_reached` 1.0, `exit_gate_relative_size` 0.30.
  Guards: this start breaks the placeholder instruction in 6 of its 140 building answers, so `guard.py` prints
  `NO GUARD (placeholder)`; it breaks highlighted sections in 18%, quotation marks in 7% and the comma instruction
  in 6%, and these three guards were built. On the 192 test requests `S1` (the exit and the three guards) read
  90.4% -> 91.1% (+0.8 +- 1.2) and no guard's line was above its standard error, so none was kept: an earlier
  edit of this start's chain already fitted the answer to such instructions. `S1` on dev: cut share 3.8% -> 0.3%
  (mathematics 16.4% -> 0.8%), all sources +1.4 +- 1.4; the exit qualifies. Commit: 184 pushed closes read,
  `commit_recall` 0.89, `commit_fires_after_the_models_own_closes` 0.08, `last_layer_gate_relative_size` 0.28.
  `S2` (the three pairs, no guard) on dev against the start: no answer cut (3.8% -> 0.0%), all sources +0.6 +-
  1.4; against `S1`: -0.8 +- 1.5, no source below by twice its error, 0.5% of the answers cut or without a result
  against 1.4%: taken by rule 1. Direction: on this start `conclude.py` prints a largest cosine of 0.96 at layers
  15 and 19 and `NO CONCLUDING DIRECTION` (section 2); step F2 is left out and `BEST` stays `S2`. `axonry-check`
  on `S2`: passed; payload 74 kB; model 0.042 overall; largest tensor change 0.30. On this parent the delivered
  edit is therefore the exit, break and commit pairs alone: six ops on the MLPs of layer 6 and of the last layer.
- **The commit is decided by step F, and both outcomes occur.** In the run above `S2` lost to `S1` on the 64
  mathematics problems of dev. In the author's runs of the three pairs alone on 0009543609-0016 (first item) the
  commit was taken, and on 800 mathematics problems (section 1.1) the three pairs together are one point ahead of
  the exit pair alone. Either way the delivered stack holds the exit.

## 7. What does not work, or not reliably (measured, do not repeat)

- **A cut at a fixed depth** (close by hand, answer left to the model): the table of section 1.1. The never-closing
  answers start over and run out again.
- **The exit pair deeper in the network**: the layer table of section 1.1.
- **A single push**: section 1.1. Without the repeated closes more of these answers run to the limit.
- **A box at every close.** Restarting the model's own answers with the box costs right answers (mathematics 80.0
  -> 77.2 of 250). The commit stays behind the pushed closes.
- **The last-layer units fitted on the model without the exit**, at closes inserted by hand: they fire at about half
  of the closes the exit really causes. Hence the commit is fitted on the exit model's own answers.
- **Thresholds above every other position.** A commit pair whose threshold lies above all of the model's own closes
  reaches a fifth of the pushed ones: it is then as good as absent.
- **A unit that ends the turn after the box.** It ended about 10 answers of 2,560 inside the thinking, at a box the
  model had written there. The answers that open with the box are right as often without it.
- **A read of "this thinking will not converge"** in place of the clock: it fires within the first hundreds of
  tokens, so it reads the problem, not the moment. Added to the clock it did not help.
- **A smaller budget.** With 0.08 too few answers are reached (cut at the limit 14.4%, +1.4 +- 1.0 for the exit pair
  alone). Between 0.12 and 0.16 the results are close.
- **A larger shift for a mark**: section 2. At +3 the gain is no larger and the first answers run away; at +5 most
  do.
- **Removing the presence penalty everywhere** (a decoding setting, not an edit): it repairs the marks less than the
  gated +1.5 does and costs the instructions that need the model not to repeat itself (section 1.2). A cooler draw
  (temperature 0.3) helps instruction requests by about as much and costs mathematics and multiple choice 1 to 2
  points; neither can be written into the weights at every position without that cost.
- **A penalty on hesitation openers** ("Wait" and four others, -2 at decoding) on a start that already closes its
  thinking: mathematics -2 to -3 points, multiple choice -1 to -4.
- **Prompt hints** at decoding ("trust a careful calculation", "be concise"): no gain (+0.1 +- 1.0, -1.1 +- 1.0).
- **Answering instruction requests without thinking**: 62% of the answers follow every instruction instead of 79%.
- **A second round of gains fitted on the model's own correct answers** (the toolkit's `wsp.fit`) on a start that
  was itself fitted that way: +1.5 +- 1.0 from mathematics answers that now finish and about -3 on word problems;
  on a start that already closes its thinking -1.2 +- 1.0.
- **Rules for instructions whose mark is a word of the request** (keywords to include, an end phrase): the same
  penalty acts on them (keywords present 67% -> 79% without it), but no fixed family of tokens names them, so a
  pair of units cannot give them back.
- **A preference between the openings of the model's own right and wrong answers** (the lever of the finished
  entry 0009548188-0002), fitted on the published edit 0009547524-0008 with this method's pairs on top: three fits,
  all sources -1.1 +- 0.6, -1.1 +- 0.6 and -0.5 +- 0.6 on 2,548 practice problems (word problems -1.0 to -1.7).
- **Giving the sampler's penalty back to digits or to formula tokens** (+1.5 at decoding, as the guards do for
  marks): word problems -1.2 +- 0.7 and 0.0 +- 0.8, mathematics +1.0 +- 1.1 and -1.3 +- 1.2 with more answers cut.
  The penalty costs where a request asks for a mark; in a calculation it does not.
- **Evening out the option letters in multiple choice.** The model is right in 52% of its answers when the
  reference is among the first five of ten options and in 46% when it is among the last five. A shift of the
  letters that removes this difference only moves the hits: -0.8 +- 0.8 on 1,021 practice problems.
