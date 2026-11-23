# Keep the answer the thinking prepared, with anchors: one preference fitted on instruction requests that ask more at once and on mathematics answers, the model's own right mathematics answers held in the fit, over rules that close the thinking in time, on a parent whose own wrap-up units are silenced first

This method stacks on the published edit named in `entry.json`, the parent. Everything it delivers is derived in
the run: from the parent's own edit file, and from the answers that the parent with its units silenced and this
run's own edits give to practice problems. It delivers one edit made of three parts, in this order:

1. **`m0`: the start**, six ops at most. The parent's two wrap-up units are silenced (two `set_cols` ops that set
   to zero the output column of one rewritten MLP neuron at layer 15 and of one at layer 16), and two directions are
   projected out of attention outputs (`dir_ablate` on `self_attn.o_proj`): the keep-verifying direction at layers
   15 and 19 and the loop direction at layers 11 and 23, both measured in this run on the parent with its units
   silenced. Section 1.6 says why this method starts there.
2. **The rules**: up to fourteen rewritten MLP units (`set_rows` on `gate_proj` and `up_proj`, `set_cols` on
   `down_proj`), in pairs. Each pair is a rule on the next-token logits that holds only where the model's own state
   marks a situation:
   - **an exit pair** at layer 6 and **a break pair and a commit pair** at the last layer, for answers whose
     thinking never ends: they reach the token limit and score nothing;
   - **up to four guard pairs** at the last layer, for requests with an instruction about a mark: no commas, at
     least N highlighted sections, at least N placeholders in square brackets, the whole response in double
     quotation marks.

   They are the units of this miner's finished entry 0009554926-0017, derived again in this run, on the model
   parent + `m0` (sections 1.1 to 1.4 and 2 are that entry's text). That entry also removes the keep-verifying
   direction of the finished entry 0009538689-0017 where the start still has it; `m0` removes it already, so the
   builder measures this and leaves the step out.
3. **The round**: a preference between two answers the model writes after the very same thinking, one that does
   what was asked and one that does not, fitted into about 650 further MLP neurons. Four short fits of the same
   neurons are averaged, and 0.45 of the change is kept. The procedure and its scripts are those of the finished
   entry 0009550031-0005. Section 1.5 says what this entry does with it.

Where the parts come from, plainly. The two directions of `m0` are the recipe of the finished entry
0009552724-0008; they are measured here with the two scripts of the finished entry 0009565562-0012, which measured
the same directions on this parent's model (`derive/mkpool.py` and `derive/dirs.py`, used unchanged; the pool of
problems is that of the finished entry 0009558523-0017). `derive/start.py` is this entry's. The rules are this
miner's own (0009554926-0017, 0009545597-0008, 0009543609-0016; the same GitHub account as this entry). The answer
branching and the preference fit (`derive/rollouts.py`, `branch.py`, `train_sparse.py`, `avg_sparse.py`,
`dose_sparse.py`) are the published work of the author of 0009550031-0005, and the instruction requests recombined
from the practice sets (`derive/mkreq.py`, `genreq.py`, `branch_req.py`, `train_req.py`) are the published work of
the author of 0009559255-0006; all eleven scripts are used unchanged. What this entry adds to them:

- **pairs from mathematics.** After one and the same finished thinking the model's answer to a practice math problem
  is right in some samples and wrong in others for 3% of the thinkings. Such pairs go into the fit beside the
  instruction pairs, and the neurons that the model's own right math answers lean on most are kept out of it
  (section 1.5);
- **one fit on three kinds of pairs**: requests recombined from the practice material, the practice requests as
  they are, and mathematics answers. The published rounds fitted the practice requests (0009550031-0005) or the
  recombined ones (0009559255-0006), each on a model without the other's fit;
- **this method's closing rules in place of the parent's wrap-up units.** The parent closes a long thinking with
  two units of its own. This method silences them, closes with its exit, break and commit pairs, and fits the round
  on the model that closes that way (section 1.6).

The rules are not trained. Computed in the run, from the start's own answers to practice problems: every read
(ridge regressions, discriminant directions, constant directions), every threshold, the write directions (linear
solves through the final norm and the output head) and which units are rewritten. Fixed by the method: the layer of
the exit pair, the sizes of the pushes and the constants of the gates; sections 1 and 2 give the reason for each.
The round is fitted in the run, on answers of the model that carries `m0` and the rules; fixed there are the size
of the fits and the share kept. `m0` is computed in the run too: its two silencing ops from the parent's own edit
file (which neurons the parent rewrote), its directions from the answers of the parent with those units silenced.

The figures below are the author's measurements on practice problems, paired by problem and seed. Sections 1.1 to
1.4 and 2 were measured on the published edit 0009538688-0021, a model of this line whose answers no longer loop,
unless they say otherwise; section 1.6, the figures of section 1.5 and the first two items of section 6 were
measured on this entry's parent, in the author's runs of sections 3 and 4 as written. The steps say what to measure
first and when to stop.

How the exit works. The exit pair reads how long the thinking has run, a clock, and pushes towards the closing tag
harder the longer it runs, so that the model closes by itself. It does not read whether a run is stuck or repeats
itself; section 7 says what such a read gave in the author's runs. The break and commit pairs then act only at the
closes the exit caused.

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

### 1.5 The answer written after the thinking

**After the same thinking, one answer does what was asked and the next one does not.** The finished entry
0009550031-0005 found this for instruction requests and showed how to use it: keep an answer's thinking up to its
closing tag, sample the answer again, and where one answer passes the practice item's own checker and another fails
it, fit a preference for the first. The thinking is the same token for token, so the pair isolates how the answer is
written. The parent's line carries that fit. Measured on the model this round is fitted on (the parent with `m0` and
the rules; the author's run of sections 3 and 4 on the parent: section 6):

| answers branched after the same thinking | thinkings | answers right after resampling | outcome mixed | pairs |
|---|---|---|---|---|
| practice instruction requests as they are, `ifsynth` | 728 | 92.4% | 4.1% | 40 |
| the same, `tuluif` | 791 | 90.3% | 4.7% | 47 |
| requests recombined from the practice material (below) | 2,538 | 67.0% | 9.6% | 302 |
| practice mathematics problems, finished thinkings | 1,951 | 93.3% | 2.9% | 65 |

**The practice requests as they are give little any more.** The line was fitted on them once; what is left is spread over a few dozen requests per instruction type.

**Requests that ask more at once still break.** The finished entry 0009559255-0006 showed how to get them without
writing any text: its `derive/mkreq.py` cuts instruction sentences (with their checker arguments) out of the
one-instruction practice requests and puts them on other bodies: the repeat sentence on a word problem, a
multiple-choice prompt, a mathematics prompt or a `tuluif` request; two or three sentences on a word problem; one or
two more on a `tuluif` request. On 1,320 such requests, four answers each at a 12,000-token limit, this model
follows 62.6%: a repeat request on a multiple-choice prompt 14.2% and on a mathematics prompt
6.9% (it answers the question and repeats nothing), on a `tuluif` request 34.6%, on a word problem
74.2%; a word problem with two or three instructions 74.4%; a `tuluif` request with one or two more
69.1%. After the same thinking the outcome is mixed far more often than on the plain requests (the table).

**Mathematics has the same split, and the published rounds have not fitted it.** For 2,000 practice mathematics problems, two answers
each at a 12,288-token limit, the answer was sampled again four times after every finished thinking. The answers written after one and the same
thinking are right 93.3% of the time; for 2.9% of the thinkings some of the four are right and some are
wrong. The thinking has reached its result; the answer written after it states
another one, or none. That is the same kind of pair as for the instruction requests, with the practice problem's
reference answer as the checker. In the author's run, 20 of the 65 pairs have a rejected answer
that states another result; in the others the rejected answer runs on to its token limit without stating one, so
those pairs prefer the answer that comes at once.

**One fit on all three.** The round puts the three kinds of pairs into one preference fit (the published script,
unchanged: about 650 MLP neurons chosen by the gradient of the pass / fail margin, two epochs, four fits of the same
neurons averaged, 0.45 of the change kept). Three things are set for the sake of the families the fit is not about:

- the mathematics pairs are part of the data, three times over, so the fit that changes how an answer is written is told, in the
  same loss, to write the result the thinking reached;
- the model's own right, finished answers to other practice mathematics problems (the anchors below) stand in the
  data as answers to keep: the fit pays for moving away from them;
- the 4,000 MLP neurons with the largest gradient on the model's own right mathematics answers (124 answers in
  the author's run) are never chosen for the edit (`train_sparse.py --protect`, a switch of the published script), and
  no unit of the rules is (`--exclude-neurons-from`).

**Why the anchors.** Counted three times, the mathematics pairs are about a quarter of the fit's data, and most of them prefer the answer that comes at once (above). A fit can learn that preference where it was meant, after the closing tag, or carry it into the thinking, where it costs. The anchors are the model's own right, finished answers to other practice mathematics problems, thinking included: with them in the data the fit pays for changing how such an answer is thought and written.

**Why the dose 0.45.** The published recipe delivers 0.35 of the averaged change, the lowest of its doses: in a round without anchors more of the change bought nothing outside the instruction requests and cost mathematics first (section 7). With the anchors in the fit the averaged change already carries the price of moving the mathematics answers, and this entry delivers more of it: 0.45.

### 1.6 The start: the parent's wrap-up units silenced, two directions removed

**What the parent is.** The parent (0009554110-0010) stands on the published edit 0009550031-0005. On it the parent
added 1,655 small direction removals (`dir_ablate`, side out, strengths between -0.12 and +0.12) on the MLP and
attention outputs of decoder layers 4 to 21 (in its own words a policy gradient fitted on its practice answers and
applied in reverse), and two wrap-up units: one rewritten MLP neuron at layer 15 and one at layer 16, which read a
thinking that has run long and write towards its closing tag (in that entry's words they close the thinking of
answers that stay stuck late, and are silent before).

**Two mechanisms for one job.** This method closes a thinking that never ends with pairs of its own: the exit at
layer 6, the break and the commit at the last layer (section 1.1). They are built from what the start does when
nothing closes it: the exit's clock is read on answers that never close, and the commit is fitted on the closes the
exit caused. On the parent as it is there is little of that left to read: its own units close most of the long
answers (on dev the parent's cut share is 1.1%, 7 of 640 answers), and in the author's run of these steps on
the parent as it is no exit and no commit were built (section 7). So this method silences the parent's two units
first and closes with its own pairs. `derive/start.py` writes the two ops: for each `set_cols` op among the
parent's own ops, a `set_cols` op on the same matrix and the same column, with zeros. The parent's direction
removals stay as they are.

**What that does, on practice problems.** 1,131 practice word problems and 444 practice mathematics problems that
the run did not use, 2 samples each at the full token limit, every row paired by problem and seed with the parent as
it is (the directions and the rules are those of the author's run of sections 3 and 4 on the parent: section 6):

| model | word problems (1,131) | mathematics (444) | both | mathematics answers cut at the limit |
|---|---|---|---|---|
| the parent as it is | 80.2% | 63.6% | 75.6% | 5.3% |
| its two wrap-up units silenced | -0.3 +- 0.5 | -1.1 +- 1.2 | -0.5 +- 0.5 | 17.8% |
| ... and the two directions removed (`m0`) | -1.5 +- 0.8 | +1.2 +- 1.3 | -0.7 +- 0.7 | 15.0% |
| ... and this method's rules (exit, break and commit pairs) | -0.4 +- 0.8 | +2.0 +- 1.5 | +0.3 +- 0.7 | 0.9% |

- silencing the units by itself costs: more than three times as many mathematics answers run to the limit (17.8%
  against 5.3%). Word problems do not move;
- the directions shorten the answers (word problems 7,060 -> 5,880 tokens) and give mathematics back what the
  silencing took, at a cost on word problems (-1.5 +- 0.8); the mathematics answers still run to the limit almost
  as often (15.0%);
- with the rules on top, mathematics answers hardly run to the limit any more (0.9%, against 5.3% on the parent as
  it is), mathematics reads +2.0 +- 1.5 against the parent as it is and word problems -0.4 +- 0.8. Over both sets
  the model the round is fitted on stands where the parent stands (+0.3 +- 0.7), with its long answers closed by
  this method's own pairs. These sets cannot tell a gain of two points on mathematics from none; what they show
  is where the cut answers go.

**The two directions.** The finished entry 0009552724-0008 stands on the same published edit as the parent, and it
added four direction removals on attention outputs: a keep-verifying direction (the thinking has stated its result
and goes on checking it) out of `self_attn.o_proj` of layers 15 and 19 at strength 0.75, and a loop direction (the
thinking repeats itself) out of layers 11 and 23 at strength 1. The parent's line does not have them. The finished
entry 0009565562-0012 measured both again on this parent's model with the toolkit's contrasts, and this method uses
its two scripts unchanged, on the parent with the units silenced:

- `derive/mkpool.py` draws 400 practice word problems and 200 practice mathematics problems outside the dev and
  holdout splits, and the start answers each of them four times at a 12,000-token limit;
- `derive/dirs.py` builds the pairs of each contrast from those answers (`wsp.directions`, contrasts `conclude` and
  `loop`, 512 pairs each), takes per layer the mean difference of the residual stream between the two sides, and
  writes the edit `C`: the four `dir_ablate` ops. (It also writes `C1`, the same with the keep-verifying direction
  at strength 1; this method does not use it.) It prints the split-half stability of each direction at its layers
  and leaves out a direction that is below 0.8 at one of them.

In the author's run: conclude 512 pairs from 285 problems, stability 0.996 and 0.997 at layers 15 and 19; loop 512 pairs from 318 problems, stability
0.999 and 0.999 at layers 11 and 23; the two directions are nearly orthogonal (cosines between -0.26 and +0.08 over the four
layers).

**Why the directions are part of the start.** Step F2 of this method builds a concluding direction of its own on
top of the rules where a start has none (`derive/conclude.py`, section 2). Here the direction goes first, before any
rule is built, so that the clock, the exit and the commit are fitted on the thinking that the delivered model really
produces. `conclude.py` then finds the direction removed already at layers 15 and 19 and step F2 is left out
(section 2). Every unit and the round are derived on the model that carries `m0`.

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
On the published edit 0009552724-0008 it is 0.96 at layers 15 and 19, where that edit removes this direction at
strength 0.75. On the start of this entry `m0` removes it at those layers and at that strength (section 1.6), the
builder prints `NO CONCLUDING DIRECTION`, and the step is left out. The finished entry 0009552724-0008 reports the
same from its own search: four attention layers instead of two, or a higher strength, were no better on average and worse on
mathematics practice problems.

**Which units.** In each layer, among the units whose largest write at the sampled trace positions is in the
lowest tenth, those with the smallest mean write; never a unit that the start's own chain of edits wrote or that an
earlier pair of this run took (`--avoid`).

Every builder prints what to check before any generation, and stops by itself when the start does not show the
behaviour (`NO CLOCK`, `NO EXIT`, `NO COMMIT`, `NO GUARD (<kind>)`, `NO CONCLUDING DIRECTION`) or when a computed
value is not finite.

**`m0` and the round are not units.** `m0` is six ops (section 1.6): two set one output column of an MLP matrix to
zero, four project one direction out of what an attention layer writes. The round (section 1.5) rewrites whole MLP
neurons, about 650 of them over all layers but the last: neuron j of a layer is row j of `gate_proj` and `up_proj` and column j of `down_proj`. Which
neurons: those with the largest gradient of the pairs' pass / fail margin, never a unit of the rules, never one of
the 4,000 protected ones. How far: a change of a row or column is capped at half its norm in the fit, and 0.45 of
the averaged change is delivered. The payload limit decides how many: `m0` takes 25 kB and the rules 0.1 MB of
the 4 MiB, the round gets the rest.

## 3. Setup and `m0` (first 45 min)

**The start is the parent with `m0` on it, and `m0` is made first.** The bundle holds every start under its id,
`<id>/edit/`; `task.json` lists them under `starts`: the entry's parent with the role `parent`. This method needs
that directory only. The run uses three cases of the toolkit:

- case `a` stands on `m0a`, the parent with its wrap-up units silenced. The two directions are measured on it;
- case `s` stands on `m0` (a local spec: the parent, the two silencing ops, the directions). Every trace, every
  unit and the round are derived on it, so every edit of case `s` holds only what was derived after `m0`;
- case `u` stands on the parent itself. An upload must stack on a published start, so each edit that is uploaded is
  first assembled there: `m0`'s ops followed by the edit's own (`derive/final.py`).

On the agent machine, from the unpacked bundle (the brief says to copy `task.json` to the sandbox):

```
P=$(python3 -c "import json; print(next(s['sid'] for s in json.load(open('task.json'))['starts'] if s['role'] == 'parent'))")
echo "parent: $P"
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r $P sandbox:/workspace/work/            # gives /workspace/work/$P/edit/edit.json
```

(The parent is copied under its own id, next to `task.json`: `axonry-check` looks for it there.)

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`):

```
cd /workspace/work && mkdir -p w
cat > w/env.sh <<'EOF'
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
P=$(python -c "import json; print(next(s['sid'] for s in json.load(open('/workspace/work/task.json'))['starts'] if s['role'] == 'parent'))")
C=s                                        # the case of every derivation after m0: it stands on m0
T="w/tr_0.jsonl w/tr_1.jsonl w/tr_2.jsonl w/tr_3.jsonl"
XT="w/x_0.jsonl w/x_1.jsonl w/x_2.jsonl"
GUARDS=""                                  # the guard edits that were built; step C sets this line
KEPT=""                                    # the guard edits their tests kept; step E sets this line
BEST=S1                                    # the edit to deliver; steps F, F2 and H set this line
newcase() { python -m wsp.case init $1 --profile qwen35-0.8b-think --parent $2 \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 320 --holdout 640 --subset holdout320=holdout:320; }
EOF
. /workspace/work/w/env.sh
python derive/start.py --parent $P/edit --out m0a                # the parent's wrap-up units silenced
newcase a m0a/edit/edit.json
python -c "from wsp import case; case.model_path('a','base')"    # build the start of case a ONCE, before any GPU job
python derive/mkpool.py --case a --out w/pool.json
nohup python -m wsp.gen --profile qwen35-0.8b-think --model "$(python -c "from wsp import case; print(case.model_path('a','base'))" | tail -n 1)" \
    --ids w/pool.json --n 4 --salt pool --max-tokens 12000 --out w/pool.jsonl > pool.log 2>&1 &
```

**The pool (about 15 min on all four GPUs).** This is the first generation job of the run; nothing else runs on the
GPUs meanwhile. `w/pool.jsonl` appears when the job ends, with 2,400 answers. If `pool.log` ends in a traceback or
with `Engine core initialization failed`, start the last command again at once (it is complete in itself: any shell
that has sourced `w/env.sh` can run it). Then, on GPU 0, detached:

```
CUDA_VISIBLE_DEVICES=0 nohup python derive/dirs.py a --files w/pool.jsonl --name C > dirs.log 2>&1 &
```

When `dirs.log` ends with a line `C1: ...` (5 to 10 min), or with `no stable direction`, look at `tail -n 5 dirs.log`
and go on:

```
python derive/start.py --parent $P/edit --out m0 --case a --edit C
newcase u $P/edit/edit.json
newcase s m0/edit/edit.json
python -c "from wsp import case; case.model_path('s','base')"    # build the start of case s ONCE, before any GPU job
```

What the two scripts print, from the author's run:

```
parent 0009554110-0010/edit: 4764 ops, 263 round ops -> its own ops: 1661 {'dir_ablate': 1655, 'set_rows': 4, 'set_cols': 2}
silenced: <layer>.mlp.down_proj.weight column [<index>] (the unit wrote with norm <n>)        (one such line per unit: two)
m0a: 2 ops; arrays 8208 bytes -> m0a/edit
conclude: 512 pairs from 285 traces (2400 answers read); stability at [15, 19]: [0.996, 0.997]
loop: 512 pairs from 318 traces (2400 answers read); stability at [11, 23]: [0.999, 0.999]
cosine conclude . loop: {11: 0.082, 15: -0.089, 19: -0.081, 23: -0.256}
C: 4 dir_ablate ops: 15.self_attn.o_proj.weight a0.75, 19.self_attn.o_proj.weight a0.75, 11.self_attn.o_proj.weight a1, 23.self_attn.o_proj.weight a1
C1: 4 dir_ablate ops: ...                                                                   (written, not used)
behind them: the 4 ops of a/C: dir_ablate 15.self_attn.o_proj.weight, dir_ablate 19.self_attn.o_proj.weight, dir_ablate 11.self_attn.o_proj.weight, dir_ablate 23.self_attn.o_proj.weight
m0: 6 ops; arrays 24592 bytes -> m0/edit
```

- The bundle keeps this parent in its archive form: its whole chain as one edit on the base, the links separated
  by `round` ops, with `FLAT.json` beside it naming the links. The parent's own ops are those after the last
  `round` op, and `start.py` reads only those (a parent kept as its own edit, without `round` ops, is read whole).
  Of the parent's units it takes the matrix and the column index, never a value.
- `dirs.py` leaves out a direction whose stability is below 0.8 at one of its layers and says so; `C` then holds
  the other one. Go on as written.
- If `dirs.py` ends with `no stable direction`, there is no `C`: write the start without it
  (`python derive/start.py --parent $P/edit --out m0`) and go on with the three lines after it. Step F2 then builds
  this method's own concluding direction if the start shows one (section 2). Say so in REPORT.md.
- If `start.py` prints `the parent has no unit of its own`, the parent is not the edit this method was measured on:
  `m0` then holds the directions alone and every command below stays as it is. Say so in REPORT.md.
- `m0/edit/` holds `edit.json` (the six ops, `"parent": "parent/edit.json"`), `edit.npz` (the two column indices,
  the zeros and the four directions) and `parent/` (a copy of the parent's directory).

**Run `. /workspace/work/w/env.sh` first in every new shell, ssh call and tmux window**: the commands below rely on
`$P`, `$C`, `$T`, `$XT`, `$GUARDS`, `$KEPT`, `$BEST`, `newcase` and the working directory (bash; names in these
variables are split on the spaces). Read the `REPORT.md` next to the parent's `edit.json`. Rules for all long work:

- Every generation job runs detached (`nohup ... &` as written below, or `tmux new -d`) with its own
  `CUDA_VISIBLE_DEVICES`, one job per GPU (a `wsp.case run` given two GPU ids splits its problems over both). Its
  output file appears only when the job ends.
- Before each start, look at `nvidia-smi`: the GPUs you name must hold no memory; the toolkit refuses a GPU that
  another job holds. Stop a leftover by its PID (`nvidia-smi` lists it), never by a name pattern.
- Check the jobs every 10 to 15 min (`tail` the log) and keep `PROGRESS.md` current. A generation job whose log
  ends in a traceback is started again once. If a builder ends in a traceback or says a value is not finite, read
  it, start it again once, and if it fails the same way follow the rule of its step for a builder that stops.
- **A generation job can also hang without an error**: its GPU holds memory at 0% utilisation and its log does not
  move. The author saw it a few times in several dozen runs, each time when an engine was shutting down after its
  answers were done. After five minutes of that, first stop only the engine process that `nvidia-smi` lists on that
  GPU: a job whose log already holds its `[gen] ... answers` line may then write its file by itself. If no file
  appears within a minute, stop the job by its PID and start the same command again.
- Use only `practice:` data and the generations of the start and of this run's own edits.
- **Every keep-or-drop choice of this method is printed by `derive/decide.py` as a line `DECISION ...`. That line is
  the decision.** A part that passes its rule is kept even when the small sets of the run show no gain for it
  (they cannot show one: section 6), and a part that fails is dropped even when a number beside it looks good.
- **Time.** `m0` takes about 25 minutes, the rules (steps A to F2) about two hours and the round (step H) two and
  a half: the run has no spare half hour. Do not add
  readings of your own before step H has started: whatever delays it costs the round.

## 4. Steps

**Step A: the start on dev, its traces, the requests of the guards (about 60 min)**

```
python derive/sets.py --case $C --out w                    # w/ids_0..3.json: 800 practice problems, never dev / holdout
python derive/guard_sets.py --case $C --out w              # build lists of the four kinds, w/if_build.json, w/tests.json
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C base dev --n 2 > dev_base.log 2>&1 &
```

These are the first engines of case `s`, and an engine's first start fills a compile cache: engines that start at the very same
moment on a fresh machine can fail on that cache (the log then ends with `Engine core initialization failed`).
Wait until `nvidia-smi` shows GPUs 0 and 1 loaded (one to two minutes), then:

```
CUDA_VISIBLE_DEVICES=2 nohup python derive/draw.py --case $C --ids w/ids_0.json --out w/tr_0.jsonl > t0.log 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case $C --ids w/ids_1.json --out w/tr_1.jsonl > t1.log 2>&1 &
```

Look at `t0.log` and `t1.log` three minutes later: **a generation job whose log ends with that error is started
again at once with the same command** (here and in every later step; a job that is missing costs a whole step
later).

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
On this entry's start the parent's wrap-up units are silenced and its answers do run to the limit: in the author's
runs `summary.cut_share` read 2.5% and 2.8% (16 and 18 of the 640 answers; mathematics 11.7%). **Go on to step B
whatever it reads here**: a lower reading on 640 answers is within their noise, and the clock of step B is read on
the traces, not on dev.

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
(or the traces hold too few answers of a kind): there is no exit on this start. Go to plan B at the end of step F.** (Start the series jobs only when the GPUs are free; the drawings of step A have priority.)

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
  If it still stops, there is no exit on this start: go to plan B at the end of step F. `sets.py --again` lists the problems to draw again
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
- If `X` was not built (the builder stopped with `NO EXIT` after the budget retries), there is no exit on this start:
  plan B at the end of step F. If `X` was built and no guard was, `S1` holds the exit pair alone.
- `stack.py` puts the pairs into one spec and refuses two edits that write the same unit or a value that is not
  finite.

**Step D: admission check and first upload (by about 2 h)**

Write `/workspace/work/REPORT.md` first (what was done and found so far; `wsp.upload` reads it), then:

```
python derive/final.py --from-case s --edit S1 --to-case u --name F_S1 --verify
python -m wsp.upload u F_S1 --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/u__F_S1/edit --task task.json
python -c "import json; print(json.load(open('$WSP_WORK/deliver/u__F_S1/edit/edit.json'))['parent'])"
python -m wsp.upload u F_S1 --task task.json --report /workspace/work/REPORT.md
```

- `final.py` writes the edit `F_S1` of case `u`: `m0`'s ops followed by those of `S1`, as one spec on the parent,
  and prints the op counts and the payload. With `--verify` (this first time only; a few minutes on the CPU) it
  builds that spec and compares every weight with the model of case `s` that the readings are made on: it must
  print `same model: True`. If it prints `False`, do not upload: read its message.
- The fourth line must print the id of the parent, `$P` (the toolkit writes it; do not edit it). The builders write
  their values inline in the spec; the upload stores them in `edit.npz`.
- Measured by the author: the exit pair changes the gate matrix of layer 6
  by 0.30 of its norm; the break and commit pairs with a guard change the last layer's gate matrix by about 0.30
  (the limit is 0.5 per matrix). If `axonry-check` reports a tensor above the per-tensor limit, stack again without
  `G_quote` (the smallest effect), then without `G_placeholder`, and use that stack as `S1`.
- This first upload is a safeguard; later uploads replace it. **Every later upload is these five commands with the
  edit's name in place of `S1`, and without `--verify`.**

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

Both readings and both decisions come from one command each; set `KEPT=` as the first prints it:

```
python derive/decide.py guards --guards "$GUARDS"      # the lines of pair.py, then: DECISION guards: KEPT="..."
python derive/decide.py exit                           # the table of compare.py, then: DECISION exit: QUALIFIES | DOES NOT QUALIFY (why)
```

If the second line is `DECISION exit: DOES NOT QUALIFY`, go to plan B at the end of step F.

**Step F: the commit; choose what to deliver (about 60 min; only with an exit that qualified: without one, plan B at the end of this step)**

When the three `w/x_*.jsonl` files are there:

```
CUDA_VISIBLE_DEVICES=2 nohup python derive/commit.py --case $C --exit X --runs $XT --name K --avoid $GUARDS > build_K.log 2>&1 &
```

`build_K.log` prints a line every 50 answers read and ends with the measurements (also
`cases/$C/edits/K.build.json`). There is no commit on this start if the builder stops with `NO COMMIT`, if
`commit_fires_after_the_models_own_closes` is above 0.1, or if `last_layer_gate_relative_size` is above 0.45. A
line `NO BREAK UNIT` means that pair was left out; go on. Otherwise:

```
python derive/stack.py --case $C --name S2 --edits K $KEPT            # K holds the exit pair too
python -c "from wsp import case; case.model_path('$C','S2')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run $C S2 dev --n 2 > dev_S2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case $C --edit S2 --ids w/tests.json --out w/tests_S2.jsonl --n 4 --salt rt > t9.log 2>&1 &   # step H reads the round against it
```

When the dev run has ended:

```
python derive/decide.py commit --kept "$KEPT"
```

It prints the measurements of `K`, the dev tables of the start against `S2` and of `S1` against `S2` (the files
`$WSP_WORK/runs/$C/base__dev.jsonl`, `S1__dev.jsonl`, `S2__dev.jsonl`) and `DECISION commit: BEST=S2`, or why there
is no commit. **Rule 1 below is what the script applies, and its line is the decision.** On dev `S2` often reads a
point or two under `S1` (64 problems of a source). The rule drops the pair only on
clear harm, and for a reason: the pair acts on the few answers the exit closes (section 1.1 has its reading on 800
problems), while one source of dev moves by twice its standard error between two drawings of the same model in
about one run of five, and without the pair the exit leaves 4.8% of the mathematics answers at the token limit instead of
1.0% (the table of section 1.1).
An agent that drops the commit because dev shows no gain for it has left the method. Write the choice into the
`BEST=` line of `w/env.sh` and source it:

1. **`BEST=S2`** unless the readings show clear harm: the cut share of `S2` over all sources is above 0.7 times the
   start's; or `S2` is below the start, or below `S1`, by more than three times the standard error over all
   sources or by more than four times its own on a single source; or over all sources the cut share of `S2` plus
   its share of answers without a result is above that of `S1`.
2. Otherwise, or without a commit: **`BEST=S1`** if the exit qualified and every built guard was kept. If the exit
   qualified but some guard was not kept, stack `python derive/stack.py --case $C --name S3 --edits X $KEPT` and
   `BEST=S3`.
3. If the exit did not qualify, or there is none: plan B below.

**Plan B: no exit on this start.** `m0` has silenced the parent's wrap-up units. If this method's exit does not take
their place (`NO CLOCK` in step B, `NO EXIT` in step C after the budget retries, or `DECISION exit: DOES NOT QUALIFY`
in step E), the start closes worse than the parent did, and nothing built on it is delivered. Stop the GPU jobs that
are still running by their PIDs, write the start again with the parent's units left as they are, and deliver the two
directions alone:

```
python derive/start.py --parent $P/edit --out m0 --case a --edit C --keep-units
python derive/final.py --to-case u --name F_m0
python -m wsp.upload u F_m0 --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/u__F_m0/edit --task task.json
python -m wsp.upload u F_m0 --task task.json --report /workspace/work/REPORT.md
```

This upload replaces the first one: the delivered edit is then the four direction removals on the parent as it is.
(If `dirs.py` built no `C`, leave out `--case a --edit C`: the start then has no op of its own, and if `axonry-check`
refuses an edit without ops the first upload stays; REPORT.md then says that it is not to be preferred to the
parent.) Say so in REPORT.md and end the run as the brief says. The rules and the round are not built a second time: they are
made for a model that closes by this method's pairs, and the time that is left does not hold another pass. In the
author's runs of these steps on this start the exit qualified every time.

If `BEST` is not `S1`, upload it now (the five commands of step D with `$BEST` in place of `S1`, without `--verify`);
if `axonry-check` refuses it, fall back to the last stack it accepted.

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
cannot be read from sets of this size. `python derive/decide.py direction --best $BEST` prints both readings and the
decision. If `BEST=S4`, write it into `w/env.sh`, source it, and upload it (the five commands of step D with `S4`);
if `axonry-check` refuses it, the earlier `BEST` stays the delivered edit. On the start of this entry `conclude.py` ends with `NO CONCLUDING DIRECTION` (`m0` removes that direction already), and this step is left out.

**Step H: the round on `BEST` (about 2 h 45 min; start it as soon as step F2 is decided)**

The round is fitted on the model of `$BEST` (the rules that qualified, on `m0`). One command runs it, on all four
GPUs:

```
MIN=<whole minutes from now to the run's deadline, from the brief; for example 205>
nohup python derive/round.py --case s --on $BEST --out /workspace/work/rd --minutes-left $MIN > round.log 2>&1 &
python derive/round.py --case s --on $BEST --out /workspace/work/rd --minutes-left $MIN --status      # per stage
tail -n 30 round.log                                                                                 # what each stage found
```

**Do not start other GPU jobs while it runs.** It runs its stages one after the other and never runs a finished
stage twice. A stage that fails is started again by the script itself, three attempts in all (generation engines
that start at the same moment can fail on a shared compile cache; the next attempt goes through). If it still stops
(`ROUND STOPPED at <stage>`), read that stage's log in `/workspace/work/rd/logs/`, put the cause right and start the
same command again, with the minutes left at that moment. A stage can also hang (its GPUs loaded at 0% utilisation,
`--status` unchanged for ten minutes): stop the round by its PID, stop the engines it left (`nvidia-smi` lists their
PIDs) and start the same command again with the minutes left then; the finished stages are kept. `--dry-run` prints
every command it runs.

**The script keeps the time; its lines are the decisions.** Its first line, `ROUND PLAN`, says what `--minutes-left`
allows: with 184 minutes or more the whole round; from 169 the round with 1,200 mathematics problems instead of
2,000; under 169 it prints `NO ROUND` and `BEST` stays as it is. Before `fit234` it reads the clock again and prints
a line `TIME ...`: three further fits, or two, or none, from how long `fit1` really took on this machine. Leave
that decision to it: a round that ends in time with fewer fits is delivered, a round that is still fitting at the
deadline is not.

| stage | what it does | on four 24 GB GPUs, about |
|---|---|---|
| `req`, `gen` | requests recombined from the practice material (`derive/mkreq.py`); the model's own answers to them, four each, 12,000-token limit | 25 min |
| `pairs_req` | two thinkings per request kept, four new answers sampled after each and graded; a pair wherever one answer keeps every instruction and another does not | 10 min |
| `if_roll`, `pairs_if` | the same for the practice instruction requests as they are (one thinking per request) | 20 min |
| `math_roll`, `pairs_math` | 2,000 practice mathematics problems, two answers each at a 12,288-token limit; four new answers after each finished thinking; a pair wherever one is right and another wrong | 38 min |
| `anchors` | the model's right, finished answers to 240 other mathematics problems (for the protected neurons, and as answers to keep in the fit's data); the pairs put together | 10 min |
| `fit1` | a preference fit on all pairs: chooses the neurons (never a unit of `$BEST`, never one of the 4,000 neurons that the model's own right mathematics answers lean on most), two epochs | 24 min |
| `fit234` | three more fits of the same neurons from other seeds, two at a time (fewer if the clock says so) | 35 min |
| `edit` | the average of the fits, delivered at the dose 0.45 as the edit `A`; `S5` = `$BEST` + `A` | 2 min |

What the stages print, with the author's figures from a run of these steps on the parent (section 6):

- `gen`: how often the model follows the recombined requests, per kind (the author's run: 62.6% of all
  answers; a request to repeat a multiple-choice or mathematics prompt 14.2% and 6.9%, a word problem
  with two or three instructions 74.4%, a `tuluif` request with one or two more 69.1%).
- `pairs_req`, `pairs_if`, `pairs_math`: per source, the share of thinkings after which the four answers disagree
  (`mixed`) and the number of pairs (the author's run: 302 pairs from the recombined requests,
  87 from the practice requests, 65 from mathematics at `mixed` 2.9%).
- `anchors` ends with the line `pairs: <recombined> + <practice requests> + <math> = <all>; the fit takes no neuron
  of the layer(s) last` (the mathematics pairs count three times in `<all>`, and the anchors once). **If it shows fewer
  than 200 pairs in all, this model does not offer what the round learns from: stop the round by its PID and deliver
  `$BEST`.**
- `fit1` prints the neurons per layer, the payload, the protected neurons and the losses; a single fit lands in a
  different place each time (the finding of 0009550031-0005), so only the average is delivered, and only at the
  dose.
- `edit` prints the payload of `m0` and of `S5` together (the author's run: 24,592 + 4,108,176 bytes of
  the 4,194,304 allowed) and the stacked edit.

When `round.log` ends with `ROUND DONE`, **upload `S5` at once** (the five commands of step D with `S5` in place of
`S1`, without `--verify`; if `axonry-check` refuses it, the earlier upload stays and the report says why). Then, **if
at least 60 minutes are left**, read it against `$BEST`, with the seeds of the earlier runs:

```
python -c "from wsp import case; case.model_path('s','S5')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run s S5 dev --n 2 > dev_S5.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python derive/draw.py --case s --edit S5 --ids w/tests.json --out w/tests_S5.jsonl --n 4 --salt rt > t10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python derive/draw.py --case s --edit $BEST --ids w/tests.json --out w/tests_$BEST.jsonl --n 4 --salt rt > t11.log 2>&1   # unless that file exists
```

(`$BEST` has its dev run from the earlier steps; an `S3` that was never run on dev is run now on GPU 2 first:
`python -m wsp.case run s $BEST dev --n 2`.) When all of them have ended:

```
python derive/decide.py round --best $BEST
```

It prints the dev table (`derive/compare.py`), the line of the test requests (`derive/pair.py`) and the decision.
(The test requests of the guards are practice requests, and the round has drawn on them like on the others. Their line is therefore not a held-out reading; it is there to
show harm.)
**The round is kept (`BEST=S5`) unless the readings show clear harm: over all sources of dev it is below `$BEST` by
more than three times the standard error, or on a single source by more than four times its own, or on the test
requests by more than twice theirs, or the share of answers cut at the limit rises by more than two points.** The
sets of the run cannot show what the round gains (64 problems of a source on dev); they can show a fit that went wrong, and that is what the rule is for. If the decision is `BEST=S5`,
write it into `w/env.sh` and source it; `S5` is uploaded already. If it is `BEST stays`, upload `$BEST` again (the
five commands of step D), so that the last complete upload is the edit the readings support, and say so in
REPORT.md.

**If fewer than 60 minutes are left when the round ends**, the upload of `S5` with its report is the last act of the
run: no readings, the round is delivered, and REPORT.md says that it was not read.

**Step G: deliver**

The last complete upload is the delivered edit: `S5` if step H kept the round, otherwise `$BEST` as step H left it.
If at least 45 minutes are left after step H's decision, confirm on the holdout split (the start of case `s` is the parent with `m0`):

```
python -m wsp.case freeze s $BEST
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run s $BEST holdout320 --n 2 > hold_best.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m wsp.case run s base holdout320 --n 2 > hold_base.log 2>&1 &
```

and, when both have ended, `python derive/compare.py $WSP_WORK/runs/s/base__holdout320.jsonl
$WSP_WORK/runs/s/${BEST}__holdout320.jsonl`. With less time, leave the holdout out and say so.

The final REPORT.md covers: the parent; the lines of `start.py` and of `dirs.py`; the start (parent + `m0`) on dev,
its accuracy and cut share; the clock screen and its choice; the measurements of every builder; every
`DECISION` line of `derive/decide.py` with the table above it; the lines of the round's stages (how often the
recombined requests are followed, the share of thinkings with a mixed outcome and the pairs per kind, the neurons,
the payload); the dev table and the test requests of `S5` against the rules' stack, if they were drawn; the holdout table if it was run;
what was dropped and why. Upload again with the final REPORT.md (the five commands of step D with the delivered
edit's name), so that the last complete upload is that edit with its report. After this upload nothing of the
method is left to run: end the run as the brief says.

## 5. Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | what runs |
|---|---|
| 0:00-0:20 | setup: the bundle; `start.py` (the units silenced); case `a` and its start built; the pool started |
| 0:20-0:45 | **`m0`**: the pool (all four GPUs), `dirs.py` (GPU 0), `start.py`; cases `u` and `s`, the start of case `s` built |
| 0:45-1:25 | start on dev x2 (GPUs 0-1); the four trace lists (GPUs 2-3, two after two); building requests and test requests on the start (GPUs 0-1) |
| 1:25-1:40 | clock reads (GPU 0), then clock series (GPUs 0-3); screen |
| 1:40-1:55 | build X, the guards and the direction; stack S1; `final.py --verify`; admission check; **first upload (F_S1)** |
| 1:55-2:15 | S1 on dev x2 (GPU 0); X's own answers (GPUs 1-3); then the test requests on S1; `decide.py guards`, `decide.py exit` |
| 2:15-2:35 | build K (GPU 2); S2 on dev x2 and on the test requests; `decide.py commit`; upload `BEST` |
| 2:35-5:05 | **step H: the round** (`round.py`, all four GPUs) |
| 5:05-5:15 | **upload F_S5** with the final REPORT.md; at this pace fewer than 60 minutes are left, so by step H this is the last act of the run |
| only if 60 minutes or more are left when the round ends | S5 on dev x2 and on the test requests; `decide.py round`; holdout320 if 45 minutes are left after it; REPORT.md, final upload |

Steps A to F as written here are the steps of this miner's entry 0009554926-0017; that entry's run on the sandbox's
four GPUs went through them and a holdout in 2 h 16 min, and the times of the rules above are that run's. The
round's times are estimates: the author's run on faster GPUs scaled by the rates
the published entries 0009550031-0005 and 0009559255-0006 give for their rounds on 24 GB cards. Because they are
estimates, the round keeps the time itself (step H): it runs fewer fits when `fit1` shows that this machine is too
slow for all of them. What gives way when the run is late:

- the round is never started later than its own plan allows (`--minutes-left`: 169 minutes at least), and nothing
  else is run on the GPUs while it runs;
- if fewer than 60 minutes are left when the round ends: upload `F_S5` and end the run (step H);
- **the first engine start fills a compile cache.** Generation engines that start at the very same moment on a
  fresh machine can fail on it (`Engine core initialization failed`, a `FileNotFoundError` under
  `torch_aot_compile`). In step A start the dev run first, wait until `nvidia-smi` shows its two GPUs loaded (one
  to two minutes), then start the trace lists; a generation job that still ends this way is started again at once
  (it costs a minute, a missing file later costs a step). `round.py` restarts a stage of its own that fails;
- the holdout of step G and the drawing without the presence penalty of step A are the first readings to leave out;
- whatever the rate, the first upload of step D goes out before anything of step E is waited for, and each later
  `BEST` is uploaded as soon as it is decided, so a complete upload is always in place.

The forward passes of steps B, C and F hold answers of up to 32,768 tokens (about 3 GiB of GPU memory each, one at a
time); `commit.py` reads at most 450 long answers and 300 others. If it has not ended 45 minutes after its start,
stop it and go on without a commit (`decide.py commit` then prints that there is none).

## 6. What to expect

From the author's runs of these scripts and of their parts, on practice problems. Each figure says which start it
belongs to.

- **This entry's steps on its parent** (the author's run of sections 3 and 4 as written; 114 minutes on
  GPUs several times faster than the sandbox's). `start.py` and `dirs.py`: the lines of section 3. Start (parent +
  `m0`) on dev (320 problems x 2): accuracy 75.0%, cut share 2.5% (mathematics 11.7%). Screen:
  budget 0.16 (89% of the never-closing answers cross before 28,000 tokens, median 23,500; none of the
  early ones). Exit: `never_closing_answers_reached` 1.0, `exit_gate_relative_size` 0.30. Guards:
  this start breaks the comma instruction in 6 of its 125 building answers, placeholders in 5 of 135 and quotation marks in 3 of 123, and `guard.py` prints `NO GUARD` for these three; it breaks highlighted sections in 8%, and that guard was built (gate: `recall` 0.82). On the 192 test requests `S1` read 91.9% -> 91.3% (-0.7 +- 1.0) and the guard's line was not above its standard error: not kept. `S1` on dev: cut share 2.5% -> 0.6% (mathematics 11.7% -> 3.1%), all sources +0.3 +- 1.4; the exit
  qualifies. Commit: 166 pushed closes read, `commit_recall` 0.92,
  `commit_fires_after_the_models_own_closes` 0.08, `last_layer_gate_relative_size` 0.29. `S2` on dev
  against the start: cut share 2.5% -> 0.2%, all sources +2.0 +- 1.4; against `S1`: +1.7 +- 1.5, no source below by twice
  its error, 1.1% against 1.9% of the answers cut or without a result: taken by rule 1. Direction:
  `NO CONCLUDING DIRECTION`. `final.py --verify`: `same model: True (488 tensors compared: bit for bit)`.
  `axonry-check` on `F_S2`: passed; payload 98,368 bytes; model 0.042 overall; largest tensor change
  0.30.
- **The round in that run** (`round.py --minutes-left 230`: one set of recombined requests). The model follows
  62.6% of the 1,320 recombined requests; 302 + 87 + 65 pairs; 124
  mathematics anchors; 656 neurons over 22 layers, most of them in layers 10 and 6; 58 minutes for the whole round. `axonry-check` on `F_S5`: passed; payload
  4,132,768 bytes of 4,194,304; model 0.042 overall; largest tensor change 0.30. (That run was stopped
  after the upload of `S5`: the readings of step H and the holdout were not drawn.)
- **What the sets of the run can show.** Dev has 64 problems of a source and the test requests are 192: a
  standard error of 2.5 to 4.5 points for a source and about 1.3 for the test requests. The round's effect on
  the instruction requests is large enough to be read there; its effect on mathematics, word problems and multiple
  choice is not, in either direction. One fit differs from the next (the finding of 0009550031-0005, and the reason for four fits and their average), and
  so does one round from the next.

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
- **On a start without wrap-up units of that kind** (the author's run of steps A to F on the published edit
  0009552724-0008, which stands on the same published edit as this entry's parent; 47
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
  on `S2`: passed; payload 74 kB; model 0.042 overall; largest tensor change 0.30.
- **The commit is decided by step F, and both outcomes occur.** In the run above `S2` lost to `S1` on the 64
  mathematics problems of dev. In the author's runs of the three pairs alone on 0009543609-0016 the commit was taken, and on 800 mathematics problems (section 1.1) the three pairs together are one point ahead of
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
- **A higher dose of a round without anchors.** The author's earlier readings of a round on this line without mathematics pairs and anchors (0.35 / 0.5 / 0.7 of the
  averaged change, on 2,560 practice problems outside the fit: all sources +1.7 +- 0.6, +0.8 +- 0.6, +1.1 +- 0.7;
  mathematics +1.0 +- 1.1, -0.5 +- 1.2, -0.1 +- 1.4): there more of the change bought nothing outside the instruction
  requests and cost mathematics first, and the finished entry 0009559255-0006 reports the same of its round. With the
  anchors in the fit this entry delivers 0.45, and no more.
- **Units for the two instructions the recombined requests break most.** A guard pair of section 2 for "at least N
  words" (the end-of-turn token lowered while the answer is short) and one for the repeat instruction (the first word
  of the instruction sentence lowered at the start of the answer), built and tested like the other guards: +0.0 +-
  2.9 on 18 kept practice requests and -2.8 +- 5.0 on 9. What these requests need is not a token family.
- **The parent's wrap-up units left in place.** In the author's run of these steps on the parent as it is (no
  `m0`), the start's cut share on dev was 1.1% (7 of 640 answers) and the exit, break and commit pairs were
  not built: what the method then added was a guard, its own concluding direction (step F2) and the round. Section
  1.6 has the practice reading of the parent as it is against this method's start with its rules.
