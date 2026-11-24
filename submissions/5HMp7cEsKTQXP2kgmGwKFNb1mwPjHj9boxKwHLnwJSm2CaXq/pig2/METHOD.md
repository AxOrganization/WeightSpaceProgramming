# A repair layer and a word-problem layer beside the declared parent's closing units, each fitted in the run from the model's own answers: a failing answer mended by a program and accepted by the practice checker is preferred to the answer as sampled, and an own right answer to a word problem is preferred to an own wrong one

Parameters of this method. They are the defaults at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
DOSE_E=0.35          # the repair layer: share of its averaged fits' change that is delivered
SEEDS_E="0 14"       # the repair layer's fits, on neurons chosen once (with the first seed)
NEURONS_E=650        # MLP neurons of the repair layer
E_START="base"       # the repair layer's model: its answers make the layer's data and it is fitted on it
POOL=8               # answers per practice instruction request (the repair layer's model answers them)
E_SELECT="pass"      # the repair layer's neurons are chosen in a pass without training
MAX_LEN=20480        # token limit of a row in the repair layer's fits
DOSE_G=0.25          # the word-problem layer: share of its averaged fits' change that is delivered
SEEDS_G="7 35"       # the word-problem layer's fits, on neurons chosen once (with the first seed)
NEURONS_G=650        # MLP neurons of the word-problem layer
GSM=1600             # practice word problems the start answers for the word-problem layer
GSM_N=8              # answers per word problem
```

This entry starts from the champion (it was written while `0009570269-0024` held that place) and names
`0009569144-0010` as its parent in `entry.json`. It delivers one edit made of three parts, all on the champion's
model:

- **U, the parent's start changes and closing units.** Twelve of the 78 ops that `0009569144-0010` added to its
  own start: two ops that silence two wrap-up units of `0009554110-0010` (the edit both the parent and the champion
  stand on), four direction removals on attention outputs, and three rewritten MLP unit groups (an exit pair at
  layer 6, a break pair and a commit pair at the last layer). They are read from the parent's published
  `edit.json` and put on the champion's model as they are (`derive/take_units.py`). Nothing is recomputed and
  nothing is refitted. The parent's fitted layer (its other 66 ops, 656 MLP neurons) is **not** taken, and its
  answer round is not run.
- **E, a repair layer** (650 MLP neurons, dose 0.35), made on the champion's model: from its own answers and fitted
  on it. The champion's model often breaks a typed instruction that it could satisfy: it writes the answer in
  capitals but leaves one word in mixed case, repeats the request with a line missing, uses a letter once too
  often. For such an answer a small program can make the missing step (upper-case the answer, put the exact request
  first, drop the sentence that carries the letter), and the practice checker says whether the mended answer now
  follows every instruction of the request. Where it does, the mended answer is preferred to the answer as sampled,
  after the same thinking. These repair pairs, pairs of an own passing answer over an own failing answer to the same
  request, and own passing answers to keep go into one preference fit, on requests written in the run with two and
  three instructions and on the practice requests.
- **G, a word-problem layer** (650 MLP neurons, dose 0.25), made on the champion with U. That model answers a third
  of the practice word problems sometimes right and sometimes wrong. For such a problem its own right answer (the
  shortest) is preferred to its own wrong answer, whole answers, each with its own thinking; its own passing answers
  to instruction requests are kept as they are.

The two layers are fitted independently, each on its own model, and each is kept off every neuron the champion's
edit and U name. U therefore leaves the repair layer's neurons as they are: their weights are the same in the
champion's model and in the champion with U, and the layer is put on U as it was fitted (the rest of the model it
then acts in carries U's changes). The two are joined into one neuron set (a neuron both fits chose gets both
changes) that is written as int8 directions and stacks on the local edit U.

*Why the repair layer is made from the champion's own answers, on the champion's model.* That is the form in which
we measured it on practice data (section 2.4). A sibling run of ours made it from the answers of the champion with
U, on that model (section 2.5): its rows are the units' model's own answers, the chosen answers start at a lower
held-out loss, and the fit moves them about half as far (0.320 -> 0.251 and 0.261, against 0.42 -> 0.27 and 0.26).
The rows made from the champion's own answers move as far on the champion with U as on the champion's model (0.43 ->
0.27 and 0.27 in a fit of ours), so it is the rows, not the model they are fitted on, that make that difference. We
have no practice reading of the other form alone. `E_START="U"` in the block above makes it (section 6 says what
each setting costs).

**What is the parent's, what is published, and what is ours, plainly.**

*U is the declared parent's, taken unchanged.* The ops are copied from the published edit of `0009569144-0010`;
this entry adds nothing to them and claims nothing about them beyond the readings of section 1.

*The kinds of pair are not new, and we say so.*
- A failed answer mended by rule, graded again and preferred to the answer it was made from, after the same
  thinking, is the published idea of `0009565562-0012` (its `derive/repair.py`: 14 remedies that write no word the
  request does not itself give, on requests recombined from the practice sets, one pair per request, fitted
  together with sampled answer-branch pairs). Of that entry this one uses the scheduler (below) and none of its
  repair files: `derive/repair_rows.py` is written anew. What this entry adds to the idea:
  - **repairs for the instructions that need text added or taken away**, which the earlier remedies leave out by
    their rule: a count of letters, keywords that must appear or appear N times, "at least N" words or sentences,
    placeholders, a postscript, a title, numbered sections. The added text is the answer's own wherever that is
    possible (a recap made of its own sentences, a list of its own words that carry the letter); 23 instruction
    types have a repair (section 2.2);
  - **verification under five seeds of the checker's language detector**: the practice checker's all-capitals and
    all-lowercase checks call a detector that is random unless seeded, and one grading of a short answer is a coin;
  - **requests with two and three instructions written in the run** as the main source (that is where this model
    fails), beside the practice requests as they are, with up to four repaired answers per request;
  - **what such a layer does and where it breaks**, measured (section 2.4): fitted on repair pairs alone, the added
    closing text becomes a habit and breaks requests that fix how an answer must end; own pass-over-fail pairs in
    the same fit hold that; a larger dose does not help.
- An own right, finished answer preferred to an own wrong, finished answer of the same practice problem, whole
  answers, with own passing answers to instruction requests as answers to keep, is the method of our own published
  entries `0009568016-0005` and `0009571168-0014`, where the problems were practice math problems;
  `derive/build_pairs.py` is their file, byte for byte. Here it is given the start's answers to practice **word
  problems** (section 3). The trainer, its objective and the averaging of fits are the published work of the
  author of `0009550031-0005` (`derive/rollouts.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`,
  `derive/dose_sparse.py`, unchanged).

*This is our fourth entry on these units* (same author; among the earlier ones `own-two-outcomes` and `pig1` were
in progress when this was written). What it shares with them: U and the way it is taken
(`derive/take_units.py`), the list of neurons to stay off (`derive/exclude_neurons.py`) and the scheduler
(`derive/pipeline.py`). It shares no fitted layer with them: `pig1` fits the parent's answer round on these units,
this entry does not run that round at all.

*New in this entry:* the two layers fitted in one run, each on its own model, and joined in one edit on U; the
word-problem layer in this form (on the champion with U, from word problems); and the repair layer. These files are
ours: `derive/repair_rows.py` (the repairs and their verification), `derive/mix_rows.py`,
`derive/passfail_rows.py`, `derive/wreq.py`, `derive/payload.py`, `derive/screens.py`, and `derive/units_check.py`
(whether the champion already carries the units, section 4).

Where the other files come from. `derive/reqgen.py` (the request writer; here with another seed than that entry's
run), `derive/union_sparse.py` (the joining), `derive/encode_dirs.py` (the int8 form) and `derive/cmp.py` are files
of `0009570269-0024`, unchanged. `derive/genreq.py` and `derive/train_req.py` are files of the declared parent's
`derive/` (the published work of the author of `0009559255-0006`), unchanged. The scheduler in
`derive/pipeline.py` is that of `0009565562-0012` as used in `own-two-outcomes` and `pig1` (with its stop of
engines that hang at shutdown); its job list is this entry's.

Every number below is our local measurement on practice data, for orientation; where a number of a step comes from
another run than a run of exactly this pipeline, the text says which. No arrays are shipped: U is read from the
parent's directory in the bundle, and both layers are derived in the run from practice data, requests written in
the run and the own generations of the champion's model and of the champion with U.

## 1. The units on this start

The parent `0009569144-0010` stands on `0009554110-0010`, and so does the champion: the two wrap-up units the
parent silences are in the champion's model, and its closing units (the exit pair at layer 6 that pushes a long
thinking to close, the break and commit pairs at the last layer that bring the result at once after such a close)
act on the same residual stream. On the champion's model, in our builds (4 answers each, the profile's own token
limit, paired with the champion's model on the same problems and seeds):

- 300 practice instruction requests of the dev and holdout splits: 91.8% -> 92.1% followed (+0.3 ± 0.9);
- 300 fresh practice math problems: 65.9% -> 67.5% right (+1.6 ± 1.1); with U 0.7% of the answers are cut at the
  limit, 4.7% finish without an answer, and an answer is 12,700 tokens long on average;
- the largest relative change of a tensor is 0.304 (`layers.6.mlp.gate_proj`) and 0.278
  (`layers.23.mlp.gate_proj`), the same values the parent has on its own start.

## 2. The repair layer

### 2.1 The behaviour, and its cause as far as we measured it

The champion's model (without the units), practice data, four answers per request at the profile's own token
limit.

| requests | every instruction followed |
|---|---|
| 300 practice instruction requests (150 `ifsynth`, 150 `tuluif`) | 90.4% |
| of them the 222 with one instruction / the 78 with two | 93.4% / 82.1% |
| 300 requests written by `derive/reqgen.py` (its dev split; 100 each with one, two, three instructions) | 86.2% |
| of them with one / two / three instructions | 96.8% / 89.5% / 72.2% |

The failures sit in a few mechanical instruction types. Share of answers that follow the instruction, on the two
sets (requests carrying it in brackets):

| instruction | practice requests | written requests |
|---|---|---|
| a letter must appear at least / fewer than N times | 61.1% (18) | 67.6% (27); "fewer than": 22.5% (10) |
| the whole answer in capitals | 64.5% (19) | 76.0% (25) |
| N paragraphs, the i-th starting with a given word | 91.7% (9) | 76.9% (26) |
| N numbered sections | 94.8% (24) | 82.4% (27) |
| exactly N bullet points | 96.7% (15) | 86.5% (26) |
| at least / fewer than N words | 91.3% (23) | 87.5% (26) |
| words in capitals at least / fewer than N times | 94.1% (17) | 88.0% (27) |

What we know about the cause, from reading the 115 failing answers of the 300 practice requests with their
thinking:

- **The model does not forget the instruction.** In every one of the 115 the thinking restates the instruction it
  then breaks. It checks, and the check is wrong: 47 are miscounts (letters, words, highlighted parts; "g-r-o-w-s.
  No 'w'"), 22 carry the violation in a draft inside the thinking that the thinking's own check approved, 12 are
  slips made while writing the answer down, 7 are requests nobody can satisfy.
- **25 are answers in correct capitals or lowercase that the practice checker's language detector does not read as
  English.** The detector is random unless seeded; an all-capitals answer under five words passes it once in 14,
  one of 10 to 20 words 13 times in 18, one of more than 40 words always. Under a case instruction this model
  answers tersely ("255 SECONDS REMAINED.").
- **The answer is close.** Three quarters of the failing answers are a copy of a draft in the thinking, and what
  separates most of them from an answer that passes is one mechanical step: section 2.2 counts how often a program
  can make that step.

So after its thinking the model writes an answer that is one step short of what its own thinking set out to do.
A pair of two sampled answers needs another sample of the same model to make the step; for a request whose
answers all fail there is none. A repaired answer is that missing preferred side.

### 2.2 The repairs

`derive/repair_rows.py` takes every finished answer that breaks an instruction, keeps its thinking as it is, mends
the answer part for each broken instruction, grades the result with the practice checker, mends again what the
first repair broke (up to five passes), and keeps the mended answer only if it then follows **every** instruction
of its request under five seeds of the detector.

| instruction | repair |
|---|---|
| all capitals / all lowercase | the answer upper- / lower-cased; where a correctly cased answer is not read as English, one plain closing sentence in the same case (one of six wordings) |
| repeat the request | the exact request first; the model's inexact copy taken out |
| at least N words / sentences | a recap made of the answer's own first sentences |
| fewer than N words / sentences | the last sentences dropped |
| keywords must appear | an inflected form replaced by the exact word, else one closing sentence naming them |
| a keyword N times | closing sentences with the word; or the sentences with surplus uses dropped |
| a letter at least N times | a closing list of the answer's own words that carry the letter |
| a letter fewer than N times | the sentences that carry it most dropped (at least 40% of the words stay) |
| capital words at least / fewer than N | own words raised to capitals / surplus ones lowered |
| forbidden words; no commas | deleted |
| end phrase; quotation marks | what follows the phrase cut, or the phrase appended; the answer wrapped |
| postscript; placeholders; title | a postscript line; a sign-off of bracketed placeholders; the title line bracketed |
| highlights; bullets; sections; `***` paragraphs; i-th paragraph | first words of plain sentences starred; surplus bullets joined or numbered lines turned into bullets; "SECTION ONE" written "SECTION 1"; dividers added or removed; paragraphs joined or split |
| JSON; two responses | comments and text around the object removed; the separator normalised |

How many failing answers that mends (finished answers that broke an instruction; a request counts once per
answer):

| source (the champion's model) | failing answers | mended and accepted |
|---|---|---|
| 1,748 practice requests, eight answers each (13,984 answers) | 1,208 | 951 (79%), on 377 requests |
| 1,200 written requests with two or three instructions, four answers each (4,800) | 939 | 478 (51%), on 289 requests |

These are the answers the delivered form of the layer was fitted from (section 2.4). The practice pool there was
an earlier pool of ours, 1,748 practice requests of another case; in a run of this pipeline it is the practice
requests outside the case's dev and holdout splits (about 1,620), `POOL` answers each at a 12,288-token limit.

On the champion **with U** (the form `E_START="U"` makes), in a full run of a sibling pipeline of ours on this start
(the same repair layer jobs; the written requests as above, and the 1,622 practice requests outside the dev and
holdout splits with four answers each, two of them at a 6,144-token limit): 1,430 finished failing answers among
11,288, **872 mended and accepted** (467 of written requests, 405 of practice requests), on 519 requests. By the
number of instructions in the request: one 281 of 330, two 301 of 469, three 290 of 630.

By instruction (mended / failing; practice requests, written requests; the first two sources): all capitals 279/281,
104/144; letter count 31/77, 79/160; repeat the request 130/130, 10/10; all lowercase 103/108, 4/10; keywords 75/77,
14/34; word count 31/57, 31/69; sentence count 34/49, 28/58; highlights 32/60, 39/47; capital words 25/32, 38/52;
keyword frequency 31/32, 35/37; placeholders 32/37, 26/30; `***` paragraphs 21/23, 28/50; bullets 24/48, 31/88;
quotation 38/39, 9/14; end phrase 33/33, 4/6; i-th paragraph 11/36, 15/140; sections 1/42, 7/93; two responses 4/19,
0/4; JSON 1/13, 1/1. The program does not know how to write a missing section or a second response, and it does not
put a required first word in front of a paragraph.

### 2.3 The fit

Rows (`derive/mix_rows.py`, `derive/passfail_rows.py`; the numbers are the caps, constants of
`derive/pipeline.py`):

- up to **700 repair pairs**: after the same thinking, the mended answer is chosen and the answer as sampled is
  rejected. Written requests first (up to four per request), then practice requests (up to two); at most 90 whose
  only repair is the plain closing sentence;
- up to **450 own pairs**: an own passing answer chosen over an own failing answer of the same request, whole
  answers, one pair per request (270 from written requests, 180 from practice requests);
- **250 anchors**: own passing answers to other requests, as answers to keep.

One preference fit with the published trainer and objective (`--objective dpo --select margin --epochs 2 --accum 2`)
on the champion's model, rows up to 20,480 tokens: 650 MLP neurons chosen once in a pass without training (the
gradient of the chosen / rejected margin; never a neuron of the champion's edit or of U, never the last layer), two
fits of those neurons (seeds 0 and 14) averaged, 0.35 of the change delivered, measured from the champion's model.
The fits' spec names U as its parent: the layer is written as int8 directions on the champion with U (the edit
`FE`), where its rows are the champion's (none of them is one of U's).

Our fit of this form (the rows of the first table of section 2.2): `examples: train 1355 (pairs 1117), val 45`, no
row longer than the limit, 650 neurons over 22 layers (most in layers 10, 11, 14, 12 and 13), held-out loss of the
chosen answers 0.42 -> 0.27 and 0.26 (seeds 0 and 14), 30 minutes a fit on two GPUs. Two differences from a run of
this pipeline: there the first seed's fit chose the neurons at its start (the same computation as the pass, with
the same seed; `E_SELECT="fit"` does it that way, at the cost of the fits running one after the other), and the
neurons were also kept off the parent's fitted layer and off a layer of ours that this entry does not deliver.

### 2.4 What the layer did on practice requests it had not seen, and where it breaks

Measured with the layer in the form this entry makes it (fitted on the champion's model from its own answers, the
fit of section 2.3; 1,400 rows) and read on that model **without the units**, against it, four answers per
request, paired by request and seed. Gain in points of "every instruction followed", with its standard error:

| fit | rows | dose 0.35 | dose 0.7 | dose 1.0 |
|---|---|---|---|---|
| **repair pairs + own pairs + anchors (delivered)** | 700 + 450 + 250 | **+2.1 ± 1.0** | +1.8 ± 1.0 | +0.5 ± 1.4 |
| repair pairs + anchors | 1,000 + 300 | +1.8 ± 1.0 | -2.9 ± 1.5 | -9.0 ± 1.9 |
| repair pairs, pairs at the draft (below) + anchors | 700 + 406 + 250 | +1.8 ± 1.1 | +1.8 ± 1.2 | -9.1 ± 1.8 |
| mended answers as plain likelihood rows + answers of other tasks held by a keep term | 1,000 + 713 | +0.1 ± 0.9 | +1.1 ± 0.9 | +1.2 ± 1.1 |
| repair + own + counter pairs (below) + anchors, on the delivered fit's neurons | 500 + 250 + 300 + 150 | +2.4 ± 0.9 | +1.1 ± 1.0 | not read |

(300 practice requests, 150 `ifsynth` and 150 `tuluif`, none of them in any fit; the champion's model follows
90.4%.) The delivered fit, by instruction, at 0.35:

| | requests | champion's model | with the layer | gain |
|---|---|---|---|---|
| all | 300 | 90.4% | 92.5% | +2.1 ± 1.0 |
| one instruction | 222 | 93.4% | 94.4% | +1.0 ± 1.0 |
| two instructions | 78 | 82.1% | 87.2% | +5.1 ± 2.6 |
| a letter count | 18 | 61.1% | 70.8% | +9.7 ± 8.8 |
| all capitals | 19 | 64.5% | 71.1% | +6.6 ± 7.1 |
| a word count | 23 | 91.3% | 96.7% | +5.4 ± 2.7 |
| keywords must appear | 29 | 91.4% | 94.0% | +2.6 ± 3.1 |
| end phrase | 29 | 97.4% | 98.3% | +0.9 ± 0.9 |
| numbered sections | 24 | 94.8% | 92.7% | -2.1 ± 1.4 |
| keyword N times | 11 | 100% | 97.7% | -2.3 ± 2.3 |

On the 300 written requests of the dev split (none in a fit; champion's model 86.2%): **+1.7 ± 1.0**; one / two /
three instructions +1.5 ± 1.1 / +1.0 ± 1.9 / +2.5 ± 2.1; all capitals 76.0% -> 86.0% (+10.0 ± 6.5); a letter
"fewer than N times" 22.5% -> 32.5% (+10.0 ± 5.5); capital words +5.6 ± 3.4; exactly N bullets 86.5% -> 78.8%
(-7.7 ± 3.3). On 304 practice puzzles 91.1% -> 91.4% (+0.3 ± 0.6), on 300 practice math problems 62.4% -> 63.3%
(+0.9 ± 1.4; two answers each with the layer).

The gain is one to two standard errors on either set; the instruction types the repairs are made for move in the
expected direction on both, each inside its own error.

**Where it breaks.** Many repairs add text at the end (a closing sentence, a recap, a word list). Fitted on repair
pairs alone, the layer learns "add something at the end" as a habit, and at dose 0.7 the requests that fix how an
answer must end or how it is laid out fail: "end with this exact phrase" 97.4% -> 73.3%, "N paragraphs, the i-th
starting with" 91.7% -> 61.1%, "exactly N bullets" 96.7% -> 83.3%; at 1.0 answers also begin to lose their answer
part (3% finish without one). With the own pass-over-fail pairs in the same fit the end phrase holds (97.4% at 0.7) and
dose 1.0 reads +0.5 instead of -9.0; that is why they are in the delivered fit. No dose above 0.35 read better.

### 2.5 What did not work (ours, practice data; none of it is delivered)

- **The mended answers as plain likelihood rows with a keep term** (the model's own answers to other tasks held by
  a KL term, the way `0009570269-0024` fits its workings): +0.1 ± 0.9 at 0.35, +1.2 ± 1.1 at 1.0. Without the
  rejected side the fit does not learn what to stop doing.
- **Pairs at the draft.** Where the failing answer stands word for word inside the thinking (271 of 700), a pair
  was added at that place: the mended text preferred to the model's own draft. Total gain unchanged (+1.8 ± 1.1 at
  0.35, +1.8 ± 1.2 at 0.7); at 0.7 all capitals read 64.5% -> 82.9% and letter counts 61.1% -> 81.9%, but word
  counts fell by 5.4 ± 2.7, the end phrase by 7.8 ± 4.7, and dose 1.0 read -9.1 ± 1.8.
- **Counter pairs**: an own passing answer preferred to the same answer with one of the repairs' closing sentences
  appended, kept only where the checker rejects the appended one (300 pairs). +2.4 ± 0.9 at 0.35 and +1.1 ± 1.0 at
  0.7 (quotation marks -9.7 ± 4.1): not different from the delivered fit at 0.35, worse at 0.7.
- **Doses 0.7 and 1.0**: the table above.

**The layer made on the champion with U, from that model's answers** (not delivered; `E_START="U"`, with `POOL=4`,
`MAX_LEN=12288` and `NEURONS_E=640`, the settings of the sibling run named in section 2.2; neurons chosen in a pass
of their own): 1,400 rows (700 repair pairs, 466 of them from written requests; 450 own pairs; 250 anchors), 640
neurons in 22 layers (most in layers 14, 12, 11 and 13), held-out loss of the chosen answers 0.320 -> 0.251 and
0.261: these rows are the units' model's own answers, the chosen answers start likelier, and the fit moves them
about half as far as the rows made from the champion's own answers move (0.42 -> 0.27 and 0.26 on the champion's
model, 0.43 -> 0.27 and 0.27 on the champion with U). That run delivered the layer joined with another layer (an
answer round of 656 neurons), so its screens read the two together, against the champion's model: 300 practice
instruction requests of the dev and holdout splits 90.6% -> 91.6% (+1.0 ± 1.1), the 300 written dev requests 84.8%
-> 86.8% (+2.0 ± 1.1), 213 practice puzzles 90.7% -> 91.0% (+0.2 ± 0.8). We have no practice reading of this form
alone, and do not deliver it.

Not measured by us on practice data: the delivered layer **read on the champion with U** (it was read on the
champion's model, the model it is fitted on); the repair layer **joined with the word-problem layer** (section
3.4); a second run (one fit pair of one draw of answers per row of the table).

## 3. The word-problem layer

### 3.1 The behaviour

The start (the champion with U) answers 1,600 practice word problems eight times each, at the profile's own token
limit (one run, 12,800 answers):

| | |
|---|---|
| a single answer is right | 78.5% |
| the answer most of the eight agree on is right | 83.8% |
| at least one of the eight is right | 93.8% |
| problems with both a right and a wrong answer | 36.4% (582 of 1,600) |
| answers cut at the limit | 0.0% |
| median length of a right answer / of a wrong answer | 4,360 / 8,299 tokens |

So for more than a third of the problems the model can produce the right answer and does not do so reliably, and
its wrong answers are the long ones: a wrong answer takes twice the thinking of a right one. The answer the samples
agree on is right five points more often than a single sample. That is the gap the layer is fitted on: nothing is
taught that the model does not already write; among its own answers the right, short one is preferred.

### 3.2 The pairs and the fit

`derive/build_pairs.py` (ours, published with `0009568016-0005`; unchanged): for every problem with both a right
and a wrong **finished** answer (not cut, thinking closed), the shortest right answer is chosen over the first
wrong one, and the second shortest over the second wrong one: at most two pairs a problem, whole answers, each
with its own thinking. 200 of the start's own passing answers to practice instruction requests (one per request)
are added as answers to keep: the start answers 450 such requests once for them, in the job `apool` (with
`E_START="U"` the answers of the job `pool` are the start's and serve instead). In the run above: 911 pairs from 578
problems, 200 anchors (of the start's 399 passing answers to 450 requests, one answer each, 12,288-token limit),
1,111 rows.

One preference fit with the published trainer (`--objective dpo --select margin --epochs 2 --accum 2`, rows up to
20,480 tokens: the wrong answers are long), 650 MLP neurons chosen once in a pass without training (never a neuron
of the champion's edit or of U, never the last layer), two fits of those neurons (seeds 7 and 35) averaged, 0.25
of the change delivered. No protected-neuron list and no keep term. In that run: `examples: train 1053 (pairs
827), val 52`, 21 to 26 rows longer than the limit left out, neurons in 23 layers (most in layers 14, 15, 9 and
13), held-out loss of the chosen answers 0.235 -> 0.297 and 0.303 (it rises: the preference pulls at the
likelihood of whole answers, which is why only a share of the change is delivered).

### 3.3 What the layer did on practice word problems it had not seen

Read with the same recipe **fitted on the champion's model without the units** (its own pool: 949 pairs from 595
of 1,600 problems; nine fits of that recipe, each the average of two seeds), on 300 fresh practice word problems,
four answers each, two such sets, paired with the champion's model:

- **dose 0.25**: +0.2 to +3.4 points over the nine fits (standard error 1.0 to 1.2 each); the fit with the seeds
  of this entry read 76.4% -> 78.1% (+1.7 ± 1.2) and 74.7% -> 76.6% (+1.9 ± 1.0). The thinking gets longer (mean
  6,500 -> about 8,200 tokens), and 3 to 5% of the answers run into the token limit, on a model without the units.
- **dose 0.5**: -0.3 to -5.8 points, with 11 to 20% of the answers cut at the limit. The layer makes the model
  think longer, and past 0.25 it thinks past the limit. 0.25 is the dose the practice screens allow.

The units of the start are what closes a long thinking (section 1), and the start's own pool has no cut answer at
all; that is why this layer is fitted on the champion with U and not delivered without U.

### 3.4 Not measured

Not measured by us on practice data: a practice screen of the word-problem layer fitted on the champion **with
U**, alone; **the two layers made in one run and their union on U** (each layer was fitted in a run of its own,
the repair layer in the run of section 2.3 and the word-problem layer in the run of section 3.1; the union was never
read on a practice set); a second run of the word-problem layer on this start (one pool, one pair of fits). The
screens of section 5, step F, are the first reading of the delivered edit.

## 4. Setup (first 25 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009569144-0010/edit/` and the champion `<its id>/edit/` (what each builds on is inside it). On the
agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009569144-0010 sandbox:/workspace/work/   # always: the units are read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on.** This entry was planned while `0009570269-0024` was the champion.

- The champion in `task.json` is `0009570269-0024`: start from it and make both parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one (`--parent
  <champion id>/edit/edit.json` at `init`). Take the units as below, with two exceptions: if `0009569144-0010` is
  in the new champion's lineage it carries them already, and if `0009554110-0010` is not in its lineage the two
  silencing ops have nothing to silence. In both cases skip `take_units.py`.
- **The champion may carry the units without the parent in its lineage.** An entry that put the parent's units on
  `0009570269-0024` as ops of its own (`0009576150-0005` and `0009576553-0024` do) carries them as its own ops once it
  is champion, and its lineage does not name `0009569144-0010`. Taking them again would remove the four attention
  directions a second time and write the unit rows over whatever that entry put there. So `take_units.py` runs only
  after `derive/units_check.py` (below) has said `TAKE`. It reads the two edits of the bundle and nothing else, and
  says `SKIP` (exit code 3) when the parent is in the champion's lineage (`FLAT.json` beside the champion's
  `edit.json`) or when the champion's own edit holds the parent's unit and start ops (an op of the same kind on the
  same tensor that sets the same rows or columns to the same values, or removes the same direction). Some but not
  all of them found: `SKIP` as well, with the ops named: say so in the report. It also says when
  `0009554110-0010` is not in the champion's lineage.
- Only one start in the bundle (`0009569144-0010` is the champion, or no champion is offered): `--parent
  0009569144-0010/edit/edit.json` at `init` (leave out the champion's `scp` line), skip `take_units.py`.

Whenever `take_units.py` is skipped there is no edit `U` in the case, and the pipeline then samples, fits and
delivers on the case's own start without any other change: `F` is the two layers alone, and it is uploaded the same
way. (Neither layer was fitted by us on a start other than `0009570269-0024`, with and without U; without the
units the word-problem layer leaves 3 to 5% of the answers to word problems cut at the limit, section 3.3. When the
champion carries the units itself, both layers are made on it and from its answers: for the repair layer that is
close to the form of section 2.5, whose rows the fit moves less.)

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
P="gsm8k_train math_train mmlupro ifsynth tuluif"
python -c "from wsp import data; data.load('practice:bbhsynth')" 2>/dev/null && P="$P bbhsynth"   # if the toolkit has it
python -m wsp.case init rq --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json --practice $P --dev 480 --holdout 800
python derive/units_check.py --champion <champion id>/edit --from 0009569144-0010/edit && \
  python derive/take_units.py --case rq --from 0009569144-0010/edit --name U     # the units as the case's edit U, unless the champion has them
python -c "from wsp import case; print(case.model_path('rq', 'base')); print(case.model_path('rq', 'U'))"   # build both ONCE, before any GPU job
# (when take_units.py was skipped, build 'base' only: there is no edit U then)
```

- `units_check.py` prints one line per unit or start op of the parent (`not in the champion edit` for each of the
  twelve on `0009570269-0024`), then `champion 0009570269-0024: 8763 ops, lineage [...]; parent 0009569144-0010: 12
  unit / start ops, 0 of them found in the champion's edit` and `TAKE: ...` (ours). On `SKIP` the `&&` leaves
  `take_units.py` out.
- `take_units.py` prints `parent 0009569144-0010/edit: 78 ops on '0009554110-0010' = 12 unit / start ops (taken) +
  66 ops of its fitted layer (left out; ...)`, then the twelve ops it took: `set_cols` on `15.mlp.down_proj` and
  `16.mlp.down_proj` (one column each), `dir_ablate` on `self_attn.o_proj` of layers 11, 15, 19 and 23, and
  `set_rows` / `set_rows` / `set_cols` on `gate_proj`, `up_proj`, `down_proj` of layer 6 (two units) and of layer 23
  (four units); last `{"taken": 12, "left_out": 66, "payload_bytes": 98368}`. If it stops because the parent's
  edit is not made as described, do not force it: go on without the units (delete nothing; the pipeline sees that
  there is no edit `U`) and say so in the report.
- The two model builds take about seven minutes each on the CPU.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/reqgen.py` and the models' own generations. Never load
  `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 5. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the checks, the uploads, the report.

```
setsid nohup python derive/pipeline.py --case rq --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
python derive/pipeline.py --case rq --out /workspace/work/rd --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case rq --out /workspace/work/rd --dry-run     # every command it runs, for reference
```

The minutes are ours on four RTX 4090s, from three runs: the generation jobs of the repair layer from the full run
of the sibling pipeline of section 2.2 (run A, the same jobs on the champion with U, two answers per practice request
where this pipeline makes eight: scaled), its fits from the fits of section 2.3 (run C: 30 minutes a fit on two
GPUs), the word-problem layer's jobs from the run of section 3.1 (run B: eight GPUs shared with other work, minutes
given for four). The clock is their sum with 25 minutes of setup in front; this exact job list was run by us from
end to end only as rehearsals with a twentieth of the requests and problems.

| jobs (logs in `/workspace/work/rd/logs/`, files in `/workspace/work/rd/w/`) | what they do | GPUs | minutes (ours) | done at about (clock of the run) |
|---|---|---|---|---|
| `exclude`; `wreq`; `screens` | the neurons the champion's edit and U already use (`w/X.json`); the written requests (`w/wreq.json`, `w/wscr.json`); the screen sets (step F) | none | 1 | 0:26 |
| `warm` | eight requests answered by one engine alone: the compiled kernels every later job starts from | 1 | 2 (A) | 0:28 |
| `pool` | the champion's model on the practice instruction requests as they are, eight answers each | 4 | 23 (A: 7 for two answers each) | 0:52 |
| `wgen` | the champion's model on the 1,200 written requests, four answers each; then the repairs, their verification and the repair layer's data `w/pairs_e.jsonl` | 4 | 14 (A: 25 on two GPUs, and under 1 for the rows) | 1:07 |
| `ESEL` beside `apool`; then `E0` and `E14` side by side | the repair layer's neurons; the start's answers to 450 practice instruction requests (the word-problem layer's answers to keep); the repair layer's two fits | 2 and 2; 2 each | 4 (A) and about 4 (B: 2 on eight GPUs); 30 (C) | 1:42 |
| `FE` | the repair layer's average and dose, the int8 form: the edit `FE` on `U`; payload | none | 1 (A) | 1:43 |
| `gpool` | the start on 1,600 practice word problems, eight answers each; then the word-problem layer's data `w/pairs_g.jsonl` | 4 | 47 (B: 197 GPU-minutes), and under 1 for the pairs | 2:31 |
| `GSEL`; then `G7` and `G35` side by side | the word-problem layer's neurons; its two fits | 2; 2 each | 9; 38 to 42 (B) | 3:22 |
| `FG`; `F` | the word-problem layer's average and dose (`FG`); both layers joined, the int8 form, the payload check | none | 1; 2 (A, for a union of the same size) | 3:25 |
| `build` | `F`'s model built for the screens (an upload does not wait for it) | none | 9 to 21 (A and an earlier run) | 3:45 |
| `base_ifscr`, `base_wscr`, `base_gsmscr`, `base_puzscr`, `base_mathscr` | the champion's model on the screens, from the end of the fits | 4 each | 4; 4; 4; 4; 18 (A; `gsmscr` not run by us at full size) | 4:00 |
| `F_ifscr`, `F_wscr`, `F_gsmscr`, `F_puzscr`, `F_mathscr`; `guard` | `F` on the same sets; the paired comparisons, in `logs/guard.log` | 4 each | the same | 4:35 |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not; the rollout jobs keep what their workers had
  finished, and `wgen` does not sample its answers again when it finds them complete).
- **Each layer's chain stands alone.** No job of one layer waits for a job of the other (`gpool` comes after `pool`
  only so that the generation jobs run in turn). If a job of one layer fails three times, the other layer's edit
  (`FE` or `FG`) is still written, and it is the edit to deliver (steps B and D).
- `warm` exists because four engines of one job that start on an empty kernel cache compile side by side and some
  of them fail at start (the job then fails only at its end and is started again: 16 minutes lost in one of our
  runs). One engine alone fills a cache, and the pipeline copies it into the cache of every later job before that
  job starts. In run A no job was lost.
- A generation job can hang when its engines shut down (every engine has printed its shutdown line, the GPUs show
  memory in use at 0% utilisation, the job does not end). The pipeline sees that after three minutes and stops the
  job's engine processes; the job then writes the answers it has and ends (`pipeline.log` says `hung at engine
  shutdown: ... the job goes on`).
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own. The uploads of steps B
  and E use the CPU only and can be made while it runs. `F`'s model is built by a job of its own (`build`), for the
  screens; the upload of `F` does not wait for it.
- To change an option (section 6), stop the pipeline (`--stop`), wait for `pipeline.log` to say it has ended, and
  start it again with the option added to the same command. Give that option with every later call, `--status`
  included.

**Step A: the repair layer's answers and rows (at about 1:07)**

- `logs/wreq.log`: `reqgen.py` prints its counts (2,400 requests, 23 instruction types), `wreq.py` prints `written
  requests: 1200 to answer (600 with two instructions, 600 with three) ...; screen 300 (100 / 100 / 100 ...)`.
- `logs/pool.log` ends with the rollouts' summary per source (`acc`, `cut`, `mixed`). Ours (run A: the champion with
  U, two answers each at this limit): `ifsynth` 90.9% followed, 1.9% cut; `tuluif` 90.1%, 0.4% cut. The champion's
  model follows about as often (90.4% of 300 practice requests, section 2.1).
- `logs/wgen.log` has the answers' table first (`ALL ... followed ..%`, then `kind w2`, `kind w3` and a line per
  instruction type; ours for the champion's model: 85.2% of the answers to requests with two instructions follow all
  of them, 71.2% with three), then three blocks. From `repair_rows.py`: `finished_failing`, `repaired`,
  `requests_with_a_repair`, then per instruction type the failing and the repaired answers, the repairs used and the
  reasons for no repair; ours for the champion's model, the two sources counted apart (section 2.2): 1,208 and 939
  finished failing, 951 and 478 mended (in run A, the champion with U at four answers per practice request:
  `"answers": 11288, "finished_failing": 1430, "repaired": 872`). From `mix_rows.py`: `"repair_pairs": 700,
  "plain_sentence_only": 90, "from_written_requests": 478, "anchors": 250` (ours, section 2.3). From
  `passfail_rows.py`: `450 own pass-over-fail pairs (written requests 270)`. Last line: `repair layer: 950 repair
  pairs and anchors + 450 own pairs = 1400 rows`.
- Read three pairs of `w/repair_pairs.jsonl` (rows with `"kind": "repair"`): `prefix` is a whole thinking, `rejected`
  the answer as sampled, `chosen` the same answer with the one thing the instruction asks for.
- **The repair layer applies while the repairs are there.** Fewer than 200 repair pairs: the job `wgen` fails by
  design and the repair layer is left out; the word-problem layer goes on, and `FG` is the edit (step D). Fewer
  than 450 repair pairs is outside what we measured: go on, and say it in the report.

**Step B: the repair layer's fits, and the first upload: `FE` (at about 1:43)**

- `logs/ESEL.log` prints `examples: train 1355 (pairs 1117), val 45` (ours, section 2.3; no row longer than the
  limit), `650 neurons over 22 layers (margin, ..s): L0:4, ... L10:103, L11:102 ...`, never the last layer and none
  of `w/X.json`, and writes `ESEL.json` without training (`payload 3,994,900 bytes`, the float16 form). `logs/E0.log`
  and `logs/E14.log` print the same neurons with `(margin, 0s)` (they train them), each step's `max change` (the
  cap is 0.5 per row), `val loss: start .. -> end ..` and `wrote ... 66 ops, 66 tensors, payload 3,994,900 bytes`.
  `logs/apool.log` ends with the rollouts' summary of the start's 450 answers (ours, run B: 399 of them right, 12
  cut). `E0` starts on the two GPUs `ESEL` leaves, `E14` on the two `apool` leaves (in our rehearsal three minutes
  later: there the engine's start was most of `apool`).
- The repair layer's held-out loss **falls** (ours 0.42 -> 0.27 and 0.26): the chosen side of a repair pair is a
  text the model finds unlikely, and the fit makes it likelier. That is expected.
- `logs/FE.log`: `[avg] EAVG.json: 66 ops from 2 fits` (three per layer the neurons are in; 66 in ours), `[dose]
  ... EA.json: 66 ops, k=0.35`, `[encode] ... FE.json: about 1,950 ops (a few rows kept as float16) | payload about
  2,000,000 bytes (was about 4,000,000) | encoded change differs from the wanted one by about 0.8% of its length` and
  `FE: 66 tensors, payload ... | with the units about 2,100,000 of 4194304` (ours for a layer of 640 neurons in run
  A: 1,919 ops, 1,977,140 bytes, 0.77%, with the units 2,075,508). **Check:** the encoded change is within 2% of the
  wanted one.
- **Upload `FE` now.** It is a complete candidate (the repair layer on U), and it stands if anything later fails:

  ```
  python -m wsp.upload rq FE --task task.json --dry-run --report /workspace/work/cases/rq/NOTES.md
  axonry-check $WSP_WORK/deliver/rq__FE/edit --task task.json
  python -c "import json, collections; j = json.load(open('$WSP_WORK/deliver/rq__FE/edit/edit.json')); print(j['parent'], len(j['ops']), dict(collections.Counter(o['op'] for o in j['ops'])))"
  python -m wsp.upload rq FE --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `FE` stands on the case's edit `U`, so the upload carries U's twelve ops, twelve `round` ops and the layer's ops
  as one edit on the champion. **Check:** the third command prints the champion's id as the parent and the op
  counts (about 1,975 ops: U's 12, 12 `round`, the layer's about 1,950; ours in a rehearsal at a twentieth of the
  data, 640 neurons: 1,944 ops); `axonry-check` prints `"passed": true`, a payload below 4,194,304 (about
  2,100,000), every tensor's relative change below 0.5 (the largest are the units': 0.30 on
  `layers.6.mlp.gate_proj`, 0.28 on `layers.23.mlp.gate_proj`; a layer moves a tensor by 0.014 at most) and the
  global relative change below 0.2. The files of
  `$WSP_WORK/deliver/rq__FE/edit/` together must stay below 8,388,608 bytes (ours for one layer of this size:
  about 3.1 MB). Use these commands, with the other name, for every later upload.

**Step C: the word-problem layer's answers and pairs (at about 2:31)**

- `logs/gpool.log` ends with the rollouts' summary (`acc`, `cut`, `mixed`, `mean_tokens`), then the pairs' lines.
  Ours (run B): 78.5% of the answers right, none cut, 582 problems with both outcomes, 5,900 tokens an answer;
  `pairs 911 from 578 of 1600 problems with a right and a wrong finished answer; anchors 200 -> ...` and
  `{"problems": 1600, "problems_with_both": 578, "pairs": 911, "anchors": 200}`.
- Read two pairs of `w/pairs_g.jsonl`: the same problem, `chosen` a whole answer that ends in the right result,
  `rejected` a whole (usually longer) answer that ends in another.
- **The word-problem layer applies while the pairs are there.** Fewer than 200 pairs: the job `gpool` fails by
  design and the layer is left out; `FE` (uploaded in step B) is the edit. Fewer than 600 pairs, or more than 5% of
  the pool's answers cut at the limit, is outside what we measured: go on, and say it in the report.

**Step D: the word-problem layer's fits (at about 3:22)**

- `logs/GSEL.log` prints `examples: train 1053 (pairs 827), val 52` (ours, run B; 21 to 26 rows longer than 20,480
  tokens are left out), `650 neurons over 23 layers (margin, 281s): ... L9:77 ... L14:93, L15:90 ...`, never the
  last layer and none of `w/X.json`, and writes `GSEL.json` without training. `logs/G7.log` and `logs/G35.log`
  print the same neurons with `(margin, 0s)`, each step's `max change`, `val loss: start .. -> end ..` and `wrote
  ... 69 ops, 69 tensors, payload 3,994,900 bytes`.
- This layer's held-out loss **rises** (ours 0.235 -> 0.297 and 0.303); it is not a reason to stop.
- `logs/FG.log`: `[avg] GAVG.json: 69 ops from 2 fits`, `[dose] ... GA.json: 69 ops, k=0.25`, `[encode] ...
  FG.json: about 1,950 ops (a few rows kept as float16) | payload about 2,010,000 bytes | encoded change differs
  from the wanted one by about 0.8% of its length` and `FG: 69 tensors, payload .. bytes | with the units ..` (a
  layer of 650 neurons in this form was 2,006,116 bytes in a run of ours). `FG` is the word-problem layer alone on
  U; it is the edit if the repair layer was left out, and otherwise not uploaded unless the report has a reason to.

**Step E: the joined edit and its upload (at about 3:25: `F` done)**

- `logs/F.log` shows `[union] ... EG.json: 69 ops, N rows / columns from 2 layers`, `[encode] ... F.json: N ops (a
  few rows kept as float16) | payload .. bytes (was ..) | encoded change differs from the wanted one by ..% of its
  length`, then the payload lines of `F`, `FE`, `FG` (`F: 69 tensors, payload .. bytes | with the units N of 4194304
  (M to spare)`) and `neurons: EA.json 650, GA.json 650, in both .., together ..`. Ours, for unions of the same
  kind: a repair layer with the answer round of run A, 656 + 640 neurons of which 41 in both: `payload 3,872,944
  bytes ... 0.77%`, with the units 3,971,312; the repair layer of section 2.3 with a word-problem layer fitted on the
  champion's model, 650 + 650 neurons of which 122 in both: 3,732,788 bytes with the units; the same repair layer
  with the word-problem layer of section 3.1: 650 + 650 neurons of which 78 in both (about 3,870,000 bytes with the
  units). With no neuron in both, 650 + 650 neurons come to about 4,110,000 bytes with the units, still 40,000
  under the limit.
- **Check:** `F` with the units leaves at least 40,000 bytes under 4,194,304 (the job fails by itself otherwise,
  and `FE` stays the edit), and the encoded change is within 2% of the wanted one.
- Upload `F` with the four commands of step B (`F` in the place of `FE`). **Check** as there: the champion's id
  as the parent; `axonry-check` prints `"passed": true`; payload, relative changes and file sizes inside the
  limits. Ours for the union of run A: 3,787 ops = 3,759 `dir_ablate`, 8 `set_rows`, 8 `set_cols`, 12 `round`; the
  check passed without a violation, global relative change 0.042, the largest per-tensor values the units' (0.30
  and 0.28), the two layers 0.014 at most; `edit.json` and `edit.npz` together 5,886,134 bytes. The files of the
  edit together must stay below 8,388,608 bytes: two layers have about twice the arrays of one.
- If `axonry-check` refuses `F` for a limit, do not change `F`: `FE` stands.

**Step F: the screens (at about 4:35)**

`logs/guard.log` holds five paired comparisons of `F` with the champion's model (`base`), on the same problems and
seeds, four answers each, none of them a problem a fit used:

- `ifscr`: 150 + 150 practice instruction requests of the dev and holdout splits;
- `wscr`: the 300 dev requests of `derive/reqgen.py` (one, two and three instructions; `w/wscr.json`);
- `gsmscr`: the practice word problems of the dev and holdout splits (213);
- `puzscr`: the practice puzzles of the dev and holdout splits (213; left out when the toolkit has none);
- `mathscr`: 300 practice math problems.

Read in each the `all:` line (gain and its standard error; for `wscr` the line `F ... ALL`), the `cut` line and the
lines per source. Nothing is tuned here. **F stays** unless one of these holds:

- `ifscr` or `wscr` is down by more than two standard errors;
- `puzscr` is down by more than two standard errors;
- `gsmscr` or `mathscr` is down by more than two standard errors, or its `cut` share rises by more than five
  points.

If one holds, a single layer is tried on the set that showed it, and only there: `FE` when `gsmscr` or `mathscr`
showed it (the word-problem layer is the one that lengthens the thinking), `FG` when `ifscr` or `wscr` did, `FE`
and then `FG` for `puzscr`. The commands, when the pipeline has ended: `python -m wsp.case run rq FE <set> --n 4`
(the model is built by that command, about seven minutes; for `wscr`: `python derive/genreq.py --case rq --model FE
--items rd/w/wscr.json --out rd/w/wscr.FE.jsonl --n 4 --salt ws1`, then `python derive/cmp.py rd/w/wscr.base.jsonl
--edit FE=rd/w/wscr.FE.jsonl`), then `python -m wsp.case compare rq base FE <set>`. If the single layer is not down
by more than two standard errors there, it is the edit: upload it (again, if it is `FE`), so that it is the last
upload. If it is down as well, upload `U`, the units alone. The report says which rule decided. A set of 213 to
300 problems has a standard error of 1 to 2 points: these screens can show harm; they cannot show a gain of the
size either layer has.

**Step G: report and deliver**

- REPORT.md: the champion's id and the parent's; the last line of `units_check.py` and `take_units.py`'s lines; the
  lines of steps A and C; per fit the examples, the pairs kept, the neurons per layer, the held-out loss at start
  and end; the `[avg]`, `[dose]`, `[union]` and `[encode]` lines and the payload lines with the neurons the two
  layers share; each uploaded edit's parent, op counts, payload, file sizes and relative changes; the five
  comparisons of step F with standard errors and cut shares, and the rule that decided; everything that failed or
  was skipped and which fallback was used.
- Upload the final report with the edit. Make sure the last complete upload is the edit the readings support, and
  run `axonry-done`.

Time left over is best spent on more samples of the two models on the screens (`python -m wsp.case run rq base
ifscr --n 8`, then the same for `F`: only the missing samples are added), not on anything new.

## 6. Time plan (6 hours, 4 GPUs) and what to do when it slips

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:25 | setup: copy, the case, the units check and the units, the two models built | none |
| 0:25-1:07 | the neurons to stay off; `warm`; `pool` (the champion's model, eight answers per request); `wgen` with the repair layer's rows | 4 |
| 1:07-1:42 | the repair layer: its neurons (beside them `apool`), its two fits side by side | 4 |
| 1:43-2:00 | `FE`; **upload FE** (CPU) | none |
| 1:43-2:31 | `gpool`: the word problems, eight answers each, and the pairs | 4 |
| 2:31-3:22 | the word-problem layer: its neurons, its two fits side by side | 4 |
| 3:22-3:45 | `FG`, `F`; **upload F** (CPU); `build` beside it | none |
| 3:22-4:35 | the champion's model, then `F`, on the five screens; step F | 4 |
| 4:35-4:55 | step G, final upload, report | none |

About five hours with the screens; `FE` is uploaded at about 2:00 and `F` at about 3:45. One generation job lost and
repeated adds 14 to 47 minutes (a lost `gpool` is the most: `F` is then uploaded at about 4:30, still an hour before
the end). What the settings of the repair layer cost, against the clock above: `POOL=4` makes `pool` about 10
minutes shorter; `E_START="U"` has no `apool` and changes no other time; `E_SELECT="fit"` runs the two fits one
after the other (about 30 minutes more); `MAX_LEN=12288` leaves out the few rows longer than that (3 of 1,400 in run
A, where a fit took 26 minutes; 30 with the rows of section 2.3 at 20,480). Rules for a run that is late, by the
clock of the run. The word-problem layer is the long one (about 100 minutes) and comes last:

- **`gpool` has not started at 2:50:** stop the pipeline and start it again with `--no-gsm`: `FE` (uploaded in
  step B) is the edit; it is built and read on the screens in the place of `F`.
- **`F` is not done at 4:40:** `FE` is the edit. Stop the pipeline and start it again with `--no-gsm` to read `FE`
  on the screens.
- **The repair layer's jobs have failed and `FG` is done:** upload `FG`; start the pipeline again with
  `--no-repair` to read it on the screens.
- **Neither `FE` nor `FG` is done at 4:45:** upload the units alone (`U`), say that nothing fitted is delivered.
- **A fit fails for lack of memory:** start again with `--max-len 12288` (the repair layer's fits; a few long rows
  lose their place) or `--max-len-g 12288` (the word-problem layer's; more of its long rows lose their place in the
  fit; the same value for both of its fits).
- **The screens are not done at 5:20:** the last upload stands; say what was and was not read.
- No new job after 5:20. The last upload must be complete and checked before 5:35.

## 7. What to expect

The repair layer was read in its delivered form on the champion's model, without the units; the word-problem layer
was read in sibling fits on the champion's model, without the units. **The two together on U, and the word-problem
layer fitted on the champion with U, were not read by us on practice data.** The screens of step F are the first
reading of `F`.

- **Plain instruction requests** (`ifscr`, standard error about 1.0): the repair layer +2.1 ± 1.0 on the champion's
  model; U +0.3 ± 0.9; a repair layer made on the champion with U, joined with another layer, +1.0 ± 1.1 (section
  2.5). Between -1 and +4 for `F` against the champion's model.
- **Written requests with one to three instructions** (`wscr`, standard error about 1.1): the repair layer +1.7 ±
  1.0 on the champion's model. Between 0 and +4.
- **Word problems** (`gsmscr`, standard error about 1.3 on 213 problems): the word-problem layer +0.2 to +3.4 over
  nine fits on the champion's model. Between -1 and +4; longer answers (about 1,700 tokens more on that model).
- **Puzzles.** No change: the repair layer +0.3 ± 0.6; the word-problem layer was not read on puzzles.
- **Math** (`mathscr`, standard error about 1.2): U by itself +1.6 ± 1.1; the repair layer +0.9 ± 1.4; the
  word-problem layer was not read on practice math problems. It makes the model think longer: watch the `cut`
  share (the limit of step F is five points).
- **One run is one draw.** The pairs of another run are other pairs.
- **Limits.** Payload between about 3,700,000 and 4,110,000 bytes of the 4,194,304 allowed, depending on how many
  neurons the two fits share; relative changes far below the limits (step E).

## 8. What we know does not work, and what we do not know (ours, practice data)

| what | finding |
|---|---|
| The two layers made in one run, and their union on U | **not read by us on practice data** before this entry was written (sections 2.5, 3.4); the whole job list ran end to end only in rehearsals at a twentieth of the data |
| The repair layer made on the champion with U, from that model's answers (`E_START="U"`) | its fit moves the chosen answers about half as far (held-out loss 0.32 -> 0.25 and 0.26, against 0.42 -> 0.27 and 0.26 for the rows made from the champion's own answers); not read alone on practice data; not delivered (sections 2.2, 2.5) |
| The repair layer at doses 0.7 and 1.0 | +1.8 ± 1.0 and +0.5 ± 1.4 against +2.1 ± 1.0 at 0.35 (section 2.4) |
| Repair pairs without own pass-over-fail pairs | +1.8 ± 1.0 at 0.35, -2.9 ± 1.5 at 0.7, -9.0 ± 1.9 at 1.0: the added closing text becomes a habit and breaks the end phrase and the paragraph and bullet counts (section 2.4) |
| Mended answers as plain likelihood rows with a keep term; pairs at the draft; counter pairs | section 2.5: +0.1 to +1.2; no gain over the delivered fit and a collapse at 1.0; no gain at 0.35 and worse at 0.7 |
| Repairs the program cannot make | numbered sections (8 of 135 failing answers mended), a required first word of a paragraph (26 of 176), a second response (4 of 23): the layer has almost no rows for them |
| The word-problem layer at dose 0.5 | -0.3 to -5.8 points on fresh practice word problems, 11 to 20% of the answers cut at the limit (section 3.3) |
| The word-problem layer without the units | at 0.25 already 3 to 5% of the answers to word problems are cut at the limit; it is fitted on and delivered with U |
| A smaller pool of word problems | not tried: every fit of the word-problem layer used 1,600 problems with eight answers each |
| The fits left free to take the champion's neurons | not delivered: every fit keeps off the 1,660 neurons the champion's edit and U name, which this entry is meant to leave as they are |
| Another run | each layer was fitted once in its delivered form: the repair layer on the champion's model, the word-problem layer on the champion with U (one draw of answers, one pair of fits each) |
