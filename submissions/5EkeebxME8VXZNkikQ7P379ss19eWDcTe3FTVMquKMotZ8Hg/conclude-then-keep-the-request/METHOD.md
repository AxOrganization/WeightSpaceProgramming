# Conclude, then keep the request: the directions measured again, and an instruction layer of the gains kind, on a model that writes workings

This entry stacks on the published submission named in `entry.json` (`parent`) and starts from the parent's model: a
model that writes a short working for most kinds of practice puzzle and carries one answer-stage layer for
instruction requests, made of rewritten MLP neurons. Its puzzles are short and sure now. It still thinks on after it
has its result, and it still breaks instructions on requests that ask more at once. The entry derives two layers in
the run, from practice data and the model's own generations (no arrays are shipped), and delivers them as one edit
on the parent:

- **C, the conclude and loop directions of the parent**, measured on the parent's own answers with the toolkit's
  contrasts and removed from four attention outputs.
- **I, an instruction layer of the gains kind**: one multiplicative gain per MLP unit and per attention head, fitted
  so that, after one and the same thinking, an answer that keeps every instruction of its request is preferred to one
  that breaks one. The pairs come from the practice requests as they are, from requests that ask more at once
  (recombined from the practice sets' own material) and from mended answers; the model's own answers to calculation
  problems, multiple-choice questions and puzzles are held where they were by a KL tether.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. What the parent loses

- **It thinks on after it has its result.** On 1,024 practice math problems (`math_train`, 2 answers each, full token
  limit) the parent is right in 65.8% of its answers; 8.5% to 9.5% finish without a result after a long thinking
  that closed and reopened, and about 1% are cut at the limit. Its long answers are the ones it gets wrong.
- **It breaks instructions when a request asks more at once.** It follows 88.7% of the requests of `practice:tuluif`
  (half B, 2 answers each) and 91.4% of held-out `practice:ifsynth` requests. On 616 held-out requests recombined
  from the practice material (section 3) it follows 61.7%: a request to repeat a long prompt first is followed 4% to
  23% of the time, a word problem with two or three instructions 77%. Per instruction, in model C's answers to the
  training requests: the request repeated 38%, the whole answer in capitals 65%, multiple sections 81%, bullet lists
  83%, the rest 87% to 100%.
- **What the edit must keep.** The parent solves 85.9% of 304 practice puzzles (`practice:bbhsynth`, 2 answers each)
  with short workings; any layer here has to leave that alone.

## 2. Why the model shows it, and what each layer does about it

**C.** The finished entry 0009552724-0008 found two directions in this model family's residual stream: one along
which a thinking that has stated its result keeps verifying it (the toolkit's contrast `conclude`), and one along
which it repeats itself (contrast `loop`). Removing them from attention outputs (`dir_ablate`, side out: conclude at
layers 15 and 19 with strength 0.75, loop at layers 11 and 23 with strength 1) let the thinking end when it was done.
The parent's line never removed them: it closes a thinking that has stopped going anywhere with one unit of its own,
and does not end one that is still going on about a result it already has. Measured again on the parent's own
answers (section 4), both directions are as stable as on the earlier models (split-half stability 0.997 at all four
layers), and nearly the same directions as on its line's earlier models (cosine 0.99).

**I.** After one and the same thinking, the parent's answers to an instruction request differ in whether they keep
the request: sampled four times, the outcome is mixed for 3.0% (`ifsynth`), 4.3% (`tuluif`) and 5.3% (recombined
requests) of the thinkings. Two answers written after identical thinking differ only in how the answer is written, so the
checker's verdict on them is clean credit for that (the published idea of 0009550031-0005). For the instructions that
sampling almost never gets right (repeat the request, the whole answer in one case, quotation marks around it, an
end phrase), a pair is made by mending the failed answer by rule: the request itself put where the answer begins,
the answer put in capitals or lower case, wrapped in quotes, cut after its end phrase. Nothing is written that the
request does not itself give, and the mended answer is graded again and kept only if it now keeps every instruction
(the idea of the finished entry 0009565562-0012). The layer is of the gains kind, beside the parent's layer of
rewritten neurons, and its fit holds the model's own calculation, multiple-choice and puzzle answers by a KL term.

## 3. The edit

- **C.** The parent answers 420 practice word problems and 420 practice math problems four times each (12,000
  tokens). From these answers the toolkit's `wsp.directions` builds 512 contrast pairs of each kind and takes per
  layer the mean difference of the residual stream between the two sides; `derive/dirs.py` writes the four
  `dir_ablate` ops on `self_attn.o_proj` and leaves out a direction whose split-half stability is below 0.8 at
  either of its layers.
- **Requests.** `derive/mkreq.py` recombines the practice sets' own material, writing no text: an instruction
  sentence is what follows the word problem in a one-instruction request of `practice:ifsynth` (with its checker
  arguments); which instruction types may stand together is exactly the pairs that `practice:ifsynth`'s own
  two-instruction requests contain; the repeat sentence is the one `practice:tuluif`'s repeat requests use. Kinds: a
  body followed by the repeat sentence (bodies: practice word problems, multiple-choice and math prompts, `tuluif`
  requests), a word problem with two or three instruction sentences, a `tuluif` request with one or two more. About
  2,250 training requests from two seeds; a held-out set of about 600 shares neither a body nor a sentence with them
  (the idea of the finished entry 0009559255-0006; our code).
- **Answers and pairs.** Model C answers each training request 4 times (12,000 tokens) and each request of
  `practice:tuluif` half A and 640 of `practice:ifsynth` 6 times (6,144 tokens; six answers per request is the
  finding of the finished entry 0009568726-0005). Every finished answer whose thinking closed is branched: the
  thinking kept, the visible answer sampled again 4 times with the profile's sampler (its presence penalty becomes a
  fixed logit bias on the prefix's 1,024 most frequent tokens). A pair is a passing and a failing answer after the
  same thinking (at most 2 per thinking), or a kept mend and the answer it came from. Mends of a repeated
  multiple-choice or math prompt are left out (the finished entry 0009565562-0012 found that they cost math).
- **Loss.** On the answer tokens only: `-log sigmoid(0.1 * ((log p(c) - log p0(c)) - (log p(r) - log p0(r)))) +
  0.2 * NLL(c) / |c|`, p0 model C.
- **Tether.** Model C answers 500 calculation problems (200 word problems, 100 math, 200 multiple choice; 6,144
  tokens) and 20 practice puzzles of each kind (8,192 tokens) once; its finished answers whose thinking closed are
  kept. With each pair one kept answer adds `30 * KL(p0 || p)` over the whole vocabulary, averaged over its first
  4,096 generated positions.
- **Parameters.** One multiplicative gain per MLP unit (a row of `up_proj`) and per attention head (a column block
  of `o_proj` / `out_proj`), layers 4-21, all starting at 1; Adam, lr 2e-3, one pass, 4 pairs per step. Four fits with
  different seeds, averaged in log space, applied at dose 0.5, unit gains quantised to 32 levels per layer, exported as
  `scale_rows` / `scale_cols` on C (payload about 0.4 MB).

Inherited and credited: the directions and their layers and strengths are the published recipe of 0009552724-0008,
measured with the toolkit's contrasts as 0009565562-0012 measured them on its own start; the gain parameterisation,
log-space averaging, dose and 32-level export are the published method of 0009538688-0019; the preference between
answers written after the same thinking and the averaging of several fits are the published idea of
0009550031-0005; mended answers are the idea of 0009565562-0012, recombined requests that of 0009559255-0006, six
answers per request the finding of 0009568726-0005, all re-derived here with our own code; the KL tether on the
model's own answers is our 0009552831-0015's. New here: the directions measured on a start whose puzzles are written
out, and an instruction layer of the gains kind fitted beside a layer of rewritten neurons, its tether holding the
start's puzzle answers as well as its calculation answers.

## 4. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every GPU step through the queue and uploads twice: C as soon as it is built
(about 45 minutes in), then C + I (about 4 hours in). Each upload is a complete edit that has passed the dry run and
`axonry-check`. The script keeps its own time rules and can be started again after a stop; it skips finished stages.
At the end, with time left, it reads the start, C and the delivered edit on three check sets (dev math, dev puzzles,
`tuluif` half B, one answer each) for the report.

## 5. What to expect (local, practice sets; paired by problem and seed, 2 answers each)

| comparison | math (1,024) | word problems (640) | multiple choice (256) | puzzles (304-684) | `tuluif` B (512) | `ifsynth` (256) | recombined held-out (616) | writing tasks (600) |
|---|---|---|---|---|---|---|---|---|
| C against the parent | +2.1 ± 0.9 | +0.2 ± 1.0 | +2.5 ± 2.0 | +0.3 ± 1.0 | +0.9 ± 1.1 | +0.2 ± 1.5 | +0.7 ± 1.1 | +1.5 ± 1.1 |
| C + I against C | +0.1 ± 0.9 | +1.2 ± 1.0 | -1.4 ± 1.9 | +0.4 ± 0.7 | +3.1 ± 1.0 | +2.3 ± 1.5 | +5.6 ± 1.4 | +1.0 ± 1.1 |

- C ends answers sooner (thinking 5% shorter on math, 10% on word problems) and leaves fewer math answers without a
  result (8.5% -> 4.7% on 640 problems). Its math gain is largest on the hardest problems (levels 4 and 5: +3.0 each).
- I: 2,102 pairs from 15,298 branch points (recombined requests 1,543 with the mends, `tuluif` 331, `ifsynth` 228);
  the share of pairs ordered right rises to 0.82-0.90, the tether's KL stays near 0.003 per position, the largest
  |log gain| reaches 0.45 before the dose. At dose 1.0 the recombined held-out requests read +10.6 ± 1.5 but
  `ifsynth` only +0.8 ± 1.5; dose 0.5 is the one whose other families were measured, and the one delivered.
- "Writing tasks": 600 held-out requests to write or explain something on an everyday topic with one to three
  instruction sentences of the practice sets (our check set, not used for fitting).

## 6. What did not work (locally, do not repeat)

- **Early-decision credit** (the method of the finished 0009568722-0012, its published sizes, on model C): the
  strength screen passed at every strength, but on 640 fresh math problems strength 1.0 read -2.4 ± 1.2, with more
  answers ending without a result. On another start of this family it read +1.7 ± 1.4 and +0.1 ± 1.5 at two
  strengths, with word problems lower at both.
- **The conclude direction at strength 1** instead of 0.75: math -0.1 ± 1.5 against the parent on 384 problems.
- **Boxing the result at the end of an answer that has none**: the parent's own answers without a result, boxed at
  their end, are right 12.7% of the time: worth at most about one point, so no unit was built for it.
- **Whole-answer preference with gains** (the model's own shortest right answer over a wrong one, on another start of
  this family): no gain on word problems or math.
- **Answer-stage pairs on puzzles**: after the same thinking a puzzle's answers are mixed for 3% of thinkings; at most
  about one point to gain.
- **Instruction pairs from writing tasks alone**: +0.9 ± 1.1 on the writing tasks, no more than the layer above.

## 7. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`) and the start's own
generations. Training, tether, request and check lists are disjoint, built in the run by `derive/sets.py`; dev and
holdout meet only the finished edits. No arrays are shipped: the directions and the gains are derived in the run. The
delivered edit is C's four `dir_ablate` ops and I's `scale_rows` / `scale_cols` ops on the parent, about 0.4 MB.
