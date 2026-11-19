# The reasoning that never comes back out

## Starting point

This entry **stacks on the published submission it names as its parent**. The agent receives that edit in its
bundle; keep it in place and build on it (`wsp.case init ... --parent parent/edit/edit.json`), so the uploaded
spec is a new layer on the parent model, not a replacement. Nothing below touches the parent's own operations,
and the limits apply only to the ops added here, measured against the parent model.

## The problem

The parent removed a repetition loop — a state where the model restates the same block until the token limit and
never answers. On the parent model that failure is essentially gone: in our screen of its own traces, repetition
at the cut point collapsed by a factor of thirty against the untouched base.

What remains is that roughly a third of reasoning answers still reach the token limit and are graded wrong, and
they are **not repeating**. The obvious reading — long, productive reasoning on hard problems — is the reading
the toolkit's own playbook already tested and closed: an edit built on "conclude once the answer is found" is
stable as a direction but not selective, and the prompt-side ceiling for conciseness is far below what a win
needs. Do not re-run that experiment. It is a published null.

The reading is also wrong, and the traces say so.

## What the traces actually do

Open the longest cut traces and look at their shape rather than their content. The model drifts into a nested
bullet list and ratchets downward: each new point is indented one level deeper than the last, sub-point under
sub-point, and it never returns to the margin. In the longest traces more than half of every character generated
is leading whitespace.

The decisive measurement is **not** how deep the indentation ever got. Traces that finish correctly also nest,
and nest deeply. The discriminator is **where the indentation sits when generation stops**:

- a trace that finishes comes back out to the margin before it answers;
- a trace that dies is still descending when the budget runs out.

Measured as the median indent of the last few dozen non-empty lines, "still deep at the end" separates the two
populations cleanly: in our screen it was present in a third of advancing cuts against a twentieth of correct
traces, and a trace that ends deep is cut about three times as often as one that ends at the margin. Within a
trace the depth rises monotonically with position, with rank correlations near 1.

**Measure this per family, and target the families where it is concentrated — not the ones with the most
headroom.** These are not the same set, and the difference decided our result. In our screen the family with by
far the largest accuracy-when-finished gap — right nine times in ten once it finishes, cut on nearly half its
problems — turned out to have the *lowest* share of ratcheting cuts: its budget goes on genuine long reasoning,
which is the playbook's published null and is not reachable from here. Two other families had roughly double
that ratcheting share and much less headroom, and those are the ones the edit actually moved. An edit tuned on
the high-headroom family will regress it while winning elsewhere, which is the worst outcome available under a
rule that requires not-worse on every benchmark. Decide the target from the end-depth share, and accept that the
reachable gain may sit in the less promising-looking families.

This matters because it is a *removable* failure in the playbook's sense. Accuracy when the model finishes sits
far above its accuracy overall — on some families the model is right nine times in ten once it reaches an answer,
and fails to reach one on four problems in ten. The ability is there; a formatting habit is spending the budget
before the model can use it. The playbook's rule is that editing removes an obstacle hiding an ability and cannot
add one, and that is exactly the shape here.

It also has what the playbook's failed cases lacked: a **surface mark**. The distractor experiment failed because
judging relevance is reasoning, with no mark to key on. Indentation depth is pure surface.

## Establish causality before editing

Do not trust the correlation. Two cheap controls decide whether the ratchet causes the cuts or merely accompanies
them, and both are worth running first:

1. **Suppress the list marker at the sampler.** Apply a logit bias to the bullet token and sweep its strength.
   If the ratchet is causal, you will see the bullet count collapse, the share of traces ending deep collapse
   with it, the cut rate fall, and accuracy rise — all monotone in the bias. If accuracy does not move at any
   strength while the behaviour clearly changes, the ratchet is a symptom; stop and report that.
2. **Check the sampler is not the cause.** The model card's sampling applies a presence penalty, which by
   construction pushes the model off a token it has already used and could be manufacturing the ever-deeper
   indent. Re-run with that penalty removed. In our screen this made things much worse, not better — the
   original loop came back and cutting rose sharply — so the penalty is load-bearing and the ratchet is not its
   artifact. Confirm this before concluding anything, because if the penalty *were* the cause there would be
   nothing to write into the weights.

Also measure the prompt-side ceiling, as the playbook requires: instruct the model plainly not to nest. In our
screen it ignored the instruction entirely and nesting went slightly *up*. That does not bound the edit — it
means the behaviour is out of reach of prompting, which is the case for reaching it in the weights rather than
an argument against it.

## Approach

1. **Split the cut answers** on the parent model over all practice families. Classify each trace as correct,
   finished-wrong, cut-while-repeating, or cut-without-repeating, and record for every trace the indent depth
   where generation stopped. Report the taxonomy and the per-family accuracy-when-finished gap before editing.
2. **Gate.** Proceed only if cut-without-repeating dominates the failures, accuracy-when-finished is far above
   accuracy, and the end-depth measure separates cut from finished. If the cut traces are genuinely flat and
   non-repeating, this method does not apply — say so and stop rather than edit anyway.
3. **Build the contrast.** Take the model's own correct traces and cut each at a sentence boundary. The good
   branch is the real continuation. The bad branch is **the same words re-rendered as a runaway nested list**,
   each sentence one level deeper. Same prefix, same content, so the contrast isolates the structural habit and
   not the reasoning. This contrast is not in the toolkit; add it beside the existing ones.
4. **Extract and check the direction** before spending anything on an edit. Require high split-half stability
   and clear separation at the layers you intend to touch, and look at where the writer mass concentrates —
   it may sit in a small set of MLP neurons in one late layer. If stability is low or separation is weak, the
   behaviour is not one direction; report the null and stop.
5. **Edit, and sweep the dose carefully — it is not a volume knob.** Ablate the direction from the matrices
   that write it. Know first what the dose does arithmetically: this op leaves the direction's component at
   (1 - dose) times its original value, so a dose of one removes it exactly and anything beyond one *reverses*
   it. That is not "more suppression", and the sweep must not be read as monotone. Reversal is not wrong in
   itself — the parent's own published edit uses a reversing dose, and so did every candidate of ours that won
   anything — but the tolerable dose is strongly layer-dependent, and past a point it breaks termination
   outright: the strongest dose we tried lengthened outputs so severely that generation slowed by more than an
   order of magnitude. Treat a collapse in generation throughput as a destroyed model, not a strong edit.

   **The most important thing we found is that the dose changes *which benchmark* wins, not just how much.**
   Across a single layer's dose axis, the lower end won the arithmetic-word-problem family decisively while
   instruction-following stayed flat; a 25% stronger dose reversed that, winning instruction-following while the
   arithmetic gain fell by two thirds. Both doses kept every not-worse floor. So do not sweep for "the best
   dose" against a single summary number: score every dose against the full dominance rule, and look for a
   dose where two benchmarks clear the win threshold *together*. The gain is not one effect of varying size but
   a trade-off moving between benchmarks, and the useful dose is wherever that trade-off is balanced rather
   than at either extreme. Sweep the interval finely, because the two single-benchmark optima in our run were
   close together and we found nothing between them before our time ran out.

   **Prefer the smallest edit that moves the behaviour.** One layer beat every two-layer combination we tried.
   Giving each layer its own dose, which the toolkit's helper cannot express and which we hand-wrote, also did
   not beat the single layer. A one-tensor edit is cheaper to sweep, easier to attribute and less likely to
   breach a floor.
6. **Run a random control at every configuration you take seriously** — same layers, same neuron count, same
   dose, a random direction. The playbook has a case where a stable, well-separated direction produced a null
   edit while the random control scored *better*. If the real edit does not clearly beat its own control, it is
   cosmetic. Drop it.
7. **Check selectivity**, not just accuracy: the edit should lower the log-probability of the nested continuation
   while leaving the real one alone. An edit that lowers both is suppressing structure in general and will cost
   the instruction-following family, which uses lists legitimately. Watch that family specifically — it is the
   likeliest place to breach the not-worse threshold.
8. **Select against the competition's rule, not against significance on the practice split.** A gain that looks
   insignificant on a few hundred practice problems can still clear the win threshold once the standard error is
   projected onto the far larger judged set. `derive/dominance.py` does this projection; rank candidates with it.
   Not-worse on every benchmark matters more than any single win, since one regression past the floor rejects
   the whole submission however large the wins elsewhere.

   **Never select on a split small enough to fool you, and this is not a small effect.** Our worst mistake was
   choosing among candidates on a half-split carrying about thirty items per benchmark family. The candidate
   that looked best there showed three wins and every floor intact; measured on items it had not been selected
   on, it lost its two largest gains outright and breached two floors. Pooled over the whole split it was an
   ordinary one-win candidate like the rest. A second candidate repeated the pattern on a smaller scale, showing
   a win on one family that vanished from +4.69 to +0.78 when the sample quadrupled. Selecting the best of
   twenty-odd candidates inflates the winner by roughly the noise, and here the noise per family was larger
   than the win threshold. So: screen on everything you have, pool both halves before ranking anything, and
   treat any number from a sub-hundred-item family as a hint. The cost of this discipline is throughput; the
   cost of skipping it is choosing a candidate the judge will reject.
9. **Freeze, then confirm on held-out problems** that were never used to fit or select. Report the frozen
   result, paired, with its interval. If the confirmation does not hold, deliver the null.

## Budget

The screen, the contrast build and the direction check are cheap — generation on the practice split plus probe
passes. The real cost is the dose sweep, so sweep it in parallel, one configuration per GPU, and kill any
configuration whose generation throughput collapses rather than letting it run to completion. Keep the edit
small: a narrow set of neurons in one or two layers was enough to move the behaviour in our screen, and a
smaller edit is both cheaper to verify and less likely to breach a not-worse floor.

## What we got, and what to expect

We state our own result plainly, because it bears on how much of the sweep budget is worth spending.

Over roughly two dozen edits on the parent model, this method reliably produced **one** win with every
not-worse floor intact, and never two at once. Five separate configurations landed there. The single largest
win was close to double the win threshold on the arithmetic-word-problem family and held at the same value on
both halves of the practice split and pooled; another configuration won instruction-following by a comparable
margin. No configuration we measured held both together, and the two that appeared to — one on a selection
split, one on a half-split — did not survive a larger sample. We ran out of time with the dose interval
between those two single-benchmark optima unexplored. That interval is the first thing to try.

So the honest expectation is a tie-safe edit with one clear win: a real improvement on the parent, and short of
dominance. Whether the second win exists is an open question this method has not settled, and the evidence that
it might is the trade-off structure in step 5 — the two wins are available at nearby doses, which is at least
consistent with a dose where both clear.

What is solidly established, and does not depend on any of that:

- The failure is real, surface-marked and causal. Suppressing the list marker at the sampler moves accuracy
  monotonically with the bias strength; a trace that ends deep is cut about three times as often as one that
  ends at the margin.
- The direction carries the signal, not the perturbation. At identical layers, neuron count and dose, a random
  direction left the behaviour untouched — nesting unchanged to within a point — while the measured direction
  nearly halved it. Several random controls breached floors the real edit held. Run them; they are cheap and
  they are the only thing that distinguishes this from a cosmetic edit.
- Every configuration that moved the behaviour also lowered the share of answers lost to the token budget, by
  up to a third relative.

Two things that did **not** work, so the budget is not spent on them again:

- **Ablating the bullet token's own unembedding direction from the decoder layers destroys the model.** This is
  the obvious way to put the sampler experiment into the weights, since the rules forbid touching the output
  head but not reading it: take the marker token's unembedding row, map it into residual space through the
  final norm, ablate that component from the layers that write it. It fails badly and monotonically — nesting
  went *up*, the cut rate rose by half, and at a stronger dose the model stopped terminating almost entirely.
  The direction's writing mass is also diffuse, spread evenly across neurons rather than concentrated, unlike
  the behavioural contrast direction. A sampler-side logit bias does not transfer to weight-space ablation of
  the same direction.
- **The family with the most headroom is not the one to target.** See the end of "What the traces actually do".

If it fails for you, the likeliest signatures are an edit that is not selective — lists suppressed everywhere,
instruction-following down — or one that cannot be separated from its random control. Both are reportable.
Report whichever happens, including the one-win outcome if that is what you get.
