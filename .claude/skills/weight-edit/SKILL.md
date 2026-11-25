---
name: weight-edit
description: Improve a small LLM on benchmarks by directly editing its weights, following the Weight-Space Programming workflow - diagnose the failure, build contrast pairs, find the direction and the components writing it, edit, test selectivity and dose with controls, freeze, confirm on holdout, then deliver the edit. Use when asked to modify/repair model weights, run a weight-editing case or competition task, or continue a case under cases/.
---

# Weight editing workflow

You are the editor: you decide what to change from measurements of what parts of the network compute. Read
`playbook.md` first (what worked, what did not) and, in a competition task, the task's brief and the submission's
`METHOD.md` (its commands may predate the current toolkit: check them with `-h`). Tools: `python -m wsp.case
<command>` (`-h` for options), run wherever the GPU is. Keep decisions in
`cases/<case>/NOTES.md`; numbers come from `report` (`RESULTS.md`), never copied by hand.

## Ground rules
- No overfitting (`rules.json` -> `edit.no_overfitting`): never use the evaluation benchmarks' items or answers to
  derive or choose an edit; the gain must carry over to unseen problems.
- The deliverable is an edit spec within `rules.json` -> `edit` (editable tensors, ops, the payload and
  magnitude limits; `edit` prints the spec's tensors and payload, the build manifest its relative changes). The
  model is always rebuilt from the pinned base, so only the spec counts.
- Choose on practice problems only: `dev*` sets for selection, `holdout*` only after `freeze`, once: a candidate
  tested, repaired or re-sampled on holdout makes it a selection set (results record whether the edit was frozen).
  A spent holdout or a method whose candidates all failed does not end the work while time remains: keep improving
  on DEV and on fresh sets (`split`). In a competition task
  everything about the evaluation is closed to you (which sets, their items, answers and results): never load, read
  or look it up, and aim at no particular set. The toolkit holds no benchmark: `bench:<name>` only reads items an
  organizer supplied, and outside a competition such sets only confirm a frozen edit.
- Generation uses the profile's model-card sampling; a gain that needs another decoding setting is not a gain.
- Report everything, including losses and failed attempts.

## Time budget (competition tasks have a hard deadline)
- Plan before starting: the deadline is in the brief. Generation dominates the cost: every `run` starts vLLM (a few
  minutes), and its length grows with the number of problems and how often the model runs to the token limit;
  probes (`extract`, `selectivity`) take minutes. Time one run, then count the runs you can afford and drop what does
  not fit.
- Keep every GPU busy: a `run` splits its problems over the visible GPUs that are free; to run independent jobs side
  by side, give each its own `CUDA_VISIBLE_DEVICES`, or queue them with `tools/queue.sh` (one worker per GPU).
  Stopping a run stops its vLLM engines with it; a run refuses a GPU another job holds and says which.
- Use the cheapest evidence first: `selectivity` screens edits in minutes (a pre-screen: it does not always predict
  generation); generate only for the most promising ones, on `dev128`.
- **Upload early, then improve**: as soon as one edit does not lose on DEV, upload it (`python -m wsp.upload`, `-h`
  for the package layout; `--status` reads back what is uploaded). Re-upload when a better edit is confirmed; the
  last complete upload counts, so an early one is never wasted and a late one may never land.
- Keep the last stretch (at least 30 min) for the final upload; skip HOLDOUT if it does not fit and say so in the
  report.

## Steps
1. **Screen**: in a competition the base (and parent, if any) is given: run it on the screening problems
   (`init`, then `run <case> base dev`, and read `RESULTS.md`). Otherwise compare candidate models with
   `screen <profiles...>` and `report`. A removable failure must dominate and `acc when finished` be far above
   `acc`; read failing answers yourself. Without that, editing is unlikely to help: say so instead of forcing an
   edit.
2. **Init**: `init <case> --profile P --practice gsm8k_train math_train mmlupro ifsynth tuluif bbhsynth --subset
   dev128=dev:128 --subset holdout256=holdout:256`, then `run <case> base dev`. Cover every practice task type
   (`wsp/data.py`), since an edit that helps one type can hurt another (without `--practice` the splits are math
   only); a subset takes equally many problems of each source. `mmlucf` (exam questions) and `musr` (questions after
   a long story) are more multiple choice, for fit sets and wider checks (`split --practice`). To build on a published
   edit, add `--parent <its edit.json>`: "base" is then that model, and your edits stack on it. A task holds each
   start it offers in `<sid>/edit/` (`task.json` -> `starts` says which is the entry's parent and which the current
   champion); screen them and start from the one that suits the method (`--parent <sid>/edit/edit.json`: the
   upload names it by its sid). An entry is judged
   against the current champion: when starting elsewhere, `adopt` the champion into the case and compare with it on
   the same sets. First read the `REPORT.md` next to that edit.json: what it tried and what did not work.
3. **Pairs**: `pairs <case> --sets dev`, and held-out pairs with `--exclude-pairs pairs --out pairs_heldout`; build
   them from a base run with enough finished traces, since held-out pairs need traces of their own. The contrast
   kind is `case.json` -> `contrast` (`wsp/directions.py` `CONTRASTS`; add a generator for a new failure).
4. **Extract**: `extract <case>` -> per-layer stability, separation, writer totals. Take layers where the direction
   is stable and MLPs write much of it (if they write little, attention does: `edit --targets o_proj`); do not
   ablate from the last layer. `extract --out` keeps several direction files apart.
5. **Edit**: `edit <case> --layers 15 19 --topk 64 --alpha ...` and the same with `--random-control`; then
   `selectivity <case> <edit>` (failure log-prob down, normal log-prob ~unchanged) and `run <case> <edit> dev128` for
   the promising ones, with a dose series. Every `run` of an edit is paired with the base automatically; choose
   between edits with `compare <case> <a> <b> <sets...>` on the same problems (pooled; per source too), and trust a
   gap only when it is about twice its SE; when it is not, add samples (`run --n` adds only the missing ones) or
   DEV problems rather than choosing on HOLDOUT. Look at every source, not only the pooled gain.
   When the failure has no stable direction but the model often solves a problem in some samples and not in
   others (`pass_at_k` well above `acc` in a run with `--n 4`), fit on its own samples: `split <case> fit --n 1024`,
   `run <case> base fit --n 4`, then `python -m wsp.fit run` (`-h`: passing answers, advantages or passing vs
   failing pairs; per-unit gains, singular-value scales or rewritten MLP units; `branch` resamples from a start:
   after the whole thinking, or `--at` points inside it, where traces are decided; `@prime=` opens the thinking
   with a way of working that a fit can distil). Fit several seeds and average them (`wsp.fit spec`), and run the
   shuffled control.
6. **Freeze** the best edit (`freeze <case> <edit>`), then `run <case> base holdout256` and `run <case> <edit>
   holdout256`.
7. **Deliver**: `report`, update NOTES (and `playbook.md` outside competition tasks), then the final
   `python -m wsp.upload <case> <edit> [--task task.json]` (see "Time budget").

## Spec format
`edit` writes `dir_ablate` specs; any other allowed op is written by hand (or by `wsp.gate.ops` for one MLP
unit) into `cases/<case>/edits/<name>.json`. Lists may instead be `"file.npz#key"` relative to the spec;
`tensor` is a full name from the base's safetensors index (`wsp/edits.py` has the exact semantics):
```
{"name": "E1", "base": "<profile>", "parent": "../<case.json parent>" (when stacking), "ops": [
 {"op": "dir_ablate", "tensor": "...layers.15.mlp.down_proj.weight", "side": "out", "u": "E1.npz#u15",
  "alpha": 6.0, "cols": [3, 17]},                                   # cols optional: all columns
 {"op": "scale_rows", "tensor": "...layers.4.mlp.up_proj.weight", "rows": [11], "factor": 0.5},
 {"op": "scale_cols", "tensor": "...layers.4.mlp.down_proj.weight", "cols": [11], "factor": 0.5},
 {"op": "set_rows", "tensor": "...layers.4.mlp.gate_proj.weight", "rows": [11], "values": "E1.npz#gate"},
 {"op": "set_cols", "tensor": "...layers.4.mlp.down_proj.weight", "cols": [11], "values": "E1.npz#down"}]}
```
`values` is len(rows) x in for `set_rows`, out x len(cols) for `set_cols`.
