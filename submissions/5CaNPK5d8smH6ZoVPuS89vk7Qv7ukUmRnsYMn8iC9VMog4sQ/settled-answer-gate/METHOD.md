# Close the thinking once its answer is settled: a one-neuron gate at the settled state

Built on the published loop-removal entry (entry.json names it as `parent`), this method targets the length that
remains once loops are gone: thinking that already has its answer and does not stop.

## The problem
With loops removed, the model still loses many answers to the 32k token limit, and most of those lost answers do
not repeat themselves word for word. In many of them the right result is already in the thinking: stated, checked,
stated again, and checked again until the limit, so the answer is lost although it was found. Answers that do finish
also spend a large part of their length after the result was settled. Measure this first - on practice problems
only (GSM8K train, MATH train and MMLU-Pro practice, with the IFEval-style `ifsynth` as a guard; the benchmarks are
never used), the parent model being "base", two or more samples per problem - with `derive/settled.py label`:
- per source, the share of answers cut at the limit, and among them the share whose thinking already states the
  gold result of the practice item at least twice (kind `pos` in cut traces): settled, but never closed;
- in finished correct answers, how much text comes after the result was stated for the second time;
- how often a result given as the answer twice or more is later abandoned for the right one (kind `hard`): there,
  checking again paid off, and these positions bound how eagerly the edit may close.
Look at about ten `pos` and ten `hard` traces by hand first; the result extraction (numbers, fractions,
`\boxed{}`, option letters) has to agree with the trace's own answers. Should settled-but-cut answers turn out rare
(under a tenth of the cut answers on every source), the method does not apply, and the report says so.

## Why we think it happens
The model has what it needs to stop - the same result written several times - but closing the thinking is one token
decision that a small model, deep in a long context, keeps postponing: "verify again" is always a plausible next
sentence. The behavior to change is conditional: close when the result is settled, and only then. Removing a
"keep verifying" direction everywhere is not conditional: in a related model the toolkit's `conclude` direction,
ablated from its writers, also lowered the closing continuation, because normal reasoning reads the same direction.
A conditional write does not have that problem: one MLP unit that stays silent at every position except in the
settled state, and there adds the model's own "about to close" signal. The toolkit builds such a unit in closed form
(`wsp/gate.py`; ops `set_rows` / `set_cols` on one unit's gate, up and down weights). A settled spiral restates its
result many times, so the gate needs only moderate recall per position; what it must have is a very low false-fire
rate everywhere else - in prompts, in normal reasoning, in answers, and at results the model later abandons.

## Approach
1. **Parent runs.** Create the case with `python -m wsp.case init <case> --profile qwen35-0.8b-think --parent
   parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --subset dev128=dev:128 --subset
   holdout256=holdout:256`. `run` drives a single GPU; to use all four, give each GPU its own id list
   (`CUDA_VISIBLE_DEVICES=<i> python -m wsp.gen --profile qwen35-0.8b-think --model <dir> --ids <ids_i.json> --n 2
   --out <part_i.jsonl>`, the dir from `wsp.case.model_path(<case>, <model>)`), join the parts as
   `$WSP_WORK/runs/<case>/<model>__<set>.jsonl`, and let `python -m wsp.case run <case> <model> <set> --n 2` grade
   and pair that file (seeds depend only on problem and sample, so nothing is generated twice). The parent on
   `dev` comes first and takes longest.
2. **Labels.** `python derive/settled.py label <runs of the parent> --out labels.jsonl`, run on the sandbox from
   `/workspace/work/` (copy `derive/` there): each sentence end of the thinking becomes `pos` (the trace's own
   final result - or the practice item's gold result when the trace was cut - stated for the second time or
   later), `hard` (a result given as the answer twice or more, then abandoned for the right final answer), `work`
   (before the final result first shows up) or `close` (the last sentence before the model's own `</think>`).
   `ifsynth` items are graded on their instructions and have no gold result, so all their sentence ends are
   negatives (`work`): they keep the gate silent on instruction-following answers. Correct the extraction if it
   misses results; never tune it on edit results.
3. **Reads.** `python derive/settled.py read labels.jsonl --model <parent model dir> --layers <every second layer
   from about a third of the depth to the last but one> --out reads.npz`: one forward pass per trace, recording at
   the labelled positions and at random prompt, thinking and answer positions the MLP input (what a gate reads),
   the layer output (where its write lands) and the unit activations.
4. **The gate, per layer.** `python derive/settled.py fit reads.npz` fits, for each layer, a read direction of
   `pos` against all other positions (`wsp.gate.build`, threshold above every training negative) and reports, on
   problems held out, the recall on `pos` and the false fires on each kind of negative. A threshold above every
   training negative sits on a few outliers and keeps almost no recall; set it instead for a small target
   false-fire rate (`--fire-rate`, e.g. 1e-3 of the negatives) and read the held-out curve: recall on `pos`
   against fires on `rand_think`, `rand_prompt`, `rand_answer`, `work` and `hard`. Pick the layer and rate with the
   best held-out recall at false-fire rates the dose can tolerate (step 6 decides); middle-to-late layers are
   preferred (the write has to reach the next-token decision); the last layer is out.
5. **The write and the dose.** The unit writes `beta` times the model's own difference between the state where it
   does close (`close`) and the settled state where it goes on (`pos`), from the same layer; `beta` is the dose. The
   unit it replaces is the layer's least-used one (smallest mean activation). `python derive/settled.py spec
   reads.npz --layer L --beta B --case <case> --name <edit>` writes the spec, stacked on the parent, with the unit's
   values inline in the spec (`wsp.upload` carries only `u` / `cols` / `rows` arrays into an upload, so values kept
   in an npz would be missing from the delivered edit).
6. **Selectivity and controls.** Controls come from the same command with `--control`: `random_write` (a write of
   the same norm in a random direction), `shuffled_read` (the gate fitted to random positions instead of settled
   ones) and `silence` (the replaced unit only zeroed: what giving it up costs). `python derive/settled.py select
   labels.jsonl --reads reads.npz --case <case> --edits <edit> <controls>` measures, teacher-forced on the held-out
   problems, the log-prob of closing the thinking at settled positions (it must rise for the edit and for no
   control) and of the model's own next text at `work` and `hard` positions (within ~0.02 nats/token of the parent
   for every model). Screen layers and doses like this, in minutes, and generate only the best few. Run the dose
   series here first: the closing log-prob at settled positions does not keep growing with the dose (a write that
   is too strong overshoots), and a gate that only nudges it does not stop real spirals - in a first trial on
   the parent, a dose that raised it by about 0.2 nats/token changed nothing in generation. Take forward only
   doses that raise it clearly while `work` and `hard` stay flat; if no dose does, move the gate to a later layer
   or derive the write from the closing itself (the mean gradient of the closing text's log-prob with respect to
   the layer output at settled positions) before generating.
7. **Real generation.** A short `beta` series and the controls on `dev128` with 2 samples, spread over the GPUs as
   in step 1, each paired with the parent. Keep the smallest dose that cuts fewer answers at the limit and raises
   accuracy while the answers that finish lose nothing on any source; `ifsynth` must not move. One source losing
   ground rules the edit out even when the average improves, since every benchmark has to hold. Noise is large at
   this size (a source can move ~5 points by chance on 128 problems x 2 samples): give the final two or three
   doses as many problems as the time allows before choosing.
8. **Deliver.** Upload the first edit that passes 6-7 as early as possible and replace it when a better one is
   confirmed: `python -m wsp.upload <case> <edit> --report <REPORT.md> --dry-run`, then `axonry-check
   $WSP_WORK/deliver/<case>__<edit>/edit --task task.json` on the package, then the same upload with `--task
   task.json`. Freeze the chosen edit, compare it with the parent on `holdout256`, and finish with a report of all
   measurements: label shares per source, recall and false fires per layer and kind, selectivity, controls, the
   doses tried, cut shares and accuracy per source.

## What to expect
Thinking that stops soon after its result is settled: fewer answers cut at the limit on math and multiple-choice
problems, shorter finished answers, the same accuracy where the model already finished, and instruction-following
answers unchanged. If no layer separates settled positions from the rest at a tolerable false-fire rate, report
the recall and false-fire figures and stop rather than ship a gate that fires in the middle of reasoning.
