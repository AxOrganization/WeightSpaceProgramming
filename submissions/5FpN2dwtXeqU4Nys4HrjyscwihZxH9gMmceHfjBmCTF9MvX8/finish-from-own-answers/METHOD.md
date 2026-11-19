# Teach the model, from its own answers, to finish

## The problem
In thinking mode the model's reasoning often never arrives. Removing the single "loop direction" (the toolkit's loop
edit) helps, but on practice problems a large share of answers still ends at the token limit: the reasoning restarts,
re-checks a result that was right, or drifts into a block that repeats, and the answer is missing. The same problem
finishes in one sample and is lost in the next, and the answers that do finish are mostly right. So the ability is
there and is used unreliably. Measure this first, on the starting model (the parent edit if the entry has one, else
the base), on practice problems of every task type (GSM8K and MATH train, MMLU-Pro practice, IFEval-style practice;
never the benchmarks), with several samples per problem:

- the share of answers cut at the limit, and of those the share whose tail repeats (`kind` = `cut_loop`);
- accuracy of the answers that finish, against accuracy of all answers;
- the share of problems solved in at least one sample, against the mean accuracy.

If few answers are cut, or finished answers are no better than all answers, this method does not apply: say so and
stop.

## Why we think it happens
A hand-made contrast (a repeated block against the real continuation) names one symptom and yields one direction.
What is left after it is not one symptom, and we expect it not to be one direction either: at many layers the MLPs
and the attention outputs write components into the residual stream that tilt the next token toward "continue as
before" rather than "take the next step and conclude", each by a little. No single one of them stands out in a mean
difference, which is why adding more hand-made directions gives diminishing returns.

The model's own generations contain the contrast without anyone naming it, in two forms:

- **what it should do**: its answers that finished and were right, on the same kind of problems;
- **what it should not do**: the tails of its own answers that ran into the limit repeating themselves.

An edit that makes the first more likely and the second less likely, token by token, and is as small as possible
otherwise, is the edit we want. The allowed op `dir_ablate` is enough to express it.

## Approach
`dir_ablate` multiplies a matrix by (I - alpha u u^T) on the side that touches the residual stream: alpha > 0 removes
what the matrix reads or writes along the unit direction u, alpha < 0 strengthens it. The toolkit takes u from a
mean difference; here u and alpha are fitted. Per layer and side: a small orthonormal set of directions shared by
the layer's matrices; per (matrix, direction): one alpha, starting at 0 (the starting model). With orthonormal
directions the fitted model is exactly a list of `dir_ablate` ops, so what is trained is what is uploaded.

1. **Splits.** Three disjoint sets of practice problems, each with all four task types: `fit` (answers to learn
   from), `dev` (every choice), `hold` (one confirmation at the end). `derive/sets.py`.
2. **The starting model's own answers.** Two jobs, started together at the very beginning, on separate GPUs,
   because both are slow: (a) the baseline, the starting model on `dev` at the full token limit, two samples per
   problem; every later comparison is paired with it; (b) the starting model on `fit`, four samples per problem,
   with a lower token limit (the fit cannot read longer answers anyway, see `fit.py --max-len`). Both use the
   model-card sampling. `derive/pgen.py` runs `derive/traces.py gen` over the GPUs it is given.
3. **Select** (`derive/traces.py pick`): as positives, per problem, finished correct answers, chosen at random among
   them and not by length (problems the model only sometimes solves count more than those it always solves); as
   negatives, answers cut at the limit whose tail repeats. Read a few of each before fitting: the negatives must
   really be loops, the positives really clean. The loop mark asks for word-for-word repetition, so the negatives
   may be far fewer than the cut answers; a few dozen are enough. Do not fill them up with other cut answers: an
   answer cut while it was still making progress is not a failure to unlearn, and the fit cannot tell the two apart.
4. **Fit** (`derive/fit.py`): maximise the log-likelihood of the positives' tokens; on the last tokens of each
   negative minimise log(1 - p(token)) instead, so that continuing the loop becomes unlikely without prescribing
   what comes instead; pull every alpha toward 0; and on the answer part of each positive (after the thinking
   closes) hold the edited model to the starting model's own next-token distribution (`--tether`), so that the fit
   changes how the reasoning proceeds and not how answers are worded. Hold out some positives and watch their
   likelihood. Which matrices
   to edit is a choice to make from evidence, not from this text: fit at least (a) the MLP matrices and the
   attention output projections only (`--targets`), and (b) all matrices, in parallel on two GPUs, and compare them
   in steps 5 and 6. A third fit without the tether (`--tether 0`) is worth a GPU when one is free: it shows what
   the tether costs and what it protects.
5. **Look at the fit before generating.** Where the strength went (layers, matrices) and whether the fits agree on
   that profile; and build the control: the same directions with the alphas shuffled within each matrix (same norm
   of change, `--export ... --shuffle-alphas`). An effect the control also gives is a generic perturbation, not
   this method.
6. **Decide on real generation**, on `dev`, at the full token limit, paired with the baseline on the same problems
   and seeds (`derive/compare.py`): accuracy per task type, share cut at the limit, share that stop without a final
   answer, length. A good edit removes most cut answers, gains on every task type, and loses on none. Fix the rule
   before reading the numbers: a task type that is down by more than one standard error counts as a loss to
   repair, not as noise, and the instruction-following type is the one to watch, since it has the least to gain
   from finishing; among edits within noise of each other overall, take the one that is best on its weakest type,
   then the one with fewer matrices. The control needs only half of `dev`.
7. **Confirm on `hold`** for the chosen edit, at the full limit, one sample per problem, paired with the starting
   model (start the starting model's `hold` run as soon as a GPU is free, it is the slow half). Then freeze, check
   the limits (`axonry-check`), and upload with a report of every step: the baseline, what each variant did, the
   control, what did not work.

Upload as soon as one edit passes step 6, with the report as it stands, and replace both when something better is
confirmed. Rewrite the report after every measurement, so that the upload never lags behind what is known.

## Running the code
`S` is the starting model's directory. With a parent, the toolkit builds it: after `python -m wsp.case init <case>
--profile P --parent <parent edit.json> ...`, `python -c "from wsp.case import model_path; print(model_path('<case>',
'base'))"` prints it; without a parent it is the base model's directory. `P` is the profile name. One job per GPU;
times are for 24 GB GPUs.

```
python derive/sets.py --out sets
# together, detached: baseline on dev (full limit) on one GPU, answers on fit on the other three
python derive/pgen.py --gpus 0     --model S --profile P --ids sets/dev.json --n 2 --salt h0 --out runs/dev_start   # 45-60 min
python derive/pgen.py --gpus 1,2,3 --model S --profile P --ids sets/fit.json --n 4 --max-tokens 8192 --out runs/fit # 40-70 min
python derive/traces.py pick --runs "runs/fit.*.jsonl" --loops 900 --out traces.jsonl
# one fit per GPU, detached (15-45 min each, by the number of answers; --max-steps caps it)
CUDA_VISIBLE_DEVICES=1 python derive/fit.py --model S --traces traces.jsonl --out fit/w \
    --targets down_proj o_proj out_proj gate_proj up_proj
CUDA_VISIBLE_DEVICES=2 python derive/fit.py --model S --traces traces.jsonl --out fit/all
CUDA_VISIBLE_DEVICES=3 python derive/fit.py --model S --traces traces.jsonl --out fit/wn --tether 0 \
    --targets down_proj o_proj out_proj gate_proj up_proj
# a fitted state -> an ordinary edit spec in the case (stacked on the case's parent when there is one)
python derive/fit.py --export fit/w/state_last.pt --name W --profile P --spec-dir <case dir>/edits \
    [--parent ../parent/edit.json]
python derive/pgen.py --gpus 1 --model <built model of W> --profile P --ids sets/dev.json --n 2 --salt h0 --out runs/dev_W
python derive/compare.py "runs/dev_W.*.jsonl" "runs/dev_start.*.jsonl"
```

`--parent` is the path of the parent spec relative to the spec directory, as the toolkit's own edits have it.
`fit.py --export` writes `<name>.json` + `<name>.npz` into the case's `edits/`, so the toolkit builds, runs and
uploads it like any other edit (`wsp.case run` builds the model on first use; `python -m wsp.upload <case> <name>
--report REPORT.md --task task.json`). The export keeps the payload under the limit as the upload counts it (an
array shared by several ops is stored twice there) by dropping the weakest directions if needed; check the
relative changes in the built model's manifest and with `axonry-check` all the same. `pgen.py` skips finished
shards, so it can simply be re-run after a failure; `--gpus` and `--per` (processes per GPU, default 2) size it.
`fit.py -h` lists the knobs (rank, epochs or steps, the pull toward 0, the weight of the negatives); the defaults
are a starting point, and `dev` decides. A negative longer than `--max-len` keeps its end, where the loop is. To
stop a job, kill its processes by PID (a generation job has one child process per shard, each with its own engine
process holding the GPU), and check `nvidia-smi` afterwards.

## What to expect
Far fewer answers cut at the limit, shorter answers, and a gain on the task types where the starting model often
failed to finish, largest where it finished least; where it already finished nearly always, no gain and no loss. Accuracy of the answers that finish will be lower than before, because problems that
used to end in a loop now end in an answer, right or wrong; what matters is accuracy over all answers. Some answers
will now stop without a final answer where they used to loop; that is still a miss, and worth reporting by task
type. If a task type loses, or the control gives the same gain, or two seeds disagree on where the strength went, reduce the scope
(fewer matrices, lower rank, stronger pull toward 0) and say so in the report rather than uploading a larger edit.
