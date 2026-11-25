# Early-decision credit for maths on the violation-onset model, held on the problems it already solves: the instruction side is won, the maths answer is decided in the first half of the thinking, and the credit must leave the easy answers alone

This entry adds one layer, **EC**, to the published model of its declared parent `0009578577-0019` (#138,
"violation-onset-on-130"; `entry.json`), and uses that model as it is. EC is derived in the run from practice maths
problems and the model's own answers: the stretch of the thinking where a maths answer is won or lost is credited,
with same-prefix continuation pairs and a loss window right after the branch point, fitted into per-unit and per-head
gains, **while the model's own answers to the practice problems it already solves are held in place**. The upload is
EC alone, stacked on #138's published edit. One driver, `derive/run_ec.py`, runs every step.

| part | what | source |
|---|---|---|
| start | #138's published edit as it is: #130 (the layer of `0009569144-0010` carried onto `0009570269-0024` as a change, request gains) + #138's violation-onset gains for instructions | the bundle (`0009578577-0019/edit/` and the chain it stands on) |
| EC credit | same-prefix pairs from early points of the thinking (10 / 25 / 50 %), loss restricted to the 2,048 tokens after the branch, MLP-unit and attention-head gains of layers 4-21; eight fits averaged in log space, strength 0.75 | method of `0009568722-0012` / `0009568726-0005` / `0009568831-0002` (#105-#107); their published `branch2.py`, `pref_gains2.py`, `fit_gains.py`, unchanged |
| easy-problem hold | every fitting step also holds one of the start's own answers to the bank problems it solved 4 times out of 4 (KL to the start, weight 30, first 6,144 generated tokens) | ours: the hold set and its use (through the published `--keep` option of `pref_gains2.py`) |

Where it comes from, plainly: EC is the published method of #105-#107; `0009580212-0015` (#143) applied it to #130.
This entry applies it to #138, the model that adds violation-onset credit for instructions to #130, and adds a hold
of its own that answers what such credit does to the problems the model already solves (section 1).

## 1. What the start gets wrong, and what the credit can get wrong

**The start's maths.** #138 keeps instruction requests well; on maths it still answers wrong problems it can solve.
On 600 practice `math_train` problems of levels 3-5 (outside the case's sets; four answers each at the profile's
decoding and token limit), 200 problems have at least one finished correct and one finished wrong answer, both longer
than 3,000 tokens. Cutting one correct and one wrong trace of each at a line break near 10 %, 25 % and 50 % of its
thinking and letting the model continue four times from there (the rest sampled again, 24k new tokens; 4,800
continuations):

| continued from | prefix of the correct trace | prefix of the wrong trace | gap |
|---|---|---|---|
| 10 % of the thinking | 48.4 % | 38.1 % | 10.2 |
| 25 % | 54.6 % | 34.2 % | 20.4 |
| 50 % | 62.1 % | 29.5 % | 32.6 |

The answer is decided over the first half of the thinking: the gap triples between 10 % and 50 %, and from the
middle of a wrong trace less than a third of the continuations end right. 683 of the 1,200 branch points (57 %) have
a mixed outcome: the same prefix gives pairs that differ only in what was decided after the branch (1,366 pairs).

**What the plain credit can get wrong.** The fitted gains scale whole MLP units and attention heads, so they change
the model's way of thinking everywhere, not only at the hard decisions the pairs come from. A credit that rewards the
continuations which go on to the right answer also rewards their habits (another check, another pass over the
computation), and on problems the start already solves in a few thousand tokens that habit only lengthens the
thinking and gives it more chances to undo a right result. The pairs say nothing about those problems, because they
come only from problems the start sometimes gets wrong. Hence **the hold**: the bank problems the start solved 4 times
out of 4, every answer under 12,000 tokens (136 of the 600 in our run), each keep the start's own shortest answer as
a KL anchor in every fitting step (`pref_gains2.py --keep`, weight 30, first 6,144 generated tokens). The gains that
move the hard decisions are then the ones that leave these answers where they were; in our fits the KL of the held
answers stayed near 0.003 per position through both epochs while the preference accuracy on the pairs reached 1.0.

## 2. The edit (layer EC)

1. **Bank, mixed problems, branches** as in section 1 (`branch2.py --kinds wrong right --points 0.1 0.25 0.5 --k 4
   --max-new 24000`, presence penalty on the prefix as published). The hold set comes from the same bank.
2. **Pairs.** At each branch point a continuation that ends with a correct answer against one that does not (an
   unfinished continuation counts as not correct: `--bad-cut`), at most 2 pairs per point.
3. **Fit.** Per-unit MLP gains and per-head attention gains of layers 4-21, DPO-style preference (beta 0.1) plus 0.2 x
   NLL of the correct continuation, the loss restricted to the first 2,048 tokens after the branch point
   (`--window 2048`), 2 epochs, plus the hold (`--keep ec_work/keep_easy.jsonl --keep-w 30 --keep-len 6144`). Eight fits
   (seeds 1-8), averaged in log space: single fits move different units, and their average is the steadier estimate.
4. **Strength 0.75**: three quarters of the averaged log-gains (`fit_gains.py export --k 0.75`). The hold keeps the
   easy answers in place during the fit, but the exported gains still act on every problem; a strength below 1 keeps
   that step moderate.

What is changed: `scale_rows` on `up_proj` and `scale_cols` on the attention output projections of layers 4-21, about
0.4 MB.

## 3. Running it (4 x RTX 4090; about 4 h 40 min)

The toolkit, the base and the environment (`WSP_WORK`, `WSP_CASES`, `WSP_MODELS_DIR`, `PYTHONPATH`) are set in the
sandbox: do not override them. Copy `submission/`, `task.json` and the declared parent's directory `0009578577-0019/`
(with the chain inside it, as the bundle has it) to `/workspace/work/`, then:

```
cd /workspace/WeightSpaceProgramming
nohup python /workspace/work/submission/derive/run_ec.py > /workspace/work/run_ec.log 2>&1 &
grep PHASE /workspace/work/run_ec.log
```

| stage | GPUs | what | time (4 x RTX 4090) |
|---|---|---|---|
| case | CPU | case `ec` on #138's published edit; its model built once | 10 min |
| lists | CPU | the bank list (600 problems, levels 3-5) | 1 min |
| bank | 4 | 600 problems x 4 answers, full token limit; mixed problems (up to 200) per GPU; the hold set | 50 min |
| branches | 4 | one shard per GPU, 3 points x 4 continuations per trace | 90 min |
| fits | 4 | seeds 1-4, then 5-8, with the hold | 2 x 65 min |
| export | CPU | log-mean of the 8 fits at strength 0.75 -> spec `F` | 1 min |

The driver handles one known fault itself: when every shard of a generation job has printed its `[gen] N answers`
line and the job has not ended five minutes later, a vLLM worker is stuck at shutdown; the driver sends that job's
workers SIGTERM (the answers are already collected). If a stage stops with an error, read its log in `ec_work/`, fix
the cause and start the driver again: finished stages are skipped. Never `pkill -f` a pattern.

## 4. Upload

When `PHASE DONE` is printed: write `/workspace/work/REPORT.md` with the numbers the run printed (mixed problems and
hold rows, the branch statistics of `ec_work/br_ec_*.log`, pairs, keep rows, pref-acc and keep-kl of
`ec_work/fit_E1.log`, the export line), then
```
python -m wsp.upload ec F --task /workspace/work/task.json --report /workspace/work/REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/ec__F/edit --task /workspace/work/task.json
python -m wsp.upload ec F --task /workspace/work/task.json --report /workspace/work/REPORT.md
axonry-done
```
The upload names #138 as its parent and carries EC's ops only (about 0.4 MB). If the second round of fits (seeds 5-8)
cannot finish 45 minutes before the deadline, stop it, run `NFITS=4 python .../run_ec.py` (stage 6 exports the
average of the four finished fits at strength 0.75 as `F`) and upload that.

Use only `practice:` data and the models' own answers. Never load `bench:` sets or anything about the evaluation;
never download models or data.
