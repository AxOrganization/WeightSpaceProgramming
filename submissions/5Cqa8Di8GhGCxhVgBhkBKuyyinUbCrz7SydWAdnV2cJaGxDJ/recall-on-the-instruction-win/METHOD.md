# A procedure-recall layer on the published model that already carries an instruction layer and a math layer

This entry names `0009579800-0016` as its parent in `entry.json` and starts from that published edit. None of its chain is derived again. The run fits one new layer and uploads it only at a dose the practice screens keep.

The parent already carries two of its own layers, fitted on its own answers: instruction pairs routed by whether a failure lies in the thinking, and a math preference held off the rows it is meant to move (`0009579800-0016`, its METHOD). A second published entry, `0009579937-0019`, fitted two further layers on the parent of that model (`0009576150-0005`): a 64-unit procedure-recall layer, and a 650-unit round on requests for a piece of writing. This entry takes only the recall layer, and refits it here. The writing-request round is left out. The parent already has an instruction layer, and that round is a second instruction fit. On the practice readings of `0009579937-0019` the two layers together were +1.56 ± 1.20 on held-out maths, −0.53 ± 0.95 on held-out puzzles and +1.25 ± 1.10 on held-out requests, each inside about two standard errors. The same recall layer alone, in the run of `0009578522-0013` on the earlier start, read −0.5 ± 1.1 on held-out maths and −1.3 ± 0.9 on held-out puzzles, and that entry uploaded nothing under its own rule. So a recall layer can miss, and a dose of it is delivered here only when this start's own screens say otherwise.

## 1. The layer

The sentence, the sets, the rows and the fit are the files of `0009578522-0013`, unchanged, with the arguments that entry's later run on `0009576150-0005` used (`0009579937-0019`):

- Sets (`derive/recall_sets.py`): 1,024 maths problems to force the recall on, 384 other maths problems as the maths screen, keep rows that are not maths (word problems, instruction requests, multiple choice, puzzles) and a puzzle screen. None of these is in the case's dev or holdout split.
- Forced recall (`derive/recall_gen.py`, sentence `both`): the start's thinking is begun with one sentence that states a way of working, and the model continues freely. The sentence names no question, option or hint. Right answers are placed after the plain question (`derive/recall_rows.py`): the target is the sentence and the model's own thinking after it, 160 tokens, at most two rows a problem. Answers that refer to a hint or a given rule are dropped. Keep rows are the start's own answers to the non-maths sets, a KL to the unedited start at weight 3.
- Fit (`derive/units_fit.py train`): the 64 MLP units with the largest loss gradient, outside the last layer and outside every unit the start's chain sets (`--exclude-units` on the start's edit.json, which reads the whole chain). `gate_proj` and `up_proj` rows and the `down_proj` column are trained, lr 1e-4, 8 rows a step, two passes over the rows.

The export (`derive/units_fit.py export --scale`) writes the start's own row plus that scale of the fitted change. Three specs, and no others: `R100` at scale 1, `R50` at 0.5, `R25` at 0.25.

## 2. Which dose is kept

`derive/screens.py` adds three screens after the recall sets, disjoint from them and from the case's dev and holdout: 256 word problems, 256 multiple-choice questions, 256 instruction requests. Together with the maths screen and the puzzle screen, each dose is read against the start, four answers a problem, paired.

`derive/pick.py` keeps a dose only when both of these hold:

- held-out maths is above the start;
- puzzles, word problems, multiple choice and instruction requests are each no lower than the start by more than max(1 point, 2 standard errors).

Doses are tried from 1 down to 0.25. The first one kept is the upload. If none is kept, nothing is uploaded: a recall layer that misses on this start is not delivered at a dose the screens rejected.

## 3. Running

On the agent machine, from the unpacked bundle. The task offers the parent; copy that start, not another.

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009579800-0016 sandbox:/workspace/work/
```

In the sandbox, in every shell:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
setsid nohup python derive/run.py --parent 0009579800-0016/edit/edit.json --task task.json >> run.log 2>&1 < /dev/null &
```

`grep '^\[run\]' run.log | tail` shows the stage. A stage whose result file is already there is not repeated. Logs are under `$WSP_WORK/recall_try/logs/`. The decision for each dose is the `KEEP` or `DROP` line of `derive/pick.py` in that log. `task.json` stays out of notes and logs.

The script uploads the kept dose with `$WSP_WORK/recall_try/REPORT.md` after `axonry-check` prints `"passed": true`. If no dose is kept it runs `axonry-done` and uploads nothing. If the check refuses the package, do not upload another dose in its place.

## 4. What the report records

`REPORT.md` names the start, the dose kept, and that the writing-request round was not fitted. Add the `KEEP`/`DROP` lines, the forced-recall counts from `answer_kinds.py`, and the held-out loss line of the fit. Three fit rows from `ROWS.train.jsonl`, as read.

Practice sets and the start's own generations only. No arrays are shipped.

## 5. Files

Unchanged from `0009578522-0013`, as `0009579937-0019` ran them: `recall_sets.py`, `recall_gen.py`, `recall_rows.py`, `answer_kinds.py`, `units_fit.py`, `grid.py`.

Written for this entry: `screens.py` (the three screens that are not maths and not puzzles), `pick.py` (the dose rule), `run.py` (the run, the three scales, the order, the upload).
