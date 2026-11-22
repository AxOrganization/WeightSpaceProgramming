# Close and unloop the champion's thinking: its keep-verifying and loop states, measured on its own answers and removed from four attention outputs

This entry stacks on its parent (`entry.json`): start from exactly that model. The parent's last layer makes the
thinking longer at the points where it decides to check again or to conclude (a reversed policy gradient), and
closes traces that stay stuck late. It has no layer that removes the state in which a trace keeps re-verifying a
result it has already stated, or the state in which it repeats its own earlier text. This method measures those two
states on the parent's own answers, removes them from four attention outputs, and chooses the strength of the first
on paired practice screens inside the run.

Everything is measured in the run, on the toolkit's practice sets and the parent's own generations. Nothing is
downloaded and no arrays are shipped.

## 1. The failure that is left, and the layer

**The parent thinks long.** On fresh practice problems, 4 answers each at a 12k cap, the parent's answers were cut
at the cap for 15% of `gsm8k_train` problems and 52% of `math_train` problems (our practice measurement; at the
profile limit of 32768 almost every word problem finishes, 0.2% cut on 600 problems x 4). Its long traces re-check
results they have already stated and repeat their own earlier text: tokens that do not change the answer and, on hard
math, run into the limit.

**Making it think longer still does not help.** We fitted the parent's own fork-credited policy gradient again on its
answers and applied it reversed (the parent's own last recipe, one round further), paired with the parent on 600
fresh `gsm8k_train` problems x 4 at 32768: -0.25 +- 0.79 points at L2 norm 1.83, -0.88 +- 0.78 at 0.92; the same with
row and column gains: -0.58 +- 0.77. The thinking needs a better end, not more length.

**The layer** (`derive/dirs.py`; the recipe of Parts A and B of the published entry decisive-steps, measured again
here on this parent with the toolkit's contrasts in `wsp/directions.py`):
- Conclude contrast: a right, finished answer of the parent cut right after its thinking first states the right result;
  the good branch closes the thinking, the bad branch keeps verifying (the model's own next text). The per-layer mean
  difference of the residual stream isolates the keep-verifying state. It is projected out of the attention outputs
  (`self_attn.o_proj`, `dir_ablate`, side out) of layers 15 and 19.
- Loop contrast: a right answer cut at a sentence boundary; the good branch is the model's own continuation, the bad
  branch repeats the model's earlier text. Projected out of `self_attn.o_proj` of layers 11 and 23 at strength 1.
- Pairs from the parent's answers to practice word and math problems, up to 512 per contrast. A direction whose
  split-half stability at one of its layers is below 0.8 is left out. Four `dir_ablate` ops in all.
- On an earlier champion (0009547524-0008, this parent's ancestor), the authors of decisive-steps measured these two
  parts on practice splits: word problems +2.2 and +1.3 points on two `gsm8k_train` splits. The parent's own layers do
  not target these states.

**Candidates:** `AB` (conclude at strength 0.75) and `AB1` (conclude at 1); the loop part at 1 in both. The same
measured directions; only the strength of the conclude part differs.

## 2. Practice sets (`derive/sets.py`, all outside the case's dev and holdout and disjoint)
- `pool`: gsm8k_train 400, math_train 200; the parent's answers, 4 each, cap 12000 (the pairs of the two contrasts).
- `scr_g`: gsm8k_train 500, 4 samples, profile limit 32768. `scr_m`: math_train 120, 2 samples, cap 16384.
  `guard`: ifsynth 80 and mmlupro 80, 2 samples, 32768.
- Every screen pairs each candidate with the parent on the same problems and per-problem seeds (same salt).

## 3. The choice (fixed before the run, `derive/pair.py choose`)
A candidate is eligible if neither screen source (word problems, math) is below the parent by more than max(1 SE,
1 point) and neither guard source (ifsynth, mmlupro) by more than max(2 SE, 4 points), SE by problem. Among the
eligible, the largest score = word-problem gain + math gain (points); ties within 0.25 go to `AB`. With none eligible,
the largest score. `AB` is uploaded as soon as it is built, so a complete upload is always in place; the choice
replaces it if it differs.

## 4. Running it
Work in `/workspace/work`. From the unpacked bundle, copy the three parts there: `submission`, the start's directory
(`parent` in the older layout, the start's `<sid>` in the newer: here `0009554110-0010`) and `task.json`, e.g.
```
scp -F ~/.axonry/ssh/config -r submission <the start's directory> task.json sandbox:/workspace/work/
```
so that `/workspace/work/task.json` and the start's `edit/edit.json` exist; `cd /workspace/work && python
submission/derive/start.py` prints the start's path.

**Start the chain** (it creates the case `k1` on the start and the sets, and runs every GPU step itself, in its own
session; it returns at once):
```
cd /workspace/work && bash submission/derive/chain.sh start
```
It reads the deadline from `/etc/axonry-deadline`. If `start` prints a deadline that is not the run's real one, stop it
(`chain.sh stop`), `rm state/deadline` and start it as `DL=$(date -d '<deadline from the brief> UTC' +%s) bash
submission/derive/chain.sh start`. `start` again resumes where the chain stopped: every finished step is kept
(`state/*.done`); while the chain runs, `start` changes nothing.

**Check on the chain at least every 10 minutes** (the run's stall rule is 20 minutes), with one command of at most
9 minutes:
```
cd /workspace/work && bash submission/derive/chain.sh status 8
```
It returns as soon as the chain writes a new status line (after 8 minutes at most) and prints the last status lines,
the steps done, what is uploaded and the minutes left. Between checks, write REPORT.md and PROGRESS.md.

**If `status` says `chain NOT running` before `DONE`**: read the last lines of `logs/status.log` and the log it names;
fix an environment problem inside `/workspace/work` (for example a full disk) and run `start` again.

**The chain's steps** (times are estimates on 4 x RTX 4090):
| step | what | time |
|---|---|---|
| 1 | case `k1` on the start, the start's build, the sets | 5 min |
| 2 | the parent's answers on `pool` (4 GPUs) | 15 min |
| 3 | the pairs and directions of both contrasts (GPU 0); `AB` and `AB1` | 10 min |
| 4 | builds; the first upload (`AB`, CPU) | 5 min |
| 5 | screens: the parent, then `AB`, `AB1`, each on `scr_g`, `scr_m`, `guard` (4 GPUs) | 30 min each |
| 6-7 | the table, the choice, its upload if it differs | 5 min |

A generation that stops is retried once (each has a 40-minute limit). A candidate's screen starts only with at least
60 minutes left; the choice is made among the candidates screened. Uploads (`chain.sh upload NAME` does the same by
hand): `python -m wsp.upload k1 NAME --report REPORT.md --task task.json --dry-run`, then `axonry-check` on the
package's `edit/`, then the upload; one at a time, none with fewer than 8 minutes left. Until `REPORT.md` exists the
chain uploads an automatic report (`REPORT.auto.md`: every status line and the screen table).

**task.json** holds the upload credentials: keep its contents out of logs, PROGRESS.md and the report.

**At the end.** Once `status` shows `DONE` (or 30 minutes before the deadline if it has not):
1. Finish `REPORT.md` (section 5), written to a temporary file and moved into place.
2. If the chain is not `DONE`: `bash submission/derive/chain.sh stop` (it waits for an upload in progress).
3. Upload the final choice with the finished report, no later than 20 minutes before the deadline:
   `bash submission/derive/chain.sh upload $(cat state/final.txt 2>/dev/null || cat state/uploaded.txt)`.
4. If `status` shows that upload and nothing is left worth trying, run `axonry-done`.

## 5. REPORT.md
- The parent on `pool`: accuracy and cut share per source.
- The two contrasts: pairs and traces, split-half stability at the four layers, the cosine of the two directions
  (`logs/dirs.log`); which parts were kept.
- The screen table (`logs/screen_table.log`: each candidate against the parent, per source, gain +- SE, cut shares),
  the choice and its reason (`runs/choice.json`), what was uploaded and when.
- What was skipped and why (a part left out, a screen not run for time).

## 6. Rules kept
Practice sets only, choices on practice screens outside dev and holdout; holdout is not used. The directions and every
choice are measured in the run from the parent's own generations.
