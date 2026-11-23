# The answer round once more, on the model that carries it: a second preference fitted on what the parent's own answers still split on, from requests and problems its round did not draw, with the parent's last-layer pairs lifted for the fit

This method stacks on the published edit named in `entry.json`, the parent: this miner's finished entry
0009569144-0010 (the same GitHub account as this entry). Everything it delivers is derived in the run: from the
parent's own edit file, from four rows of the weights of the edit the parent stands on (the bundle holds that edit
inside the parent's directory), and from the answers that the parent's model gives to practice problems. It
delivers one edit, the answer step `A`: about 650 rewritten MLP neurons below the last layer (`set_rows` on
`gate_proj` and `up_proj`, `set_cols` on `down_proj`), the average of up to seven short preference fits, 0.45 of
the change kept.

What the parent carries, by its own METHOD.md:

1. **its start**: the two wrap-up units of the edit below it silenced, two directions projected out of attention
   outputs;
2. **its rules**: an exit pair at layer 6 and a break pair and a commit pair at the last layer, rewritten MLP
   neurons that close a thinking which would otherwise run to the token limit;
3. **its round**: a preference between two answers the model writes after the very same thinking, one that does
   what was asked and one that does not, fitted on three kinds of pairs (instruction requests recombined from the
   practice material, the practice requests as they are, mathematics answers) into about 650 MLP neurons, four fits
   averaged, 0.45 of the change kept.

This entry runs that round a second time, on the model that already carries it. Nothing of the parent is built
again or changed: its start and its rules stay as published, and `A` is laid on top.

Where the parts come from, plainly. The answer branching and the preference fit (`derive/rollouts.py`, `branch.py`,
`train_sparse.py`, `avg_sparse.py`, `dose_sparse.py`) are the published work of the author of 0009550031-0005, and
the instruction requests recombined from the practice sets (`derive/mkreq.py`, `genreq.py`, `branch_req.py`,
`train_req.py`) are the published work of the author of 0009559255-0006; all nine scripts are used unchanged. The
mathematics pairs, the anchors, the protected neurons and the dose are the parent's round (this miner's).
`derive/start.py`, `units.py`, `round.py` and `decide.py` are this entry's; `final.py` and `compare.py` are those of
the parent. What this entry adds:

- **the measurement that the split is still there after one round, and how much of it.** On requests the parent's
  round never drew, the model this round is fitted on follows 69.1% of the recombined requests (the model
  before the parent's round: 62.6%, the parent's METHOD.md), and after one and the same thinking its answers still
  split for 6.9% of the thinkings (before: 9.6%). The first round took a part of the split away, on
  requests it had not seen; the larger part is left (section 1.1);
- **a second round from other material.** The recombination is random: which instruction sentence goes on which
  body. This entry draws its requests with other seeds than the parent's round and takes another draw of practice
  mathematics problems, so the second fit does not see the first one's pairs again (section 1.2);
- **a larger round.** The parent's run needed half of its time for the start and the rules. They are in the parent
  now, so this run has that time for the round: two sets of recombined requests, 3,000 mathematics problems and up
  to seven fits, where the parent's round had one set, 2,000 problems and four fits (section 1.2);
- **the fit with the parent's last-layer pairs lifted.** The parent has two pairs at the last layer, a break pair
  and a commit pair; the commit pair starts the answer after a close that the exit pushed. An answer round is about
  how the model itself writes an answer after a thinking it has closed, so this round is fitted on the parent
  without those pairs, and its answer step is delivered on the parent as it is (section 1.3);
- **the parent's units are read, not built again.** `derive/units.py` reads from the parent's edit which neurons
  its start and its rules rewrote. The fit never takes one of them, and none of the parent's values is changed
  (section 2).

The figures below are the author's measurements on practice problems, on this entry's parent, in runs of sections 3
and 4 as written, unless they say otherwise. The steps say what to measure first and when to stop.

## 1. The behaviour to change, and why it is still there

### 1.1 After the same thinking, one answer does what was asked and the next one does not: after one round, too

The finished entry 0009550031-0005 found the behaviour for instruction requests and showed how to use it: keep an
answer's thinking up to its closing tag, sample the answer again, and where one answer passes the practice item's
own checker and another fails it, fit a preference for the first. The thinking is the same token for token, so the
pair isolates how the answer is written. The parent fitted such a round on three kinds of pairs. A round is a small
step by construction: four short fits of about 650 neurons, of which 0.45 is kept, because a single fit lands in a
different place each time and more of the change costs outside the pairs (the parent's METHOD.md, sections 1.5 and
7). So the question for a second round is a plain one: after the first, do the answers written after one and the
same thinking still split?

They do. Measured on the model this round is fitted on (the parent with its last-layer pairs lifted, section 1.3),
with requests and problems that the parent's round did not draw:

| answers branched after the same thinking | thinkings | answers right after resampling | outcome mixed | pairs |
|---|---|---|---|---|
| practice instruction requests as they are, `ifsynth` | 735 | 92.9% | 2.6% | 23 |
| the same, `tuluif` | 795 | 90.7% | 3.4% | 32 |
| requests recombined from the practice material, two sets | 5,084 | 74.8% | 6.9% | 451 |
| practice mathematics problems, finished thinkings | 3,056 | 93.0% | 1.5% | 52 |

The parent's METHOD.md has the same table for the model before its round: one set of recombined requests gave 302
pairs at a mixed outcome of 9.6%, the mathematics problems 65 pairs at 2.9%. On requests it had never
seen, the parent's round took the mixed outcome from 9.6% of the thinkings to 6.9% and the answers that pass
after resampling from 67.0% to 74.8%: it carried over to new requests, and about seven tenths of the split is
left. On the practice requests as they are, which both rounds draw whole, about two thirds is left (2.6% and 3.4%
of the thinkings against 4.1% and 4.7%), and on mathematics about half (1.5% against 2.9%). Two sets of recombined
requests give 451 pairs, more than the parent's round had from one; 3,000 mathematics problems give 52.

**What is left is not the same everywhere.** Per kind of recombined request, the share of answers that follow every
instruction (four answers each at a 12,000-token limit; in brackets the model before the parent's round, from the
parent's METHOD.md): a repeat request on a multiple-choice prompt 40.5% (14.2%), on a mathematics prompt
31.2% (6.9%), on a `tuluif` request 53.9% (34.6%), on a word problem 83.1% (74.2%); a word
problem with two or three instructions 74.5% (74.4%); a `tuluif` request with one or two more 73.0%
(69.1%). The parent's round moved the repeat
requests most, the kinds that were followed least, and they are still the ones followed least. Most of the
requests whose four answers disagree are those that carry two or three instructions.

### 1.2 The second round: other material, more of it, the same fit

**Other requests, other problems.** `derive/mkreq.py` cuts instruction sentences (with their checker arguments) out
of the one-instruction practice requests and puts them on other bodies: the repeat sentence on a word problem, a
multiple-choice prompt, a mathematics prompt or a `tuluif` request; two or three sentences on a word problem; one or
two more on a `tuluif` request. Which sentence goes where is drawn from a seed. The parent's round drew with seed
0; this round draws two sets, with the seeds 1 and 2 (1,320 requests each; a request that comes up in both is kept
once). The 3,000 practice mathematics problems are another draw from the pool than the parent's 2,000 (`rollouts.py
pick --seed 12`; the parent's round took the script's own seed 11), and every answer is sampled with other seeds.
The practice instruction requests as they are (`ifsynth`, `tuluif`) are few, so both rounds take all of them; the
model has changed since, and what it still gets wrong on them is what this round fits.

**More of it.** The fit's weak point is its size. One preference fit on a few hundred pairs lands in a different
place each time (the finding of 0009550031-0005); the average over several fits is what carries over, and the dose
keeps a part of it. More pairs and more fits make the average steadier. The parent's run could afford one set, 2,000
problems and four fits, because its start and its rules took the first half of its time. This run starts the round
after twenty minutes: two sets, 3,000 problems and seven fits (one that chooses the neurons, six more of the same
neurons from other seeds, two at a time). `derive/round.py` sizes itself from the time left and falls back to the
parent's size if a machine is too slow for the larger one (step B).

**The same fit.** One preference fit on all three kinds of pairs (the published script, unchanged: about 650 MLP
neurons chosen by the gradient of the pass / fail margin, two epochs), with what the parent's round set for the
sake of the families the fit is not about:

- the mathematics pairs are part of the data, three times over, so the fit that changes how an answer is written is
  told, in the same loss, to write the result the thinking reached;
- the model's own right, finished answers to other practice mathematics problems (the anchors) stand in the data as
  answers to keep: the fit pays for moving away from them;
- the 4,000 MLP neurons with the largest gradient on the model's own right mathematics answers are never chosen for
  the edit (`train_sparse.py --protect`, a switch of the published script), no neuron of the last layer is, and no
  unit of the parent is (`--exclude-neurons-from`, section 2);
- 0.45 of the averaged change is kept, the parent's dose.

The neurons are chosen anew, on this model and these pairs. They may be neurons the parent's round changed or
others: the fit starts from the parent's weights either way, and `A` holds the whole rows and columns it ends with.

### 1.3 The fit with the parent's last-layer pairs lifted

The parent's commit pair writes the opening of a boxed result right after a closing tag that its exit pushed, and
its break pair ends a thinking there; both sit in the last layer and are silent everywhere else (the parent's
METHOD.md, sections 1.1 and 2). An answer round is about something else: how the model itself writes an answer
after a thinking it has closed. On the model that carries the commit, a part of the answers the round draws stands
behind closes where the commit has started the answer, and the fit sees them beside the model's own.

So the round is fitted on the parent with its last-layer pairs lifted: `derive/units.py` writes an edit `U` that
sets the rows and columns of those neurons back to the values they have in the edit the parent stands on, and
`round.py --on U` draws every answer and fits on that model. The answer step is then delivered on the parent as it
is, with its pairs in place. Nothing collides: the fit never takes a neuron of the last layer, and `A` holds no
value of it.

Measured in two of the author's runs of the parent's own steps (the parent's METHOD.md, sections 3 and 4), where
the round was fitted both ways from the same rules and both answer steps were delivered on the same rules with
the break and commit pairs (gains over those rules; 320 dev problems, two answers each; the parent's 192 test
requests, four answers each): fitted without the two pairs, dev +3.4 +- 1.6 and +3.6 +- 1.5, test requests
+1.2 +- 1.1 and +2.6 +- 1.4; fitted with them in place, dev +3.9 +- 1.5 and +3.1 +- 1.5, test requests
-0.4 +- 1.3 and +5.6 +- 1.3. On dev the two ways are level. The test requests are practice requests that the round
draws on, so they can show harm and not gain: the round fitted with the pairs in place lost 8.2 +- 2.4 on the
requests that ask for no comma in one of the two runs (without them, from the same rules: -1.8 +- 1.6), and its
two runs differ by 6.0 points, more than three times the error of that difference; the round fitted without them
reads the same in both within its error. These sets cannot rank the two ways by their mean. What they show is
that the fit without the pairs is the steadier one, and a run is one draw: that is the reason for it.

## 2. The parent's units, read from its edit

`derive/units.py` reads the parent's edit file. The parent's scripts keep the units of its start and of its rules in
full precision and its answer step in half precision, and its METHOD.md gives their order; the script checks both
and stops if the edit is laid out otherwise. It writes two edits of case `s`. Neither is delivered:

- `S0`: the parent's units written again with the parent's own values. It changes nothing; the model of `S0` is the
  parent. It is the list of neurons this entry's fit must not take: the two silenced wrap-up neurons (a fit could
  make them write again), the exit pair, the last-layer pairs;
- `U`: the same, but the last-layer units carry the values that those rows and columns have in the edit the parent
  stands on. For that, the case `g` stands on that edit (the bundle holds it inside the parent's directory), and the
  script reads those rows from its weights. `U` is the model the round is fitted on.

On the author's run it printed:

```
unit kept as it is:   layers.16.mlp.down_proj.weight cols [1161]
unit kept as it is:   layers.15.mlp.down_proj.weight cols [3406]
unit kept as it is:   layers.6.mlp.gate_proj.weight rows [2104, 3245]
unit kept as it is:   layers.6.mlp.up_proj.weight rows [2104, 3245]
unit kept as it is:   layers.6.mlp.down_proj.weight cols [2104, 3245]
unit lifted in U:     layers.23.mlp.gate_proj.weight rows [1439, 655, 3005, 2888] (largest difference between the parent's values and those below: 0.792)
unit lifted in U:     layers.23.mlp.up_proj.weight rows [1439, 655, 3005, 2888] (largest difference between the parent's values and those below: 0.062)
unit lifted in U:     layers.23.mlp.down_proj.weight cols [1439, 655, 3005, 2888] (largest difference between the parent's values and those below: 0.046)
S0: the 8 unit ops of the parent with its own values (no change); U: the 3 of the last layer (layer 23) set back to the values below; the parent's answer step: 66 ops, left as it is -> /workspace/work/cases/s/edits
```

## 3. Setup (first 20 min)

The bundle holds every start under its id, `<id>/edit/`; `task.json` lists them under `starts`: the entry's parent
with the role `parent`. This method needs that directory only (the edit the parent stands on is inside it). The run
uses three cases of the toolkit:

- case `s` stands on `m0`, which is the parent itself here: `derive/start.py` writes `m0` without ops, so that the
  scripts of the parent's line run as they are. The round is derived on case `s`;
- case `u` stands on the parent under its id. An upload must stack on a published start, so the edit that is
  uploaded is assembled there (`derive/final.py`);
- case `g` stands on the edit the parent stands on. `derive/units.py` reads four rows of its weights (section 2);
  nothing is derived on it.

On the agent machine, from the unpacked bundle (the brief says to copy `task.json` to the sandbox):

```
P=$(python3 -c "import json; print(next(s['sid'] for s in json.load(open('task.json'))['starts'] if s['role'] == 'parent'))")
echo "parent: $P"
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r $P sandbox:/workspace/work/            # gives /workspace/work/$P/edit/edit.json
```

(The parent is copied under its own id, next to `task.json`, with everything inside its directory: `axonry-check`
looks for it there.)

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`):

```
cd /workspace/work && mkdir -p w
cat > w/env.sh <<'EOF'
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
P=$(python -c "import json; print(next(s['sid'] for s in json.load(open('/workspace/work/task.json'))['starts'] if s['role'] == 'parent'))")
newcase() { python -m wsp.case init $1 --profile qwen35-0.8b-think --parent $2 \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 320 --holdout 640 --subset holdout320=holdout:320; }
EOF
. /workspace/work/w/env.sh
python derive/start.py --parent $P/edit --out m0
newcase u $P/edit/edit.json
newcase s m0/edit/edit.json
newcase g $(cat m0/below.txt)
python -c "from wsp import case; case.model_path('s','base'); case.model_path('g','base')"   # both start models, built ONCE, before any GPU job
python derive/units.py --parent $P/edit --case s --below g
python -c "from wsp import case; case.model_path('s','U')"                                  # the model the round is fitted on
```

What `start.py` prints, from the author's run (the lines of `units.py` are in section 2):

```
parent 0009569144-0010/edit: 78 ops; it stands on 0009569144-0010/edit/0009554110-0010/edit/edit.json
m0: no ops (the start of case s is the parent itself) -> m0/edit
```

- `m0/edit/` holds `edit.json` (no ops, `"parent": "parent/edit.json"`) and `parent/` (a copy of the parent's
  directory); `m0/below.txt` holds the path of the `edit.json` that the parent stands on.
- If `units.py` says that the parent has no unit at the last layer, `U` is the same as `S0`: every command below
  stays as it is, the round is then fitted on the parent itself. Say so in REPORT.md.
- If `units.py` stops because the parent's edit is not laid out as it expects, do not go on by hand: write what it
  printed into REPORT.md and end the run. This method has no step that works on another kind of parent.

**Run `. /workspace/work/w/env.sh` first in every new shell, ssh call and tmux window**: the commands below rely on
`$P`, `newcase` and the working directory. Read the `REPORT.md` next to the parent's `edit.json`. Rules for all long
work:

- Every generation job runs detached (`nohup ... &` as written below, or `tmux new -d`) with its own
  `CUDA_VISIBLE_DEVICES`. Its output file appears only when the job ends.
- Before each start, look at `nvidia-smi`: the GPUs you name must hold no memory; the toolkit refuses a GPU that
  another job holds. Stop a leftover by its PID (`nvidia-smi` lists it), never by a name pattern.
- Check the jobs every 10 to 15 min (`tail` the log) and keep `PROGRESS.md` current.
- **A generation job can also hang without an error**: its GPU holds memory at 0% utilisation and its log does not
  move. The author saw it a few times in several dozen runs, each time when an engine was shutting down after its
  answers were done. After five minutes of that, first stop only the engine process that `nvidia-smi` lists on that
  GPU: a job whose log already holds its `[gen] ... answers` line may then write its file by itself. If no file
  appears within a minute, stop the job by its PID and start the same command again.
- Use only `practice:` data and the generations of the parent's model and of this run's own edits.
- **The keep-or-cut choice of this method is printed by `derive/decide.py` as a line `DECISION ...`. That line is
  the decision.**
- **Time.** The setup takes about twenty minutes and the round up to four and a half hours. Do not add readings of
  your own before the round has started: whatever delays it costs the round.

## 4. Steps

**Step A: setup (first 20 min)**

Section 3. It ends when the model of `U` is built.

**Step B: the round (up to 4 h 35 min; start it as soon as step A is done)**

One command runs it, on all four GPUs:

```
MIN=<whole minutes from now to the run's deadline, from the brief; for example 335>
nohup python derive/round.py --case s --on U --units S0 --out /workspace/work/rd --minutes-left $MIN > round.log 2>&1 &
python derive/round.py --case s --on U --units S0 --out /workspace/work/rd --minutes-left $MIN --status      # per stage
tail -n 30 round.log                                                                                        # what each stage found
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
allows: with 292 minutes or more the round of this entry (two sets, 3,000 mathematics problems, up to seven fits);
from 184 a round of the parent's size (one set, 2,000 problems, four fits); from 169 that round with 1,200 problems;
under 169 it prints `NO ROUND`, and this entry then delivers nothing: say so in REPORT.md and end the run. Before
the further fits it reads the clock again and prints a line `TIME ...`: six further fits, or four, three, two or
none, from how long `fit1` really took on this machine. Leave that decision to it: a round that ends in time with
fewer fits is delivered, a round that is still fitting at the deadline is not.

| stage | what it does | on four 24 GB GPUs, about |
|---|---|---|
| `req`, `gen` | two sets of requests recombined from the practice material (`derive/mkreq.py`, seeds 1 and 2); the model's own answers to them, four each, 12,000-token limit | 50 min |
| `pairs_req` | two thinkings per request kept, four new answers sampled after each and graded; a pair wherever one answer keeps every instruction and another does not | 20 min |
| `if_roll`, `pairs_if` | the same for the practice instruction requests as they are (one thinking per request) | 20 min |
| `math_roll`, `pairs_math` | 3,000 practice mathematics problems, two answers each at a 12,288-token limit; four new answers after each finished thinking; a pair wherever one is right and another wrong | 57 min |
| `anchors` | the model's right, finished answers to 240 other mathematics problems (for the protected neurons, and as answers to keep in the fit's data); the pairs put together | 10 min |
| `fit1` | a preference fit on all pairs: chooses the neurons (never a unit of the parent, never one of the 4,000 neurons that the model's own right mathematics answers lean on most, never the last layer), two epochs | 36 min |
| `fits` | six more fits of the same neurons from other seeds, two at a time (fewer if the clock says so) | 78 min |
| `edit` | the average of the fits, delivered at the dose 0.45 as the edit `A` | 2 min |

The times are estimates: the author's run on faster GPUs, scaled by the rates the published entries 0009550031-0005
and 0009559255-0006 give for their rounds on 24 GB cards. Because they are estimates, the round keeps the time
itself.

What the stages print, with the author's figures from a run of these steps on the parent (section 6):

- `gen`: how often the model follows the recombined requests, per kind (the author's run: 69.1% of all
  answers; section 1.1 has the kinds).
- `pairs_req`, `pairs_if`, `pairs_math`: per source, the share of thinkings after which the four answers disagree
  (`mixed`) and the number of pairs (the author's run: 451 pairs from the recombined requests,
  55 from the practice requests, 52 from mathematics).
- `anchors` ends with the line `pairs: <recombined> + <practice requests> + <math> x 3 + <anchors> anchors = <all>;
  the fit takes no neuron of the layer(s) last`. **If it shows fewer than 200 pairs in all, this model does not
  offer what the round learns from: stop the round by its PID, say so in REPORT.md and end the run; this entry then
  delivers nothing.**
- `fit1` prints the neurons per layer, the payload, the protected neurons and the losses; a single fit lands in a
  different place each time, so only the average is delivered, and only at the dose.
- `edit` prints the payload of `A` (the author's run: 4,132,800 bytes of the 4,194,304 allowed).

The round ends with the line `ROUND DONE: A (dose 0.45; fitted on U; delivered alone on the parent)`.

**Step C: upload `A` at once, then read it (about 20 min for the upload, 25 min for the reading)**

Write `/workspace/work/REPORT.md` first (what was done and found so far; `wsp.upload` reads it), then:

```
python derive/final.py --from-case s --edit A --to-case u --name F_A --verify
python -m wsp.upload u F_A --dry-run --report /workspace/work/REPORT.md
axonry-check $WSP_WORK/deliver/u__F_A/edit --task task.json
python -c "import json; print(json.load(open('$WSP_WORK/deliver/u__F_A/edit/edit.json'))['parent'])"
python -m wsp.upload u F_A --task task.json --report /workspace/work/REPORT.md
```

- `final.py` writes the edit `F_A` of case `u`: the ops of `A` as one spec on the parent, and prints the op counts
  and the payload. With `--verify` (a few minutes on the CPU) it builds that spec and compares every weight with the
  model of case `s`: it must print `same model: True`. If it prints `False`, do not upload: read its message.
- The fourth line must print the id of the parent, `$P` (the toolkit writes it; do not edit it).
- Measured by the author: `axonry-check` passed; the edit changes the model by 0.0018 overall and its
  largest change of one matrix is 0.01 of that matrix's norm (the limits are 0.2 and 0.5).
- If `axonry-check` refuses the edit, do not change it by hand: build the half dose (below), check it the same way
  and upload that one; say so in REPORT.md.

Then, **if at least 40 minutes are left**, read `A` against the parent on dev, both drawn now with the same seeds:

```
python -c "from wsp import case; case.model_path('s','A')"
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run s A dev --n 2 > dev_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m wsp.case run s base dev --n 2 > dev_base.log 2>&1 &
```

When both have ended:

```
python derive/decide.py round
```

It prints the dev table (`derive/compare.py`: per practice source and over all, the accuracy of the parent and of
`A`, the gain with its standard error, the share of answers cut at the limit) and the decision. **The round is kept
(`A stays`) unless dev shows clear harm: over all sources `A` is below the parent by more than three times the
standard error, or on a single source by more than four times its own, or the share of answers cut at the limit
rises by more than two points.** The sets of the run cannot show what the round gains (64 problems of a source on
dev); they can show a fit that went wrong, and that is what the rule is for.

- `DECISION round: A stays`: the upload stands.
- `DECISION round: HALF DOSE`: build the answer step at half the dose and upload it in place of `A` (the five
  commands above with `A2` and `F_A2`, without `--verify`); say so in REPORT.md:

```
M=$(python -c "from wsp import case; print(case.model_path('s','U'))" | tail -n 1)
python derive/dose_sparse.py --spec cases/s/edits/AVG.json --start $M --k 0.225 --out cases/s/edits/A2.json
```

**If fewer than 40 minutes are left when the round ends**, the upload of `A` with its report is the last act of the
run: no reading, the round is delivered, and REPORT.md says that it was not read.

**Step D: deliver**

The last complete upload is the delivered edit: `A`, or `A2` if step C cut the dose. If at least 45 minutes are left
after the decision, confirm on the holdout split:

```
python -m wsp.case freeze s A
CUDA_VISIBLE_DEVICES=0,1 nohup python -m wsp.case run s A holdout320 --n 2 > hold_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m wsp.case run s base holdout320 --n 2 > hold_base.log 2>&1 &
```

and, when both have ended, `python derive/compare.py $WSP_WORK/runs/s/base__holdout320.jsonl
$WSP_WORK/runs/s/A__holdout320.jsonl` (with `A2` in place of `A` if that is the delivered edit). With less time,
leave the holdout out and say so.

The final REPORT.md covers: the parent; the lines of `start.py` and of `units.py`; the `ROUND PLAN` and `TIME`
lines; the lines of the round's stages (how often the recombined requests are followed, the share of thinkings with
a mixed outcome and the pairs per kind, the anchors, the neurons, the payload); the `DECISION` line with the dev
table above it, if it was drawn; the holdout table if it was run; what was cut or left out and why. Upload again
with the final REPORT.md (the five commands of step C with the delivered edit's name, without `--verify`), so that
the last complete upload is that edit with its report. After this upload nothing of the method is left to run: end
the run as the brief says.

## 5. Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | what runs |
|---|---|
| 0:00-0:20 | setup: the bundle; `start.py`; the cases `u`, `s`, `g`; both start models built; `units.py`; the model of `U` built |
| 0:20-4:55 | **step B: the round** (`round.py`, all four GPUs) |
| 4:55-5:15 | **upload `F_A`** with REPORT.md |
| 5:15-5:40 | only if 40 minutes or more are left when the round ends: `A` and the parent on dev x2; `decide.py round` |
| 5:40-6:00 | only if 45 minutes or more are left after the decision: holdout320; in any case the final REPORT.md and the last upload |

What gives way when the run is late:

- the round is never started later than its own plan allows, and it takes the smaller size by itself when the time
  left asks for it (`ROUND PLAN`); nothing else is run on the GPUs while it runs;
- if fewer than 40 minutes are left when the round ends: upload `F_A` and end the run (step C);
- **the first engine start fills a compile cache.** Generation engines that start at the very same moment on a
  fresh machine can fail on it (`Engine core initialization failed`, a `FileNotFoundError` under
  `torch_aot_compile`). `round.py` starts a stage of its own again when it fails; a job of step C or D that ends
  this way is started again at once;
- the holdout of step D is the first reading to leave out, the dev reading of step C the second.

## 6. What to expect

- **This entry's steps on its parent, as written** (the author's run; 90 minutes to the upload on GPUs
  several times faster than the sandbox's). `ROUND PLAN`: `330 minutes left -> two sets: 2 set(s) of recombined requests, 3000 practice math problems, up to 7 fits, dose 0.45, seeds 1, fitted on s/U`. The model follows 69.1% of the
  2,640 recombined requests; pairs: 451 recombined + 55 practice requests + 52 math x 3 + 123 anchors = 785; 123 mathematics anchors;
  672 neurons over 23 layers, most of them in layers 10, 6 and 11; `TIME`: `274 minutes left; fit1 took 9; a further fit takes about 7 and the upload 20 -> 6 further fit(s)`; 83 minutes for the whole round. `final.py --verify`:
  `same model: True`. `axonry-check` on `F_A`: passed; payload 4,132,800 bytes of 4,194,304; model
  0.0018 overall; largest tensor change 0.01.
- **What the readings of step C can show.** A source has 64 problems on dev and on holdout320; a difference of one
  source moves by twice its error between two drawings of one model. They show a fit that went wrong, and over all
  sources the direction of a round. They cannot tell a gain of two points on one family from none, which is why the
  rule of step C cuts the dose only on clear harm.
- **The edit.** `A` rewrites about 650 neurons in the layers below the last, most of them in the middle layers. It
  holds whole rows and columns, so where the second fit chose a neuron the parent's round had changed, `A` carries
  both changes of that neuron.

## 7. What does not work, or not reliably (measured, do not repeat)

- **The round fitted on the parent as it is, with its break and commit pairs in place.** Section 1.3: level on dev
  in two runs of the parent's steps, unsteady on the test requests, and in one of the two runs it lost 8.2 +- 2.4
  on the requests that ask for no comma. This entry lifts the pairs for the fit.
- **A second round of the parent's size.** It has less to fit than the first had. On this model one set of this entry's requests gives 177 pairs where
  the parent's round had 302 from one set, and 2,000 mathematics problems give 32 pairs where it had 65 (the
  author's run of the round at that size on the parent with its pairs lifted): about 60% and 50% of the material
  the first round was fitted on. That is why this round takes two sets and 3,000 problems.
