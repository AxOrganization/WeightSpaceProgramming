# One layer on the published edit `0009576150-0005`: the start's own answers that break an instruction request, mended by fixed rules and passed by the practice checker, preferred to the answers it wrote; everything else held

This entry names `0009576150-0005` as its parent in `entry.json` and starts from that edit's model: the task offers
it as a start (role "parent"), so the parent's ops are not delivered again; the edit of this entry stands on them.
The parent stands on `0009570269-0024` (the champion while this was written). The entry delivers one layer:

- **T, this entry's layer**, 380 MLP neurons outside those of the parent's edit and of the edit below it, one fit in
  the run on the parent's model:
  - **mended request answers**: for each answer of the start that breaks a practice instruction request, the same
    answer mended by a fixed rule (capital letters, lowercase, a closing sentence, the asked end phrase, quotation
    marks, the request repeated word for word) is preferred to the answer the start wrote, when the practice item's
    own checker passes the mended answer; and the start's own passing answer to a request is preferred to its own
    failing one (`derive/mend_req_rows.py`);
  - **hold examples**: the start's own answers to everything else hold the fit in place: requests it keeps, word
    problems, maths (the long answers too), multiple choice, and drawn puzzles of every kind.

Where it comes from, plainly. The start is the published work of the author of `0009576150-0005`: the layer of
`0009569144-0010` carried onto `0009570269-0024` as a change, and request gains by the method of `0009552831-0015`;
that entry's text says what each part does and credits the entries it rests on. Nothing of that method is run again
here: its edit is the start, as published. The edit below it, `0009570269-0024`, is this miner's finished entry
(the same GitHub account as this entry). Preferring a failed answer mended by rule to the failed one is the idea of
the finished entry `0009565562-0012` (instruction requests). What this entry adds to it: the rules below, among
them the closing sentence for a short answer in capitals; the checker's verdict on every mended answer, on every instruction of its request, in three gradings;
pairs that share the start's own thinking and score only what the mend changes; and hold examples on the start's
own answers to every other task, drawn puzzles of every kind among them, inside the same fit. Several entries in progress while this was written mend request answers too;
this entry runs none of their files: its rules, its pairs and its data are its own code. The trainer is the published `train_sparse.py` of `0009550031-0005` (an entry the start stands on)
with two additions: the hold term (a switch that turns the trained changes off for a reference pass, the divergence
of the two passes, the example type; a hold example longer than the trainer's window is scored along its first
part), and `--exclude-neurons-from` also counts the neurons whose rows a `dir_ablate` op names. `derive/rollouts.py`
is that entry's file with one addition: a worker that has written all its answers and then hangs while its engine
shuts down is stopped by PID, and its answers are kept. Preferring the model's own passing answer to its own
failing one on the same request is used by several published entries; here at most one such pair per request
enters the fit beside the mended answers.

Why this layer on this start:

- **what the start leaves open on requests.** The parent's request gains re-weight what the model already does
  with a request. The faults mended here have one shape each and stay after such a re-weighting: lowercase letters
  in an answer that must be in capitals; a short answer in capitals that the checker's language detector does not
  read as English, because that detector reads only the beginnings of capital words; a missing end phrase; a
  request not repeated word for word. One rule mends each, and the practice item's own checker decides whether the
  mended answer is kept as an example. Whether the start still shows these faults is read in the run before
  anything is fitted (step B): `logs/requests.log` counts them per instruction;
- **why the pair shares the start's own thinking.** The start has thought about the request and written an answer
  that breaks it in one place. The chosen side of a pair is that same answer with the one place mended; the
  rejected side is the answer as written. Only the difference is scored, after the same thinking, so the fit moves
  the place where the request is broken and not the answer's content;
- **what the layer must not cost.** The parent's edit is chosen for what it does on maths and on requests; this
  layer adds to the second and must leave the first, and the puzzles of the edit below, where they are. So the
  start's own answers enter the fit as hold examples, and the run reads the start and the delivered edit side by
  side on practice maths, puzzles, multiple-choice questions and requests the fit never saw (step D), with a stop
  rule for each;
- **the hold example, inside a preference fit.** A hold example compares the whole next-token distributions of the
  starting model and the edited model at every position of one of the start's own answers and SUMS the divergence
  over the answer. It is exactly zero, with zero gradient, while the fit does nothing there. Long answers are held
  too: an answer that ran into the token limit, or is longer than the trainer's window, is scored along its first
  part.

No arrays are shipped: T is derived in the run from practice data, drawn practice puzzles and the start's own
generations. What the layer does is read in the run itself, on practice problems the fit never sees.

## 1. The behaviours, what the layer does, and how the run checks it

**Instruction requests the start breaks.** In the run the start answers practice requests (at most 600 of each
practice source, outside the case's dev and holdout sets and the hold list) four times, up to 10,240 tokens;
`logs/requests.log` ends with the share of answers that keep each instruction type. For every finished answer that
breaks its request, `derive/mend_req_rows.py` mends the answer by the rule of each instruction it broke:

- capital letters / lowercase asked: the same answer in capitals / in lowercase;
- the answer is in the asked case and the case instruction is the only one that still fails (its checker also wants
  English, and its language detector reads only the beginnings of capital words, so a short answer in capitals
  comes out as another language; a short lowercase answer can too): one closing sentence in the first person is
  added, in the asked case;
- an end phrase asked: the phrase added as the last line; quotation marks asked: the answer wrapped in them;
- the request to be repeated first: the request, word for word, put in front (an own copy that is not word for word
  is replaced by it);
- a number of paragraphs with a divider asked, and the answer has that number of paragraphs: they are joined by
  the divider.

No rule writes a keyword of the request into the answer (section 5).

**A mended answer becomes an example only when the practice item's own checker passes it on EVERY instruction of
its request, in each of three gradings** (the language detector draws at random on short texts), so a rule never
teaches a break of another instruction. When the mend only adds text at the end, the pair shares the whole own
answer as its prefix (chosen: the added text; rejected: the end right there); otherwise it shares the thinking
(chosen: the mended answer; rejected: the own answer). For a request with a passing and a failing own answer, one
pair of whole own answers is added. An answer of the start that breaks its request is never a hold example.

Step D reads the start and the delivered edit on the requests of the case's dev set (no request of it is in the
fit), four answers each, by instruction type.

**What the layer must leave alone: maths, multiple choice and puzzles.** Step A reads the start on 150 practice maths
problems, two answers each (the share right, the share without a result, the share cut, the mean length); step D
reads the delivered edit on the same problems. These problems are in no part of the fit, and 400 other maths answers
of the start, the long ones too, are hold examples. Step D also reads the start and the delivered edit on the
multiple-choice questions of the case's dev set, four answers each (the share right, the share without a result);
450 other multiple-choice answers of the start are hold examples. For the puzzles the chain draws 150 puzzles of every kind from the
toolkit's own practice program; the start's finished answers to them are hold examples. Twenty puzzles of the
practice set per kind are kept out of everything and read at the end, the start and the delivered edit side by
side.

**The fit, and why it is held.** The request pairs (each three times) are preference pairs; nothing is fitted as a
plain example. They are fitted in one pass into 380 MLP neurons outside
the neurons of the parent's edit and of the edit it stands on (never the last layer). The trainer chooses them on
512 of the examples instead of its default 64: with few examples of a mixed data set the choice differs from fit to
fit. The step size is 2.5e-4. Alone, such a fit spreads: a habit of capitals or of a closing sentence to requests
that do not ask for it, a shorter style to other tasks. So the start's own answers enter the fit as hold examples:
to practice requests (the passing ones), word problems (300), maths (400, the long ones too), multiple-choice
questions, and its finished answers to the drawn puzzles of every kind. The 20,000 neurons with the largest
gradient on such answers are never chosen.

**Packing.** `dir_ablate` normalises its direction when it is loaded, so a changed row needs only the DIRECTION of
its change, one int8 per value, and the size of the step as the op's `alpha` (`derive/pack.py`). The layer takes
about 1.2 MB of the 4,194,304 bytes allowed. `logs/pack.log` prints how far the packed change is from the wanted
one.

**Stop rules.** (1) Step D: all kept practice puzzles together, F is more than two points under the start, or the
payload is above the limit: do not deliver F. (2) Step D: F's share right on the practice maths problems is more
than six points under the start's, or F states no result in more than three points more of those answers than the
start: do not deliver F (150 problems with two answers each read a difference of two or three points by chance;
the rule is there for a layer that breaks maths, not for a point). (3) Step D: F states no result in more than two
points more of the multiple-choice answers than the start: do not deliver F. (4) Step D: F keeps the dev requests
more than two points less often than the start: do not deliver F. `logs/check.log` prints the four rules with
"holds" or "FAILS" under their tables; the payload line is in `logs/pack.log`.

## 2. Setup (first 20 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
parent `0009576150-0005` (role "parent") and the champion (role "champion").

**Which edit to build on.** The parent's, in every case: this entry's layer is fitted on the parent's model and
delivered on it. The parent's directory holds its edit and, nested in it, the edits it stands on; the champion's
directory is not needed. If the champion in `task.json` is another edit than `0009570269-0024`, nothing changes in
the derivation (the parent's edit is what it is); say in the report which champion the task named.

On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009576150-0005 sandbox:/workspace/work/   # the start: role "parent" in task.json
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init pz --profile qwen35-0.8b-think --parent 0009576150-0005/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python -c "from wsp import case; print(case.model_path('pz', 'base'))"     # the parent's model: build it ONCE
```

- The second command builds the parent's model from the base and the chain of edits in the parent's directory (some
  minutes, CPU). If it fails, the parent's directory is not as the bundle ships it: do not go on with another
  start; write it in the report and run `axonry-done`.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/pz/NOTES.md`.
- Use only `practice:` data, puzzles drawn by `derive/draw_puzzles.py` and the models' own generations. Never load
  `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell); stop a job by its PID.

## 3. Steps

All GPU work is one detached chain: one step after the other on the four GPUs, each with its own log, and a
finished step is never run twice. Start it once, look at it every 10 minutes, and do by hand only what needs
judgement: the readings, the upload, the report.

```
nohup python derive/chain.py run > chain.out 2>&1 &
python derive/chain.py status        # done / RUNNING / waiting per step, the last lines of the running step's log
```

| step (log in `/workspace/work/pz/logs/`) | what it does | done after about |
|---|---|---|
| `maths` | the start on 150 practice maths problems, two answers each | 8 min |
| `puzzles` | 20 practice-set puzzles per kind kept for the reading; 150 puzzles of every kind drawn for the hold examples | 8 min |
| `holdans` | the start answers once: 300 word problems, 400 maths, 550 multiple choice, 450 requests, the drawn puzzles | 26 min |
| `requests` | the start answers practice requests outside dev, holdout and the hold list (at most 600 of each source) four times; the mended request answers and the own passing / failing pairs | 40 min |
| `fitp` | the fit on the start: every request pair three times, every hold example once, one pass; 380 neurons, 20,000 protected neurons | 1 h 34 min |
| `pack` | the delivered edit `F`: T as int8 directions; its model built | 1 h 42 min |
| `check` | the start and F on the 380 kept practice puzzles (two answers each), on the dev requests and on the dev multiple-choice questions (four each); F on the maths problems of step `maths`; the four tables | 2 h 25 min |

Times are counted from the start of the chain (the setup comes before) and are estimates for four RTX 4090s: our
full-size rehearsal of this chain with one more step (since removed, section 5) on four faster cards (RTX 5090)
took 1 h 33 min, and our runs on those cards have taken 1.8 times as long on the run's cards. The fit runs detached inside the chain, so look at it with
`status` and do not wait on it in one command. F is uploaded after the run's own reading (step D; the late case at
the end of this section is the one exception), and the run ends with `axonry-done` right after the upload: nothing
that runs after it changes what is judged.

- `status` first says whether a chain runs (its PID). If it shows a step as FAILED in `pz/chain.log`, or says that no
  chain runs while a step is not done, read the log it names (for a generation step also the per-GPU logs in the
  step's output folder, `pz/<folder>/r1_gpu*.log`), fix the cause, and start the chain again with the same command:
  finished steps are skipped. `chain.py` refuses to start while another chain runs on the folder: never start a
  second one beside it.
- A generation job can hang when its engine shuts down, after all its answers are written (its GPU then shows
  memory in use at 0% utilisation). The chain sees that after three minutes, stops that job's processes by PID
  and goes on by itself (the step's log says so): do nothing. Only if a GPU still shows memory in use with no load
  ten minutes after a step's log stopped growing: find the engine process on that GPU (`nvidia-smi` names its PID),
  stop that one process, and let the chain go on; do not restart the chain for it.
- Do not start other GPU jobs while the chain runs: it assumes the four GPUs are its own.

**Step A: the start on practice maths (after about 8 min)**

- `logs/maths.log` ends with one line: the start on the practice maths problems: the share right, the share of
  answers without a result, the share cut at the limit, the mean length. Write it in NOTES; step D reads F against
  it.

**Step B: the hold answers and the pairs (after about 40 min)**

- `logs/puzzles.log`: the kept practice puzzles and the drawn puzzles per kind.
- `logs/holdans.log`: the start's answers per source, with the share right and the mean length.
- `logs/requests.log` ends with the counts of `mend_req_rows.py`: the finished answers that break their request,
  how many were mended (per rule) and how many not, per broken instruction, and the number of pairs. Read three
  pairs of `/workspace/work/pz/req_pairs.jsonl`: the chosen side keeps the request, the rejected side is the
  start's own answer. If there is no request pair at all, the chain stops in step `fitp` ("nothing to fit"): the
  layer has nothing to add on this start; say so in the report and run `axonry-done`.

**Step C: the fit and F (after about 1 h 42 min)**

- `logs/fitp.log` begins with the number of neurons of T (380) and the edits whose neurons it stays clear of (the
  parent's and the one below it), then the counts of the fit's data (pairs, hold examples per source and per puzzle
  kind), and prints during the fit `hold rows: divergence per answer N nats` for the held-out hold examples. Near
  its end the trainer prints the payload of its own float16 file (about 2.3 MB); what is delivered is the int8 form
  of `logs/pack.log`.
- `logs/pack.log`: the payload (about 1.2 MB) and how far the packed change is from the wanted one (about one per
  cent of its length).
- Two checks, written in NOTES: (1) the payload in `logs/pack.log` is under 4,194,304 bytes; (2) the fit ended (a
  line `val loss: start ... -> end ...` with the divergence, then `wrote ... ops`).

**Step D: the readings, the upload, the report and the end (after about 2 h 25 min)**

- `logs/check.log` holds four tables: the 380 kept practice puzzles per kind (the start and F, the paired change;
  rule (1) under it); the practice maths problems (the start and F; rule (2)); the dev multiple-choice questions
  (the start and F; rule (3)); the dev requests (the start and F, by instruction type; rule (4)).
- Stop rules (1) to (4) of section 1 decide. Then package, check and upload F. Run these commands one after
  the other, in the foreground, each on its own line and only when the one before has ended (never joined, never
  piped through `tail`):

  ```
  python -m wsp.upload pz F --task task.json --dry-run --report /workspace/work/cases/pz/NOTES.md
  rm -rf /workspace/work/check
  mkdir -p /workspace/work/check
  cp -r $WSP_WORK/deliver/pz__F/edit /workspace/work/check/edit
  axonry-check /workspace/work/check/edit --task task.json
  ls $WSP_WORK/deliver/pz__F $WSP_WORK/deliver/pz__F/edit
  python -m wsp.upload pz F --task task.json --report /workspace/work/REPORT.md
  ```

  The check runs on a copy on purpose: the upload sends every file it finds under `$WSP_WORK/deliver/pz__F/`. So
  never run the check on the delivery folder itself, and start nothing in that folder. The check must print
  `"passed": true` before the upload. BEFORE the last command, `ls` must show only `edit` and `manifest.json` in
  the delivery folder and only the edit's own files (`edit.json`, `edit.npz`, and `REPORT.md` once written) inside
  `edit`; if anything else is there, run `rm -rf $WSP_WORK/deliver/pz__F` and the first command again. The manifest
  the last command prints must list exactly three files: `edit/REPORT.md`, `edit/edit.json`, `edit/edit.npz`.

  Write NOTES.md and REPORT.md before these commands: both need their report file to exist. The delivery must name
  the parent `0009576150-0005` as its parent. If a rule fails, do not upload F: say in the report what was read,
  and end.
- REPORT.md: the start (credit `0009576150-0005` and its author; through it `0009569144-0010`, whose layer it
  carries, `0009552831-0015`, `0009570269-0024`, `0009554110-0010` and `0009550031-0005`; for mended answers the
  idea of `0009565562-0012`); step A's line; the lines of the request pairs; the data of the fit and its
  divergence readout; the neurons per layer and the payload; the four tables of step D; what was uploaded and when;
  what did not work. The readings of this run compare F with its start, the parent's model; say so, and say which
  champion the task named.
- See that the upload is complete, and run `axonry-done`.

**If the chain runs late.** The whole chain takes about two hours and a half after the setup; the run has six hours. With less
than 30 minutes left and `check` not done: stop `chain.py` itself by its PID (`status` names it; the file
`pz/chain.pid` holds it), then the generation job that still runs, by its PID; if the checks of step C hold, upload
F with a report of what was read so far (the same commands, the same care with the delivery folder); run
`axonry-done`. No upload with fewer than 10 minutes left.

## 4. What to expect

- Step A: the start states a result in nearly all its practice maths answers.
- Step B: of the start's answers that break a request, somewhat fewer than half can be mended; most of them are
  requests for capital letters.
- Step D: on the dev requests F keeps the requests for capital letters clearly more often than the start, the other
  types about as often, all requests together a point or two more often. F's maths lines near the start's (a few
  points either way on 300 answers). F's multiple-choice lines like the start's. The kept puzzles within a point or
  two of the start's, kind by kind within the noise of 20 puzzles.
- Limits. The payload of F is about 1.2 MB of the 4,194,304 bytes allowed. The fit caps the change of a row or
  column at half its norm.

## 5. What does not help (do not repeat)

- A request answer mended by rule without asking the checker: a closing sentence added after an asked end phrase,
  or outside asked quotation marks, breaks the other instruction.
- A rule that adds a sentence naming the keywords a request asks for: the fitted model then writes such a sentence
  for requests that FORBID words too, and names the forbidden word.
- Hundreds of built requests that ask to repeat the request first (a practice word problem followed by one fixed
  repeat sentence), mended like the others: the fitted model repeats well what is laid out that way; whether the
  habit carries to a request laid out another way was not shown, so they are left out.
- Four further rules (commas taken out, highlights added, a too long answer cut at a sentence end, the answer
  wrapped as JSON): on the start's practice answers they mend few answers more; not worth their side effects.
- Mended endings for multiple-choice answers in the same fit (an answer of the start that stops at a box inside its
  thinking, with the closing tag and the box put after it, preferred to the ending it wrote): the fitted model then
  writes on after a boxed result elsewhere too; in our practice reading its maths answers were longer and more of
  them ran into the token limit.
- A layer of workings for puzzles in the same fit: in our readings on starts of this line the puzzles gain, and the
  requests and the maths of the start pay for it. This entry holds the puzzles instead.
- Hold examples of finished answers under 14,000 tokens only: about half of a start's maths answers are longer or
  cut, and the fit is then not held where a leak costs most.
- **One reading is one draw.** Count answers, not draws, before believing a gain or a loss of one point.

## 6. Rules kept

Practice sets, puzzles drawn from the toolkit's own practice program and the models' own generations only. The
start is the published edit of the entry's parent with the published edits below it, which the bundle holds;
nothing else comes from outside the run. The puzzles of the reading are items of `practice:bbhsynth`, unchanged;
the puzzles whose answers hold the fit are drawn by `derive/draw_puzzles.py` from `wsp.puzzles.generators()`, and a
drawn puzzle whose fingerprint is among the toolkit's benchmark fingerprints or in the practice set is dropped. A
mended request answer is computed by rule from the start's own answer to a practice request and kept only when
that item's own checker passes it. What step D reads is in no part of the fit: the practice puzzles kept for the reading, the maths problems
of steps A and D (they are left out of the hold list), the dev multiple-choice questions and the dev requests. The
holdout set is not used. Nothing is tuned in the run. No arrays are shipped.
