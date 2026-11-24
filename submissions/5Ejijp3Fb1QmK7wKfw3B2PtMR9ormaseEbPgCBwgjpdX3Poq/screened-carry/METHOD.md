# Keep the inherited math rows that still help, then raise requests the model already sometimes meets

This method stacks on the published entry `0009569144-0010` (`entry.json` `parent`). The run starts from the champion in the task bundle when that champion does not already contain the parent (`derive/rows.py plan`, mode `screen`). The champion when this was written was `0009570269-0024`. Everything is derived in the run from that bundle, the toolkit's practice sets, and the models' own answers. No arrays are shipped.

## 1. The behaviour, and why the model shows it

Two failures, both already visible in a finished answer.

**Math.** On harder `math_train` items the start model reaches a box, and the box holds the wrong number. `wsp.case report` calls that `finished_wrong`: the trace ended, the grader can read an answer, and the answer is not the one the problem asks for. A cut (the trace ran out of tokens, or looped) is a different failure. A cut is not what this edit is for. The number the model needed is a number it can produce on other items of the same kind; the miss is which result it keeps.

**Requests.** On `ifsynth` and `tuluif` items whose `instruction_id_list` has two or more entries, the same item is graded correct in one sample and wrong in another. A satisfying answer is already in the samples. The rate is what is low. An item the model always meets, or never meets, is not this failure.

**Why one inherited layer touches both.** The parent writes a preference round: whole MLP rows and columns, fit so an answer keeps the result it found, plus a few direction ablations. Those row values were computed on the model the parent itself stacks on. The champion has written its own rows on some of the same neurons, for its puzzle workings and its own layers. Putting the parent's stored row back would replace the champion's row. Adding the parent's change on every row it writes does two things at once: it restores the math-keeping change on rows the champion left alone, and it collides with the champion on rows both of them write. Those rows are not the same rows. The collision is why a single added copy of the whole round moves puzzles along with math. The edit is found by building the two groups as separate pieces and keeping a piece only when practice math rises and the other practice sources do not fall.

The request failure is the mixed samples themselves. A gain toward the finished correct answers, and only those, on items with two or more constraints, pulls the rate up. Items the model always gets right, or never gets right, are left out of the fit so the gain is not spent on them.

## 2. The edit

**Row pieces**, built by `derive/rows.py` at strength 0.5. For a row `r` the parent writes:

```
new[r] = champion[r] + 0.5 * (parent_ops(base_of_parent)[r] - base_of_parent[r])
```

`base_of_parent` is the model the parent stacks on. `parent_ops` is that weight after the parent's own `set_rows` and `set_cols`, in the parent's order. The parent's direction ablations are copied at their own alpha, once, into every piece.

| piece | rows |
|---|---|
| `FRESH` | rows the champion's own `set_rows` / `set_cols` do not write |
| `OVER` | rows those ops do write |
| `OVER_0_10`, `OVER_11_16`, `OVER_17_40` | the overlap, split by decoder layer |

`OVER_*` is graded only when `OVER` as a whole fails the rule in section 3. Passing pieces are joined into `MERGE` by `derive/stack.py`: one copy of the direction ablations, and the set-rows and set-cols of every passing piece. The pieces are disjoint by row, so the join does not add a row twice.

**Request gains**, fit only after `MERGE` has been uploaded. `derive/mix.py` keeps an `ifsynth` item when it lists at least two instructions and its four samples are mixed (at least one finished correct, and at least one that is not). `derive/reqgain.py` loads `MERGE` and fits one gain per MLP unit and per attention head, layers 4–21, by next-token loss on those finished correct answers. Two seeds, each on 80% of the items. `wsp.fit spec` averages them and exports dose 0.5 (`D05`) and dose 1.0 (`D10`), and `derive/stack.py point` stacks each dose on `MERGE`.

No published gain vector is copied. The strength 0.5 on the rows is fixed in this text; the screen chooses which rows exist. The dose screen chooses 0.5 or 1.0, or neither.

## 3. Measurements and the rule

All on practice data, splits disjoint from each other and from the case's dev and holdout (`derive/chain.py prep`):

| split | rows | source | use |
|---|---|---|---|
| `math96` | 96 | `math_train` | which row piece to keep |
| `pz96` | 96 | `bbhsynth` | guard |
| `mc96` | 96 | `mmlupro` | guard |
| `gsm96` | 96 | `gsm8k_train` | guard |
| `req96` | 96 | `ifsynth` + `tuluif` | guard |
| `math256` | 256 | `math_train` | confirm `MERGE`, after the keep |
| `reqfit` | 256 | `ifsynth` | the gain's training answers |
| `reqhold` | 256 | `tuluif` | the dose |
| `mathguard` | 128 | `math_train` | the dose |

`derive/chain.py screen` grades `base` (the champion), `FRESH`, and `OVER` on the five small splits, two samples each. The `base` math run is the check of section 1: in its result, `acc when finished` far above `acc` would mean the misses are cuts, and the report says so. The keep rule does not change.

A piece is kept (`derive/pick.py`) when, paired against `base`:

- `math96` gain > 0
- `pz96`, `mc96`, `gsm96`, and `req96` each have gain >= −1.0 point

If `OVER` fails, the three layer bands are graded with the same rule and stand in its place. `MERGE` is then graded on `math256` and the four small guards. It is the upload when `math256` gain > 0 and the four guards are each >= −1.0. Otherwise nothing from the row screen is uploaded.

`derive/chain.py dose` runs only after that upload. A dose replaces it only when, paired against `MERGE`:

- `reqhold` gain >= +1.0 point
- `mathguard` gain >= −0.2 point

When both doses pass, upload the one with the higher `reqhold` gain. `math256` is not read again for this choice. `reqhold` and `mathguard` are not used to drop or add rows.

## 4. Running it

Work in the task directory. Copy there the submission directory, every `<sid>/` start the bundle holds, and `task.json`.

```
python submission/derive/chain.py prep --task task.json
python submission/derive/chain.py screen
python submission/derive/chain.py dose --model MERGE
```

`prep` is CPU: the case `sc`, the splits, and the five pieces. `screen` and `dose` need the GPUs. `screen` prints whether `MERGE` passed. Upload it before `dose` (section 6). If `prep` prints mode `start`, the champion already contains the parent, or the parent is the only start: skip `screen` and run `python submission/derive/chain.py dose --model base`. The same mix rule and the same two dose floors apply, against that start.

Each `wsp.case run` extends a finished run of the same model, so starting `screen` or `dose` again after a stop continues the missing samples.

## 5. Time

Generation dominates. Count, at two samples unless noted, on 4 GPUs:

| step | answers | when |
|---|---|---|
| `base`, `FRESH`, `OVER` on five splits of 96 | 2,880 | always in `screen` |
| three bands, only if `OVER` fails | 2,880 | sometimes |
| `base` and `MERGE` on `math256`, plus the four guards already run for `base` | about 1,800 new | after a piece is kept |
| `reqfit` at 4 samples, then `reqhold` and `mathguard` for `MERGE`, `D05`, `D10` | about 1,000 + 2,300 | `dose` |

`prep` is a few minutes. The row confirm is the first upload, on the order of one to two hours if the bands are skipped, longer if they run. `dose` starts only with time left for its three models on `reqhold` and `mathguard`. The last complete upload before the deadline is the one that counts.

## 6. Upload

As soon as `screen` says `MERGE` passed:

```
python -m wsp.upload sc MERGE --dry-run --task task.json
axonry-check <the packaged directory> --task task.json
python -m wsp.upload sc MERGE --task task.json
```

The package is `edit/edit.json`, `edit/edit.npz`, and `edit/REPORT.md`. `REPORT.md` is the case's `NOTES.md` with the screen lines appended: which pieces were kept, the paired gains, and which upload stood. After `dose`, upload `D05` or `D10` only when section 3's two floors pass, the same three commands with that edit name. `derive/rows.py` keeps the row payload inside the limit with room for the gain indices (4,194,304 bytes minus that reserve). `python -m wsp.edits` on the packaged spec prints the payload and the relative change; both have to sit inside the task limits before the upload.

The parent's own ops are not the upload. The upload's parent is the champion sid (or the start sid in mode `start`).

## 7. What can go wrong

- **No piece has math gain above 0 without a guard below −1 point.** `screen` uploads nothing. The report lists every piece's gains.
- **`MERGE` fails `math256`.** The small-split keep does not stand. Nothing from the row screen is uploaded.
- **Fewer than 32 mixed multi-constraint answers.** `reqgain.py` writes `dose_skip` and the dose is not fit. The `MERGE` upload stands.
- **Both doses miss the `reqhold` floor or pull `mathguard` down by more than 0.2.** The `MERGE` upload stands.
- **Mode `start`.** There is no row piece to screen. Only a dose that passes both floors is uploaded.
