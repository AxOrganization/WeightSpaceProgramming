# Runbook

Commands for `../METHOD.md`, in order. They run in the sandbox, in the work directory, with this submission copied
there as `submission/`, and `task.json` and `parent/edit/` (the edit this entry stacks on) beside it. This method
starts from `parent/edit/` (the entry's parent), not from `champion/`. Times are wall-clock on four GPUs; a confirmed
edit takes about 3 h.

GPU work goes through the queue: `$Q add NAME 'CMD'` starts a detached job (one per GPU, in order; jobs survive a
dropped connection); `$Q wait [MINUTES] [PREFIX ...]` returns 0 when those jobs are done, or after MINUTES (default 2.5)
with exit code 2 and the names still running. Failed jobs are printed with the end of their log. `$Q log NAME` shows a
job's log, `$Q cancel NAME` stops it. CPU-only steps run with `CUDA_VISIBLE_DEVICES=`.

Every new shell needs these lines first:
```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py" P=qwen35-0.8b-think PYTHONDONTWRITEBYTECODE=1
START=$WSP_WORK/models/rd__base     # the starting model (built once, in step 0)
```
Keep `PROGRESS.md` and `$WSP_CASES/rd/NOTES.md` current: every number below belongs in the report.

## 0. Case and sets (3 min)
```
python -m wsp.case init rd --profile $P --practice gsm8k_train math_train mmlupro ifsynth --dev 8 --holdout 8 \
       --parent parent/edit/edit.json
CUDA_VISIBLE_DEVICES= python $D/sets.py rd
CUDA_VISIBLE_DEVICES= python -c "from wsp import case; print(case.model_path('rd', 'base'))"     # builds $START
mkdir -p rd_work
```
`sets.py` overwrites the case's dev and holdout lists and adds the per-type sets, the tuluif halves and the training
lists (see its docstring). Building the starting model once, here, keeps parallel jobs from racing to build it; the
printed path must equal `$START`.

## 1. Baselines and the starting model's answers (about 35 min, all 4 GPUs)
```
for s in dev_gsm8k dev_math dev_mmlupro dev_ifsynth; do $Q add base_$s "python -m wsp.case run rd base $s"; done
$Q add base_extra_tuluif_b "python -m wsp.case run rd base extra_tuluif_b --n 2"
CUDA_VISIBLE_DEVICES= python - <<'EOF'
import json, os
S = os.environ["WSP_CASES"] + "/rd/sets/"
math = json.load(open(S + "train_math.json"))
rest = json.load(open(S + "train_gsm8k.json")) + json.load(open(S + "train_mmlupro.json"))
tl = json.load(open(S + "extra_tuluif_a.json"))
json.dump(math, open("rd_work/train_math.json", "w"))
json.dump(rest, open("rd_work/train_rest.json", "w"))
for p in range(2):
    json.dump(tl[p::2], open(f"rd_work/train_tl_{p}.json", "w"))
EOF
$Q add gen_math "python -m wsp.gen --profile $P --model $START --ids rd_work/train_math.json --n 1 --salt rd1 --max-tokens 16384 --out rd_work/gen_math.jsonl"
$Q add gen_rest "python -m wsp.gen --profile $P --model $START --ids rd_work/train_rest.json --n 2 --salt rd1 --max-tokens 12000 --out rd_work/gen_rest.jsonl"
for p in 0 1; do
  $Q add gen_tl_$p "python -m wsp.gen --profile $P --model $START --ids rd_work/train_tl_$p.json --n 2 --salt rd1 --max-tokens 12000 --out rd_work/gen_tl_$p.jsonl"
done
until $Q wait 30 gen_ base_; do echo "still running"; done
```
Read per source (the `kind` field): right, open at the cap, finished wrong; for tuluif, the share failing its
checker. Write them in NOTES.

## 2. Rescues (about 15 min, 2 GPUs)
```
for p in 0 1; do
  $Q add br_tl_$p "python $D/branch.py --model $START --rows rd_work/gen_tl_$p.jsonl --out rd_work/br_tl_$p.jsonl --k 4 --salt rdb --penalty prefix --kinds if"
done
until $Q wait 30 br_; do echo "still running"; done
$Q log br_tl_0 12; $Q log br_tl_1 12
```
Each log ends with the number of failing answers branched and rescued. Under 15% rescued, or fewer than 60 rescues in
all: write that the method does not apply here, and stop.

## 3. Fits (about 10 min, three GPUs)
```
R="rd_work/gen_math.jsonl rd_work/gen_rest.jsonl rd_work/gen_tl_0.jsonl rd_work/gen_tl_1.jsonl"
B="rd_work/br_tl_0.jsonl rd_work/br_tl_1.jsonl"
X="$WSP_CASES/rd/sets/dev.json $WSP_CASES/rd/sets/holdout.json $WSP_CASES/rd/sets/extra_tuluif_b.json $WSP_CASES/rd/sets/parent_fit.json"
COMMON="--model $START --rows $R --heads --layers 14-21 --subsample 0.8 --loss sampler --chunk 512 --tether 1.0 --anchor-nll answer --think-tether 3.0 --max-len 28672 --exclude $X"
for s in 1 2; do $Q add fit_A$s "python $D/fit_gains.py fit $COMMON --rescues $B --rescue-branches if --per-point 2 --out rd_work/fits/A$s --seed $s"; done
$Q add fit_C1 "python $D/fit_gains.py fit $COMMON --out rd_work/fits/C1 --seed 1 --limit 280"
until $Q wait 30 fit_; do echo "still running"; done
```
The log prints the rescues and anchors used and, every 20 steps, the loss and the largest |log gain|. The control C1
uses anchors only (280 sequences, about as many as one rescue fit).

## 4. Export, limits, first upload (5 min)
```
E=$WSP_CASES/rd/edits
CUDA_VISIBLE_DEVICES= python $D/fit_gains.py export --fits rd_work/fits/A1 rd_work/fits/A2 --k 3.0 --name A30 --spec-dir $E --parent ../parent/edit.json
CUDA_VISIBLE_DEVICES= python $D/fit_gains.py export --fits rd_work/fits/A1 rd_work/fits/A2 --k 2.0 --name A20 --spec-dir $E --parent ../parent/edit.json
CUDA_VISIBLE_DEVICES= python $D/fit_gains.py export --fits rd_work/fits/C1 --k 3.0 --name C30 --spec-dir $E --parent ../parent/edit.json
python -m wsp.upload rd A30 --dry-run --report $WSP_CASES/rd/NOTES.md
axonry-check $WSP_WORK/deliver/rd__A30/edit --task task.json
python -m wsp.upload rd A30 --task task.json --report REPORT.md
```
Write REPORT.md first (the numbers so far). The spec must say `"parent": "../parent/edit.json"` relative to the
case's edits directory, as exported.

## 5. Dev screen (about 45 min, all 4 GPUs)
```
for m in A30 A20 C30; do
  $Q add ev_${m}_extra_tuluif_b "python -m wsp.case run rd $m extra_tuluif_b --n 2"
  for s in dev_gsm8k dev_math dev_mmlupro dev_ifsynth; do $Q add ev_${m}_$s "python -m wsp.case run rd $m $s"; done
done
until $Q wait 30 ev_; do echo "still running"; done
CUDA_VISIBLE_DEVICES= python $D/report.py rd A30 A20 C30
```
`report.py` prints, per edit and set, the paired gain and se against the starting model, the share cut before and
after and the thinking length ratio, then the dev and the holdout calculation averages.

## 6. Choice (fixed in advance)
Keep A30, unless A30's dev calculation average (gsm8k, math, mmlupro) is more than 2 points below A20's while A20 keeps
at least two thirds of A30's tuluif B gain: then A20. Record both against C30.

## 7. Freeze the choice, then its holdout (about 45 min, all 4 GPUs)
```
M=A30     # or A20, by step 6
python -m wsp.case freeze rd $M
for s in holdout_gsm8k holdout_math holdout_mmlupro holdout_ifsynth; do
  $Q add hb_$s "python -m wsp.case run rd base $s"; $Q add he_$s "python -m wsp.case run rd $M $s"; done
until $Q wait 30 hb_ he_; do echo "still running"; done
CUDA_VISIBLE_DEVICES= python $D/report.py rd $M
```

## 8. Deliver (10 min)
Deliver the choice of step 6. If its holdout calculation average (the last line of step 7) is more than 3 points below
the starting model, say plainly in the report that the method did not confirm here.
```
M=A30     # the choice of step 6, as in step 7
python -m wsp.upload rd $M --dry-run --report REPORT.md
axonry-check $WSP_WORK/deliver/rd__$M/edit --task task.json
python -m wsp.upload rd $M --task task.json --report REPORT.md
```
REPORT.md: the starting model's failure counts per source; the rescue yield; rescues and anchors used; gain ranges;
dev table of A30, A20 and C30 with thinking ratios; the choice; holdout; what failed.
