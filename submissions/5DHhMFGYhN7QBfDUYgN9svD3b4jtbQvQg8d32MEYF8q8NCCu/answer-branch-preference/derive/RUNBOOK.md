# Runbook

Commands for `../METHOD.md`, in order. They run in the sandbox, in the work directory, with this submission copied
there as `submission/`, and `task.json` and `<sid>/edit/` for each published start beside it. This method starts
from the entry's parent (the start whose role in `task.json` is "parent"; `$PARENT` below is its sid), not from the
champion. Times are wall-clock on four GPUs; a confirmed edit takes about 3.5 h. Use the spare time as step 9 says,
and keep the last 30 minutes free of new experiments.

GPU work goes through the queue: `$Q add NAME 'CMD'` starts a detached job (one per GPU, in order; jobs survive a
dropped connection); `$Q wait [MINUTES] [PREFIX ...]` returns 0 when those jobs are done, or after MINUTES (default 2.5)
with exit code 2 and the names still running. Failed jobs are printed with the end of their log. `$Q log NAME` shows a
job's log, `$Q cancel NAME` stops it. CPU-only steps run with `CUDA_VISIBLE_DEVICES=`. Every step is safe to re-run.
Only one agent drives the queue: before starting a step, check `$Q status` and the end of `PROGRESS.md`.

Every new shell needs these lines first:
```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py" P=qwen35-0.8b-think PYTHONDONTWRITEBYTECODE=1
START=$WSP_WORK/models/rd__base     # the starting model (built once, in step 0)
PARENT=$(python -c "import json; print([s['sid'] for s in json.load(open('task.json'))['starts'] if s['role'] == 'parent'][0])")
E=$WSP_CASES/rd/edits
```
Keep `PROGRESS.md` (append-only, timestamped) and `$WSP_CASES/rd/NOTES.md` current: every number below belongs in the
report.

## 0. Case and sets (3 min)
```
python -m wsp.case init rd --profile $P --practice gsm8k_train math_train mmlupro ifsynth --dev 8 --holdout 8 \
       --parent $PARENT/edit/edit.json
CUDA_VISIBLE_DEVICES= python $D/sets.py rd
CUDA_VISIBLE_DEVICES= python -c "from wsp import case; print(case.model_path('rd', 'base'))"     # builds $START
mkdir -p rd_work
```
`sets.py` overwrites the case's dev and holdout lists and adds the per-type sets, the tuluif halves and the training
lists (see its docstring). The printed path must equal `$START`.

## 1. Baselines and the starting model's answers (about 35 min, all 4 GPUs)
```
for s in dev_gsm8k dev_math dev_mmlupro dev_ifsynth extra_tuluif_b; do $Q add base_$s "python -m wsp.case run rd base $s --n 2"; done
CUDA_VISIBLE_DEVICES= python - <<'EOF'
import json, os
from wsp import data
S = os.environ["WSP_CASES"] + "/rd/sets/"
checks = set()
for f in os.listdir(S):
    if f.startswith(("dev", "holdout", "extra_tuluif_b")):
        checks |= set(json.load(open(S + f)))
ifs = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "ifsynth" and i not in checks)
json.dump(json.load(open(S + "extra_tuluif_a.json")), open("rd_work/train_tl.json", "w"))
json.dump(ifs, open("rd_work/train_ifs.json", "w"))
calc = (json.load(open(S + "train_gsm8k.json"))[:200] + json.load(open(S + "train_math.json"))[:100]
        + json.load(open(S + "train_mmlupro.json"))[:200])
json.dump(calc, open("rd_work/train_calc.json", "w"))
print("tuluif A", len(json.load(open("rd_work/train_tl.json"))), "ifsynth", len(ifs), "calc", len(calc))
EOF
$Q add gen_tl "python -m wsp.gen --profile $P --model $START --ids rd_work/train_tl.json --n 2 --salt rd1 --max-tokens 6144 --out rd_work/gen_tl.jsonl"
$Q add gen_ifs "python -m wsp.gen --profile $P --model $START --ids rd_work/train_ifs.json --n 2 --salt rd1 --max-tokens 6144 --out rd_work/gen_ifs.jsonl"
$Q add gen_calc "python -m wsp.gen --profile $P --model $START --ids rd_work/train_calc.json --n 1 --salt rd2 --max-tokens 6144 --out rd_work/gen_calc.jsonl"
until $Q wait 30 gen_; do echo "still running"; done
```
Read per source (the `kind` field): right, open at the cap, finished wrong; for tuluif and ifsynth, the share failing
its checker. Write them in NOTES. `gen_calc` gives the kept answers of step 3 (only finished answers with a closed
thinking are used: in our run about half, mostly gsm8k and mmlupro). The baselines may still run while step 2 starts.

## 2. Branches (about 25 min, 2 GPUs)
```
$Q add br_tl "python $D/branch.py --model $START --rows rd_work/gen_tl.jsonl --out rd_work/br_tl.jsonl --k 4 --salt rdb --penalty prefix --kinds if --if-all"
$Q add br_ifs "python $D/branch.py --model $START --rows rd_work/gen_ifs.jsonl --out rd_work/br_ifs.jsonl --k 4 --salt rdb --penalty prefix --kinds if --if-all"
until $Q wait 30 br_; do echo "still running"; done
$Q log br_tl 6; $Q log br_ifs 6
```
Each log ends with, per source, the continuations that pass and the answers that have at least one passing
continuation. The fits (step 3) print the number of pairs and of branch points. Under 5% of all branch points (both
sources together) with mixed outcomes, or fewer than 200 pairs: write that the method does not apply here, deliver
nothing new, and stop.

## 3. Four fits (about 25 min, all 4 GPUs)
```
X="$WSP_CASES/rd/sets/dev.json $WSP_CASES/rd/sets/holdout.json $WSP_CASES/rd/sets/extra_tuluif_b.json"
for s in 1 2 3 4; do $Q add fit_P$s "python $D/pref_gains.py --model $START --branches rd_work/br_tl.jsonl rd_work/br_ifs.jsonl --out rd_work/fits/P$s --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --seed $s --exclude $X --keep rd_work/gen_calc.jsonl --keep-w 30 --keep-len 4096 --chunk 512"; done
until $Q wait 30 fit_; do echo "still running"; done
$Q log fit_P1 4
```
The log prints the pairs used, the kept answers per source and, every 40 pairs, the loss, the share of pairs already
ordered right ("pref-acc"), the kept answers' KL to the starting model ("keep-kl", per position) and the largest
|log gain|. Record them.

## 4. Export, limits, first upload (5 min)
```
for k in 0.25 0.5 1.0; do n=A$(echo $k | tr -d .); CUDA_VISIBLE_DEVICES= python $D/fit_gains.py export --fits rd_work/fits/P1 rd_work/fits/P2 rd_work/fits/P3 rd_work/fits/P4 --k $k --name $n --spec-dir $E --parent ../$PARENT/edit/edit.json; done
python -m wsp.upload rd A025 --dry-run --task task.json --report $WSP_CASES/rd/NOTES.md
axonry-check $WSP_WORK/deliver/rd__A025/edit --task task.json
python -m wsp.upload rd A025 --task task.json --report REPORT.md
```
The names are A025, A05, A10. Write REPORT.md first (the numbers so far). The exported spec says
`"parent": "../$PARENT/edit/edit.json"`, relative to the case's edits directory (where `init --parent` keeps the
start: check `case.json`); the upload writes the parent's sid into the delivered `edit.json`.

## 5. Dev screen (about 60 min, all 4 GPUs)
```
for m in A025 A05 A10; do
  for s in extra_tuluif_b dev_gsm8k dev_math dev_mmlupro dev_ifsynth; do $Q add ev_${m}_$s "python -m wsp.case run rd $m $s --n 2"; done
done
until $Q wait 30 ev_ base_; do echo "still running"; done
CUDA_VISIBLE_DEVICES= python $D/report.py rd A025 A05 A10
```
`report.py` prints, per edit and set, the paired gain and se against the starting model, the share cut before and
after and the thinking length ratio, then the dev calculation average.

## 6. Choice (fixed in advance)
Take the highest dose whose dev calculation average (gsm8k, math, mmlupro) is no more than 1.5 points below the
starting model's and whose tuluif B gain is at least 1 point above the next lower dose's; otherwise the next lower
dose. If no dose has a tuluif B gain of at least +1.0, the method did not work here: say so in the report and deliver the
first upload's dose only if its dev calculation average is not below the starting model's. Call the choice F.

## 7. Freeze the choice, then its holdout (about 45 min, all 4 GPUs)
```
F=A025    # the choice of step 6 (example)
python -m wsp.case freeze rd $F
for s in holdout_gsm8k holdout_math holdout_mmlupro holdout_ifsynth; do
  $Q add hb_$s "python -m wsp.case run rd base $s --n 2"; $Q add he_$s "python -m wsp.case run rd $F $s --n 2"; done
until $Q wait 30 hb_ he_; do echo "still running"; done
CUDA_VISIBLE_DEVICES= python $D/report.py rd $F
```

## 8. Deliver (10 min)
Deliver F. If its holdout calculation average (the last line of step 7) is more than 3 points below the starting
model, say plainly in the report that the method did not confirm here. The upload carries the ops of the local edit
(the gains layer) and names the start by its sid.
```
F=A025    # as in step 7
python -m wsp.upload rd $F --dry-run --task task.json --report REPORT.md
axonry-check $WSP_WORK/deliver/rd__$F/edit --task task.json
python -m wsp.upload rd $F --task task.json --report REPORT.md
```
REPORT.md: the starting model's failure counts per source; branch statistics (mixed shares, pairs); the fits' losses
and pref-acc; gain ranges; the dev table of A025, A05 and A10 with thinking ratios; the choice; holdout; what failed.

## 9. Spare time
With more than 90 minutes left after step 8: fit four more seeds (5-8) on the same pairs, export the eight-fit average
at the chosen dose, run it on tuluif B and dev calculation (2 samples), and re-deliver it only if it beats F on
tuluif B and its dev calculation average is not lower. Never start a new
experiment in the last 30 minutes; the last upload must be complete and checked.
