"""run_ec.py: the whole derivation (METHOD.md section 3) as one bash program run by this file.

  cd /workspace/WeightSpaceProgramming
  nohup python /workspace/work/submission/derive/run_ec.py > /workspace/work/run_ec.log 2>&1 &
  grep PHASE /workspace/work/run_ec.log

Four GPUs (GPUS, default 0,1,2,3). Every stage prints a PHASE line and is skipped when its output already exists, so the
script can be started again after a stop and continues where it stopped. The final edit is the spec F in the case "ec"
(eight fits averaged, strength K = 0.75).
"""
import subprocess
import sys

SCRIPT = r'''
set -e
WK=${WK:-/workspace/work}; export WK; D=$WK/submission/derive; DG=$D/gains; GPUS=${GPUS:-0,1,2,3}; G=(${GPUS//,/ }); NG=${#G[@]}
K=${K:-0.75}; KEEP=${KEEP:-1}; NFITS=${NFITS:-8}
P=qwen35-0.8b-think; W=$WK/ec_work; mkdir -p $W/fits
PARENT=0009578577-0019
cd /workspace/WeightSpaceProgramming
ph() { echo "PHASE $1 $(date -u +%H:%M)"; }

# run a wsp.gen job; when every shard has printed its "[gen] N answers" line but the job has not ended 5 minutes
# later, a vLLM worker is stuck at shutdown: its spawn children of this job get SIGTERM (answers are already collected)
gen() {  # gen OUT LOG args...
  local out=$1 log=$2; shift 2
  [ -s $out ] && { echo "[gen] $out exists"; return 0; }
  python -m wsp.gen "$@" --out $out > $log 2>&1 &
  local pid=$! done_at=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    local n=$(grep -c "^\[gen\] .* answers" $log 2>/dev/null || true)
    if [ "${n:-0}" -ge "$NG" ]; then
      [ $done_at -eq 0 ] && done_at=$(date +%s)
      if [ $(( $(date +%s) - done_at )) -gt 300 ]; then
        echo "[gen] all shards done 5 min ago: SIGTERM to the job's stuck workers"; pkill -TERM -P $pid || true; done_at=$(date +%s)
      fi
    fi
  done
  wait $pid || true
  grep "^\[gen\]" $log | tail -2
  [ -s $out ] || { echo "[gen] $out was not written: see $log"; exit 1; }
}

# 1. the case on the declared parent's published model, as it is (#138 and the chain it stands on, from the bundle)
ph case
[ -f $WSP_CASES/ec/case.json ] || python -m wsp.case init ec --profile $P --practice math_train --dev 8 --holdout 8 \
  --parent $WK/$PARENT/edit/edit.json
START=$(CUDA_VISIBLE_DEVICES= python -c "from wsp import case; print(case.model_path('ec', 'base'))" | tail -1); echo "START $START"
PAR=$(python -c "import json,os; print(json.load(open(os.environ['WSP_CASES']+'/ec/case.json'))['parent'])")
E=$WSP_CASES/ec/edits; mkdir -p $E

# 2. the bank list (CPU): 600 practice maths problems of levels 3-5, outside the case's sets (fixed shuffle)
ph lists
CUDA_VISIBLE_DEVICES= python - <<'PY'
import json, os, random
from wsp import data
S = os.environ["WSP_CASES"] + "/ec/sets/"
used = set()
for f in (os.listdir(S) if os.path.isdir(S) else []):
    if f.endswith(".json"):
        used |= set(json.load(open(S + f)))
rows = data.load("practice:math_train")
lvl = lambda r: str(r.get("level", "")).strip()[-1:]
hard = [r["id"] for r in rows if lvl(r) in "345" and r["id"] not in used]
random.Random(105).shuffle(hard)
W = os.environ["WK"] + "/ec_work/"
json.dump(hard[:600], open(W + "bank_ids.json", "w"))
print("bank", len(hard[:600]))
PY

# 3. the bank: four answers to each bank problem at the profile's decoding and token limit (~50 min)
ph bank
CUDA_VISIBLE_DEVICES=$GPUS gen $W/bank.jsonl $W/bank.log --profile $P --model $START --ids $W/bank_ids.json --n 4 --salt ec1
CUDA_VISIBLE_DEVICES= NG=$NG python - <<'PY'
import json, collections, os
W = os.environ["WK"] + "/ec_work/"
by = collections.defaultdict(list)
for l in open(W + "bank.jsonl"):
    r = json.loads(l); by[r["id"]].append(r)
keep = []
for i, L in by.items():
    fin = [r for r in L if r.get("finish") == "stop" and r.get("n_tokens", 0) > 3000 and "</think>" in r.get("text", "")]
    ok = [r for r in fin if r["correct"]]; bad = [r for r in fin if not r["correct"]]
    if ok and bad:
        keep.append([ok[0], bad[0]])
    if len(keep) >= 200:
        break
ng = int(os.environ["NG"])
for s in range(ng):
    open(W + f"mixed_{s}.jsonl", "w").write("".join(json.dumps(r) + "\n" for p in keep[s::ng] for r in p))
print("mixed problems", len(keep))
# the easy-problem keep set: one answer (the shortest) to each bank problem the start answered right 4 times out of 4,
# every answer under 12,000 tokens
easy = [L for L in by.values() if all(r.get("correct") for r in L) and max(r.get("n_tokens", 0) for r in L) < 12000]
open(W + "keep_easy.jsonl", "w").write("".join(json.dumps(min(L, key=lambda r: r["n_tokens"])) + "\n" for L in easy))
print("easy keep rows", len(easy))
PY

# 4. early branches: each trace cut near 10 / 25 / 50 % of its thinking, continued 4 times (24k new tokens), one shard
#    per GPU (~90 min)
ph branches
for s in $(seq 0 $((NG-1))); do
  [ -s $W/br_ec_$s.jsonl ] || CUDA_VISIBLE_DEVICES=${G[$s]} python $DG/branch2.py --model $START --rows $W/mixed_$s.jsonl --out $W/br_ec_$s.jsonl \
    --k 4 --salt ecb --penalty prefix --kinds wrong right --points 0.1 0.25 0.5 --max-new 24000 > $W/br_ec_$s.log 2>&1 &
done; wait
for s in $(seq 0 $((NG-1))); do [ -s $W/br_ec_$s.jsonl ] || { echo "branch shard $s failed: see $W/br_ec_$s.log"; exit 1; }; done
cat $W/br_ec_[0-9]*.jsonl > $W/br_ec.jsonl; wc -l $W/br_ec.jsonl
grep -h "branch points" $W/br_ec_*.log | head -1

# 5. eight fits (seeds 1-8, one per GPU, rounds of NG), the loss on the 2,048 tokens after the branch point; with
#    KEEP=1 each step also holds one of the start's own answers to the easy bank problems at KL weight 30 over its
#    first 6,144 generated tokens (~65 min a round)
ph fits
KEEPARGS=""; [ "$KEEP" = "1" ] && KEEPARGS="--keep $W/keep_easy.jsonl --keep-w 30 --keep-len 6144"; echo "keep: ${KEEPARGS:-none}"
i=0
for s in $(seq 1 $NFITS); do
  if [ -z "$(ls $W/fits/E$s 2>/dev/null)" ]; then
    CUDA_VISIBLE_DEVICES=${G[$((i % NG))]} python $DG/pref_gains2.py --model $START --branches $W/br_ec.jsonl --out $W/fits/E$s --layers 4-21 --heads \
      --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --seed $s --allow-think --bad-cut --window 2048 --points 0.1 0.25 0.5 \
      --max-len 32768 --chunk 1024 $KEEPARGS > $W/fit_E$s.log 2>&1 &
    i=$((i+1))
    [ $i -eq $NG ] && { wait; i=0; }
  fi
done; wait
for s in $(seq 1 $NFITS); do [ -n "$(ls $W/fits/E$s 2>/dev/null)" ] || { echo "fit E$s failed: see $W/fit_E$s.log"; exit 1; }; done
grep -h "pairs" $W/fit_E1.log | head -2

# 6. the fits' log-gains averaged in log space and exported at strength K (0.75) as the spec F (CPU)
ph export
[ -s $E/F.json ] || CUDA_VISIBLE_DEVICES= python $DG/fit_gains.py export --fits $(for s in $(seq 1 $NFITS); do echo -n "$W/fits/E$s "; done) \
  --k $K --name F --spec-dir $E --parent ../$PAR | tail -1
ph DONE
'''

if __name__ == "__main__":
    sys.exit(subprocess.run(["bash", "-c", SCRIPT]).returncode)
