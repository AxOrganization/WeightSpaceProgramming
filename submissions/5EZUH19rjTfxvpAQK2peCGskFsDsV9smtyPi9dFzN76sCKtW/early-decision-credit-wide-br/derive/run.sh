#!/bin/bash
# The whole run, step by step (METHOD.md section 5). From the task directory (task.json next to it):
#   bash submission/derive/run.sh all            every step in order, resumable (a step with a .done marker is skipped)
#   bash submission/derive/run.sh <step>         one step: start l1_pool l1_fit br_init br_gens br_branch br_fits up1
#                                                ec_bank ec_rows ec_branch ec_fits ec_dose up2
# Layers:  L1 = reversed shortcut (published method of 0009554110-0010, its scripts) on the start
#          BR = answer-branch preference gains (published method of 0009552831-0015, its scripts) on L1's model
#          EC = early-decision credit (this entry): same-prefix maths continuation pairs from early branch points,
#               preference fit with the loss on the first 2048 tokens after the branch
# The upload is ONE edit on the start: L1's ops + one gain set = BR's gains x 0.5 and EC's gains x 0.5 merged.
# Upload 1 (L1 + BR) goes out as soon as BR is fitted; upload 2 (L1 + merged gains) replaces it at the end.
set -euo pipefail
SUB="$(cd "$(dirname "$0")/.." && pwd)"; D="$SUB/derive"; ROOT="$(pwd)"
P=${PY:-python}; export PYTHONPATH="$D${PYTHONPATH:+:$PYTHONPATH}"
TASK=${TASK:-$ROOT/task.json}; RUN=$ROOT/run_ec; mkdir -p $RUN/logs; cd $RUN
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=($(echo $CUDA_VISIBLE_DEVICES | tr "," " ")); else GPUS=($(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))); fi
NG=${#GPUS[@]}
PF=qwen35-0.8b-think
BR_K=${BR_K:-1.0}                 # BR gains' strength (export of upload 1 and share in the final merge)
BR_N=${BR_N:-6}                   # BR answers per instruction request
BR_NF=${BR_NF:-4}                 # BR fits (seeds 1..BR_NF): 4 fits, each on 1.5 times the pairs
DRAW=${DRAW:-1}                   # 1 = the seeds and salts below; n > 1 shifts every seed and salt (an independent draw)
DX=$((DRAW - 1)); SX=""; [ "$DRAW" -gt 1 ] && SX="_d$DRAW"
EC_K=${EC_K:-0.5}                 # EC gains' strength in the final merge (section 2 step 6)
SMALL=${SMALL:-0}                 # 1 = smoke test: tiny sizes everywhere (pipeline check only)
EC_BANK=${EC_BANK:-600}           # EC bank: maths problems answered 4 times (section 2 step 1)
EC_ROWS=${EC_ROWS:-200}           # EC: mixed problems used (step 2)
EC_POINTS=${EC_POINTS:-"0.1 0.25 0.5"}   # EC branch points, share of the thinking (step 3)
EC_DOSES=${EC_DOSES:-"0.5 0.75 1.0"}     # EC strengths screened on practice maths (step 6); the first is the fallback
EC_LEN_MAX=${EC_LEN_MAX:-1.18}         # a strength passes if its thinking is at most this many times the L1 + BR model's
[ -f t0 ] || date +%s > t0
left(){ echo $(( 6 * 60 - ( $(date +%s) - $(cat $RUN/t0) ) / 60 - ${T_OFFSET_MIN:-0} )); }   # minutes left in the run
log(){ echo "[$(date -u +%H:%M) | $(left) min left] $*" | tee -a $RUN/logs/run.log; }
done_(){ touch $RUN/$1.done; log "$1 done"; }
is_done(){ [ -f $RUN/$1.done ]; }
sz(){ [ "$SMALL" = 1 ] && echo $2 || echo $1; }

step_start(){
  START=$($P $D/start.py --task $TASK); echo "$START" > start.txt; log "start edit: $START"
  $P -c "import torch; print('GPUs', torch.cuda.device_count())"; done_ start; }

# ---------------- L1: reversed shortcut (scripts of 0009554110-0010) ----------------
step_l1_pool(){
  START=$(cat start.txt)
  [ -d "$($P -c 'from wsp import case; print(case.cdir("l1"))')" ] || $P -m wsp.case init l1 --profile $PF --parent $START \
     --practice gsm8k_train math_train mmlupro ifsynth --subset holdout256=holdout:256
  prev=()
  for sn in gsm8k_train:$(sz 400 24) math_train:$(sz 200 12) ifsynth:$(sz 80 8) mmlupro:$(sz 80 8); do s=${sn%:*}; n=${sn#*:}
    if [ ${#prev[@]} -gt 0 ]; then ex=(--exclude "${prev[@]}"); else ex=(); fi
    $P $D/mkdata.py ids --case l1 --sources $s --per-source $n --seed $((111 + 1000 * DX)) "${ex[@]}" --out pool_$s.json >> logs/l1_ids.log 2>&1
    prev+=(pool_$s.json); done
  $P -c "import json; ids=[i for s in ['gsm8k_train','math_train','ifsynth','mmlupro'] for i in json.load(open(f'pool_{s}.json'))]; json.dump(ids, open('pool.json','w')); print('pool', len(ids))"
  for t in 1 2; do $P $D/shard_gen.py --case l1 --model base --ids pool.json --n 4 --salt pool$SX --max-tokens 12000 \
       --out pool_run.jsonl > logs/l1_pool_$t.log 2>&1 && break; sleep 30; done
  test -s pool_run.jsonl; done_ l1_pool; }

step_l1_fit(){
  F="$P $D/svf_len.py --case l1 --runs pool_run.jsonl --objective rloo --span fork --fork-q 0.2 --group-take 4 --subsample 0.6 --src-weight ifsynth=0 tuluif=0 mmlupro=0 --anchor-src if mc --anchor-n $(sz 120 16) --kl 1.0 --lr 5e-3 --max-tokens 12000"
  L1E="$($P -c 'from wsp import case; print(case.cdir("l1"))')/edits"
  for try in 1 2; do   # try 2 reruns only the seeds that failed (e.g. a kernel-compile race on a fresh machine)
    pids=(); for i in 0 1 2 3; do s=$((i + 1)); g=${GPUS[$((i % NG))]}; [ -f "$L1E/L$s.raw.npz" ] && continue
      CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=$RUN/.triton/L$s $F --name L$s --seed $((s + 10 * DX)) > logs/l1_fit_L${s}_t$try.log 2>&1 & pids+=($!); sleep 5; done
    for p in "${pids[@]}"; do wait $p || log "an L1 fit failed (try $try)"; done
    [ -f "$L1E/L1.raw.npz" ] && [ -f "$L1E/L2.raw.npz" ] && [ -f "$L1E/L3.raw.npz" ] && [ -f "$L1E/L4.raw.npz" ] && break; done
  FT=$(for s in 1 2 3 4; do [ -f "$L1E/L$s.raw.npz" ] && printf "L%s " $s; done; true)
  [ -n "$FT" ] || { log "no L1 fit"; exit 1; }
  $P $D/lendose.py --case l1 --fit $FT --out Lm --norms -1.37 > logs/l1_lendose.log 2>&1   # seed mean, reversed, L2 norm 1.37
  log "L1 = Lm_nm1p37 from $FT"; done_ l1_fit; }

# ---------------- BR: answer-branch preference gains (scripts of 0009552831-0015) ----------------
CBR=br
step_br_init(){
  L1SPEC="$($P -c 'from wsp import case; print(case.cdir("l1"))')/edits/Lm_nm1p37.json"
  [ -d "$($P -c 'from wsp import case; print(case.cdir("br"))')" ] || $P -m wsp.case init $CBR --profile $PF \
     --practice gsm8k_train math_train mmlupro ifsynth --dev 8 --holdout 8 --parent $L1SPEC
  CUDA_VISIBLE_DEVICES= $P $D/sets.py $CBR > logs/br_sets.log 2>&1
  mkdir -p br_work
  CUDA_VISIBLE_DEVICES= $P - "$($P -c 'from wsp import case; print(case.cdir("br"))')/sets/" $SMALL <<'EOF'
import json, os, sys
from wsp import data
S, small = sys.argv[1], sys.argv[2] == "1"
checks = set()
for f in os.listdir(S):
    if f.startswith(("dev", "holdout", "extra_tuluif_b")):
        checks |= set(json.load(open(S + f)))
ifs = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "ifsynth" and i not in checks)
tl = json.load(open(S + "extra_tuluif_a.json"))
calc = (json.load(open(S + "train_gsm8k.json"))[:200] + json.load(open(S + "train_math.json"))[:100]
        + json.load(open(S + "train_mmlupro.json"))[:200])
if small: tl, ifs, calc = tl[:24], ifs[:24], calc[:24]
for n, v in (("tl", tl), ("ifs", ifs), ("calc", calc)): json.dump(v, open(f"br_work/train_{n}.json", "w"))
print("tuluif A", len(tl), "ifsynth", len(ifs), "calc", len(calc))
EOF
  $P -c "from wsp import case; print(case.model_path('$CBR', 'base'))" > br_work/model.txt; done_ br_init; }

slots(){   # run commands on the GPUs, one per GPU at a time (own kernel cache per GPU: no compile race between jobs)
  local q=("$@") i=0 pids=() gp=() rc=0
  while [ $i -lt ${#q[@]} ] || [ ${#pids[@]} -gt 0 ]; do
    while [ $i -lt ${#q[@]} ] && [ ${#pids[@]} -lt $NG ]; do
      used=" ${gp[*]:-} "; g=""; for x in "${GPUS[@]}"; do [[ "$used" == *" $x "* ]] || { g=$x; break; }; done
      [ -n "$g" ] || { sleep 5; continue 2; }
      CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=$RUN/.triton/g$g bash -c "${q[$i]}" & pids+=($!); gp+=($g); i=$((i + 1)); sleep 15; done
    np=(); ng=(); for k in "${!pids[@]}"; do if kill -0 ${pids[$k]} 2>/dev/null; then np+=(${pids[$k]}); ng+=(${gp[$k]}); else wait ${pids[$k]} || rc=1; fi; done
    pids=("${np[@]:-}"); gp=("${ng[@]:-}"); [ -z "${pids[0]:-}" ] && pids=() && gp=(); sleep 5; done
  return $rc; }

split_ids(){   # split_ids NAME: br_work/train_NAME.json -> NG interleaved shards br_work/sh/NAME_<i>.json
  mkdir -p br_work/sh; $P -c "
import json, sys
ids = json.load(open('br_work/train_$1.json')); n = $NG
for i in range(n): json.dump(ids[i::n], open(f'br_work/sh/$1_{i}.json', 'w'))"; }
split_by_tl(){   # split_by_tl IN OUT_TL OUT_IFS: rows of tuluif problems (train_tl.json) vs the rest (ifsynth)
  $P -c "
import json, sys
tl = set(json.load(open('br_work/train_tl.json'))); a, b = open('$2', 'w'), open('$3', 'w')
for l in open('$1'):
    if l.strip(): (a if json.loads(l)['id'] in tl else b).write(l if l.endswith(chr(10)) else l + chr(10))"; }

step_br_gens(){   # one combined IF call (tuluif A + ifsynth, 4 answers each) and one calc call per GPU, on interleaved shards
  M=$(cat br_work/model.txt); G="$P -m wsp.gen --profile $PF --model $M --max-tokens 6144"
  split_ids tl; split_ids ifs; split_ids calc
  cmds=(); for i in $(seq 0 $((NG - 1))); do
    $P -c "import json; json.dump(json.load(open('br_work/sh/tl_$i.json')) + json.load(open('br_work/sh/ifs_$i.json')), open('br_work/sh/if_$i.json', 'w'))"
    cmds+=("$G --ids br_work/sh/if_$i.json --n $BR_N --salt rd1$SX --out br_work/sh/gen_if_$i.jsonl > logs/br_gen_if_$i.log 2>&1 && $G --ids br_work/sh/calc_$i.json --n 1 --salt rd1c$SX --out br_work/sh/gen_calc_$i.jsonl > logs/br_gen_calc_$i.log 2>&1"); done
  slots "${cmds[@]}" || log "a BR gen shard failed"
  cat br_work/sh/gen_if_*.jsonl > br_work/sh/gen_if.jsonl 2>/dev/null || true; cat br_work/sh/gen_calc_*.jsonl > br_work/gen_calc.jsonl 2>/dev/null || true
  split_by_tl br_work/sh/gen_if.jsonl br_work/gen_tl.jsonl br_work/gen_ifs.jsonl
  log "BR answers: tuluif $(wc -l < br_work/gen_tl.jsonl), ifsynth $(wc -l < br_work/gen_ifs.jsonl), calc $(wc -l < br_work/gen_calc.jsonl)"
  test -s br_work/gen_tl.jsonl && test -s br_work/gen_ifs.jsonl; done_ br_gens; }

step_br_branch(){   # branches of every IF answer (k 4), one call per GPU on interleaved shards of the answers
  M=$(cat br_work/model.txt); B="$P $D/branch2.py --model $M --k 4 --salt rdb$SX --penalty prefix --kinds if --if-all"; mkdir -p br_work/sh
  $P -c "
n = $NG; rows = [l for f in ('br_work/gen_tl.jsonl', 'br_work/gen_ifs.jsonl') for l in open(f) if l.strip()]
for i in range(n): open(f'br_work/sh/rows_if_{i}.jsonl', 'w').write(''.join(rows[i::n]))"
  cmds=(); for i in $(seq 0 $((NG - 1))); do
    cmds+=("$B --rows br_work/sh/rows_if_$i.jsonl --out br_work/sh/br_if_$i.jsonl > logs/br_branch_if_$i.log 2>&1"); done
  slots "${cmds[@]}" || log "a BR branch shard failed"
  cat br_work/sh/br_if_*.jsonl > br_work/sh/br_if.jsonl 2>/dev/null || true
  split_by_tl br_work/sh/br_if.jsonl br_work/br_tl.jsonl br_work/br_ifs.jsonl
  test -s br_work/br_tl.jsonl || test -s br_work/br_ifs.jsonl; done_ br_branch; }

step_br_fits(){
  M=$(cat br_work/model.txt); CD="$($P -c 'from wsp import case; print(case.cdir("br"))')"
  X="$CD/sets/dev.json $CD/sets/holdout.json $CD/sets/extra_tuluif_b.json"
  BRS=$(ls br_work/br_tl.jsonl br_work/br_ifs.jsonl 2>/dev/null | tr "\n" " ")
  F="$P $D/pref_gains2.py --model $M --branches $BRS --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --exclude $X --keep br_work/gen_calc.jsonl --keep-w 30 --keep-len 4096 --chunk 512"
  NF=$(sz $BR_NF 2); cmds=(); for i in $(seq 1 $NF); do cmds+=("$F --out br_work/fits/P$i --seed $((i + 100 * DX)) > logs/br_fit_P$i.log 2>&1"); done
  slots "${cmds[@]}" || log "a BR fit failed"
  FT=$(for i in $(seq 1 $NF); do [ -f br_work/fits/P$i/state.pt ] && printf "br_work/fits/P%s " $i; done; true); [ -n "$FT" ] || { log "no BR fit"; exit 1; }
  echo "$FT" > br_work/fits.txt
  PAR=$($P -c "import os; from wsp import case as C; print(os.path.relpath(C.cdir('$CBR') / C.conf('$CBR')['parent'], C.cdir('$CBR') / 'edits'))")
  CUDA_VISIBLE_DEVICES= $P $D/fit_gains.py export --fits $FT --k $BR_K --levels 256 --name A05 --spec-dir $CD/edits --parent $PAR > logs/br_export.log 2>&1
  tail -n 1 logs/br_export.log; done_ br_fits; }

write_report(){   # REPORT.md for the upload: what ran, what came out (from this run's own logs); never stops the run
  { echo "# Early decisions: run report"; echo
    echo "Layers: L1 (reversed shortcut, published method of 0009554110-0010), BR (answer-branch preference gains, published"
    echo "method of 0009552831-0015), EC (early-decision credit, this entry). Upload = $1."; echo
    echo "## Steps"; grep -h "done\|L1 =\|upload\|payload\|merged\|pairs from\|window\|skipped\|strength\|BR answers" logs/run.log 2>/dev/null | sed 's/^/- /' || true
    echo; echo "## Data"
    [ -f pool_run.jsonl ] && echo "- L1 pool answers: $(wc -l < pool_run.jsonl)" || true
    [ -f br_work/br_tl.jsonl ] && echo "- BR branch rows: $(cat br_work/br_tl.jsonl br_work/br_ifs.jsonl 2>/dev/null | wc -l)" || true
    grep -h "pairs from" logs/br_fit_P1.log 2>/dev/null | sed 's/^/- BR: /' || true
    [ -f ec_work/bank.jsonl ] && echo "- EC bank answers: $(wc -l < ec_work/bank.jsonl)" || true
    [ -f ec_work/rows_wrong.jsonl ] && echo "- EC mixed problems: $(wc -l < ec_work/rows_wrong.jsonl)" || true
    grep -h "pairs from\|window" logs/ec_fit_E1.log 2>/dev/null | sed 's/^/- EC: /' || true
  } > REPORT.md || true; }

step_up1(){   # upload 1: L1 + BR (the chain br: start -> L1 -> A05, local layers carried into one upload)
  write_report "L1 + BR (upload 1)"
  if $P -m wsp.upload $CBR A05 --task $TASK --report REPORT.md ${UPLOAD_EXTRA:-} > logs/up1.log 2>&1; then log "upload 1 (L1 + BR) done"; done_ up1
  else log "upload 1 FAILED (see logs/up1.log): fix and rerun 'run.sh up1'; continuing with EC"; fi; }

# ---------------- EC: early-decision credit (this entry) ----------------
step_ec_bank(){
  M=$($P -c "from wsp import case; print(case.model_path('$CBR', 'A05'))"); echo "$M" > ec_model.txt; mkdir -p ec_work
  CUDA_VISIBLE_DEVICES= $P - $NG $(sz $EC_BANK 24) $DX <<'EOF'
import json, random, sys
from wsp import case as C, data
n, N = int(sys.argv[1]), int(sys.argv[2])
skip = set()
for c in ("l1", "br"):
    for p in (C.cdir(c) / "sets").glob("*.json"): skip |= set(json.load(open(p)))
ids = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "math" and i not in skip
             and str(it.get("level", "")).replace("Level ", "") in ("3", "4", "5"))
random.Random(4096 + int(sys.argv[3])).shuffle(ids); ids = ids[:N]
for s in range(n): json.dump(ids[s::n], open(f"ec_work/bank_ids_{s}.json", "w"))
print("bank problems", len(ids))
EOF
  cmds=(); for s in $(seq 0 $((NG - 1))); do
    cmds+=("$P -m wsp.gen --profile $PF --model $M --ids ec_work/bank_ids_$s.json --n 4 --salt ecbank$SX --max-tokens 32768 --out ec_work/bank_$s.jsonl > logs/ec_bank_$s.log 2>&1"); done
  slots "${cmds[@]}" || log "a bank shard failed"
  cat ec_work/bank_*.jsonl > ec_work/bank.jsonl 2>/dev/null || true; test -s ec_work/bank.jsonl; done_ ec_bank; }

step_ec_rows(){   # problems solved in some samples and not in others: one finished wrong and one finished right trace each
  $P - ec_work/bank.jsonl $(sz $EC_ROWS 6) $DX <<'EOF'
import json, random, sys
from collections import defaultdict
g = defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); g[r["id"]].append(r)
fin = lambda r: r.get("finish") == "stop" and r.get("think_closed") and not r.get("cut") and "</think>" in r.get("text", "")
rng = random.Random(11 + int(sys.argv[3])); ids = sorted(g); rng.shuffle(ids); W, R = [], []
for i in ids:
    v = g[i]
    w = [r for r in v if fin(r) and not r.get("correct") and r.get("kind") == "finished_wrong" and (r.get("n_tokens") or 0) >= 3000]
    c = [r for r in v if fin(r) and r.get("correct") is True and (r.get("n_tokens") or 0) >= 3000]
    if w and c and len(W) < int(sys.argv[2]): W.append(rng.choice(w)); R.append(rng.choice(c))
for name, rows in (("ec_work/rows_wrong.jsonl", W), ("ec_work/rows_right.jsonl", R)):
    with open(name, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
print("mixed problems used", len(W), "of", len(g))
EOF
  test -s ec_work/rows_wrong.jsonl; done_ ec_rows; }

step_ec_branch(){   # continuations of both traces from 10 %, 25 % and 50 % of the thinking, 4 each
  M=$(cat ec_model.txt)
  $P - $NG <<'EOF'
import sys
n = int(sys.argv[1])
for kind in ("wrong", "right"):
    rows = open(f"ec_work/rows_{kind}.jsonl").read().splitlines()
    for i in range(n): open(f"ec_work/rows_{kind}_{i}.jsonl", "w").write("\n".join(rows[i::n]) + "\n")
EOF
  cmds=(); for kind in wrong right; do for i in $(seq 0 $((NG - 1))); do
    cmds+=("$P $D/branch2.py --model $M --rows ec_work/rows_${kind}_$i.jsonl --out ec_work/br_${kind}_$i.jsonl --kinds $kind --points $EC_POINTS --k 4 --max-new $(sz 24000 4000) --salt ec$SX --penalty prefix > logs/ec_branch_${kind}_$i.log 2>&1"); done; done
  slots "${cmds[@]}" || log "a branch shard failed"
  cat ec_work/br_wrong_*.jsonl > ec_work/br_wrong.jsonl 2>/dev/null || true; cat ec_work/br_right_*.jsonl > ec_work/br_right.jsonl 2>/dev/null || true
  test -s ec_work/br_wrong.jsonl; done_ ec_branch; }

step_ec_fits(){   # same-prefix pairs (right vs wrong ending), loss on the first 2048 tokens after the branch
  M=$(cat ec_model.txt)
  F="$P $D/pref_gains2.py --model $M --branches ec_work/br_wrong.jsonl ec_work/br_right.jsonl --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --max-len 24576 --chunk 1024 --allow-think --bad-cut --window 2048 --points $EC_POINTS"
  NF=$(sz 4 1); cmds=(); for i in $(seq 1 $NF); do cmds+=("$F --out ec_work/fits/E$i --seed $((i + 100 * DX)) > logs/ec_fit_E$i.log 2>&1"); done
  slots "${cmds[@]}" || log "an EC fit failed"
  grep -h "pairs from\|window" logs/ec_fit_E1.log | tee -a logs/run.log || true
  ls ec_work/fits/E*/state.pt > /dev/null; done_ ec_fits; }

step_ec_dose(){   # choose EC's strength by thinking length on fresh practice maths (section 2 step 6)
  EF=$(ls -d ec_work/fits/E* | tr "\n" " "); L1E="$($P -c 'from wsp import case; print(case.cdir("l1"))')/edits"
  K0=$(echo $EC_DOSES | cut -d" " -f1)
  if [ $(left) -lt ${EC_DOSE_MIN_LEFT:-50} ]; then echo $K0 > ec_work/ec_k.txt; log "only $(left) min left: no strength screen, EC strength $K0"; done_ ec_dose; return 0; fi
  CUDA_VISIBLE_DEVICES= $P - $EC_BANK $(sz 256 8) $DX <<'PYEOF'
import json, random, sys
from wsp import case as C, data
N, n = int(sys.argv[1]), int(sys.argv[2])
skip = set()
for c in ("l1", "br"):
    for p in (C.cdir(c) / "sets").glob("*.json"): skip |= set(json.load(open(p)))
ids = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "math" and i not in skip
             and str(it.get("level", "")).replace("Level ", "") in ("3", "4", "5"))
random.Random(4096 + int(sys.argv[3])).shuffle(ids)   # the bank's order: the screen is the n problems after the bank's N
json.dump(ids[N:N + n], open("ec_work/screen_ids.json", "w")); print("screen problems", len(ids[N:N + n]))
PYEOF
  names=(base); models=("$(cat ec_model.txt)")
  for k in $EC_DOSES; do t=$(echo $k | tr -d .); names+=(k$t)
    $P $D/merge_gains.py ec_work/fits/m$t --set $BR_K $(cat br_work/fits.txt) --set $k $EF > logs/ec_dose_merge_$t.log 2>&1
    CUDA_VISIBLE_DEVICES= $P $D/fit_gains.py export --fits ec_work/fits/m$t --k 1.0 --levels 256 --name ECk$t --spec-dir $L1E --parent Lm_nm1p37.json > logs/ec_dose_export_$t.log 2>&1
    models+=("$($P -c "from wsp import case; print(case.model_path('l1', 'ECk$t'))")"); done
  cmds=(); for i in "${!names[@]}"; do
    cmds+=("$P -m wsp.gen --profile $PF --model ${models[$i]} --ids ec_work/screen_ids.json --n 1 --salt ecdose$SX --max-tokens 32768 --out ec_work/dose_${names[$i]}.jsonl > logs/ec_dose_gen_${names[$i]}.log 2>&1"); done
  slots "${cmds[@]}" || log "a strength-screen generation failed"
  $P - $EC_LEN_MAX $EC_DOSES <<'PYEOF' | tee -a logs/run.log
import json, os, sys
mx, ks = float(sys.argv[1]), sys.argv[2:]
def stats(name):
    p = f"ec_work/dose_{name}.jsonl"
    if not os.path.exists(p): return None
    r = [json.loads(l) for l in open(p) if l.strip()]
    if not r: return None
    tok = sum(x.get("n_tokens") or 0 for x in r) / len(r)
    cut = sum(1 for x in r if x.get("cut") or x.get("finish") == "length") / len(r)
    acc = sum(1 for x in r if x.get("correct") is True) / len(r)
    return tok, cut, acc, len(r)
b = stats("base"); pick = ks[0]
print("strength screen (fresh practice maths, 1 answer each): mean tokens, cut share, accuracy, answers")
if b: print(f"  base  {b[0]:7.0f} {b[1]:6.1%} {b[2]:6.1%} {b[3]}")
for k in ks:
    s = stats("k" + k.replace(".", ""))
    if not s or not b: print(f"  {k}: no screen"); continue
    ok = s[0] <= mx * b[0] and s[1] <= max(0.06, b[1] + 0.02)   # cut share: at most 2 points up, or at most 6 %
    print(f"  {k:5s} {s[0]:7.0f} {s[1]:6.1%} {s[2]:6.1%} {s[3]}  length x{s[0] / b[0]:.2f} {'ok' if ok else 'too long'}")
    if ok and float(k) > float(pick): pick = k
open("ec_work/ec_k.txt", "w").write(pick + "\n"); print("EC strength chosen:", pick)
PYEOF
  done_ ec_dose; }

step_up2(){   # upload 2: L1 + (BR x 0.5 + EC x 0.5) as one gain set, exported once (payload: one gain set, not two)
  L1E="$($P -c 'from wsp import case; print(case.cdir("l1"))')/edits"
  EF=$(ls -d ec_work/fits/E* | tr "\n" " ")
  [ -f ec_work/ec_k.txt ] && EC_K=$(cat ec_work/ec_k.txt); log "EC strength $EC_K"
  $P $D/merge_gains.py ec_work/fits/merged --set $BR_K $(cat br_work/fits.txt) --set $EC_K $EF | tee -a logs/run.log
  CUDA_VISIBLE_DEVICES= $P $D/fit_gains.py export --fits ec_work/fits/merged --k 1.0 --levels 256 --name EC --spec-dir $L1E --parent Lm_nm1p37.json > logs/ec_export.log 2>&1
  tail -n 1 logs/ec_export.log | tee -a logs/run.log
  $P - $L1E <<'EOF' | tee -a logs/run.log
import sys
from wsp.edits import spec_size
from wsp.rules import RULES
e = sys.argv[1]; a = spec_size(f"{e}/Lm_nm1p37.json")[1]; b = spec_size(f"{e}/EC.json")[1]
lim = RULES["edit"]["limits"]["max_payload_bytes"]
print(f"payload L1 {a} + gains {b} = {a + b} of {lim}")
sys.exit(0 if a + b <= lim else 3)
EOF
  $P -c "from wsp import case; print(case.model_path('l1', 'EC'))" >> logs/run.log
  write_report "L1 + merged BR/EC gains (upload 2, final)"
  if $P -m wsp.upload l1 EC --task $TASK --report REPORT.md ${UPLOAD_EXTRA:-} > logs/up2.log 2>&1; then log "upload 2 (final: L1 + merged gains) done"; done_ up2
  else log "upload 2 FAILED (see logs/up2.log): fix and rerun 'run.sh up2'"; exit 1; fi; }

run(){ local s=$1; is_done $s && { log "$s already done"; return 0; }; log "$s ..."; step_$s; }
case ${1:-all} in
  all)
    for s in start l1_pool l1_fit br_init br_gens br_branch br_fits up1; do run $s; done
    if [ $(left) -lt ${EC_MIN_LEFT:-150} ]; then log "only $(left) min left: the EC layer is skipped, upload 1 stands"; exit 0; fi
    for s in ec_bank ec_rows ec_branch ec_fits ec_dose; do run $s; done
    run up2 ;;
  *) run $1 ;;
esac
