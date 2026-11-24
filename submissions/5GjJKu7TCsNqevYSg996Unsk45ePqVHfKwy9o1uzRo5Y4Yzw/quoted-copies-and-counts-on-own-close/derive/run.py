#!/usr/bin/env python3
"""run.py: the whole GPU part of this method, resumable (a stage whose output exists is skipped). From /workspace/work:

    nohup python derive/run.py > run.log 2>&1 &

  1. keep rows: the parent's own right answers (requests of keep/reqgen.py, word problems, math, multiple choice, 200
     practice puzzles of every family), and the parent's absolute MLP units (START.json) that the fit must not take;
  2. round 2 and round 3 of the instruction requests (derive/mkreq.py), the parent's own answers x4, answers sampled
     again after the same thinking, failed answers mended by rule (derive/repair.py) and paired;
  3. one preference fit of both rounds' pairs with the keep term (keep/train_keep.py), half and full dose, int8 form.
The steps are shell commands of the toolkit and of the shipped scripts; they run in one bash process, in order.
"""
import subprocess, sys

SCRIPT = r"""#!/bin/bash
# The whole GPU part of this method, resumable (a stage whose output exists is skipped): run it detached from /workspace/work.
#  1. keep rows: the parent's own right answers (requests of keep/reqgen.py, word problems, math, multiple choice, 200
#     practice puzzles of every family), and the parent's absolute MLP units (START.json) that the fit must not take;
#  2. round 2 and round 3 of the instruction requests (derive/mkreq.py), the parent's own answers x4, answers sampled again
#     after the same thinking, failed answers mended by rule (derive/repair.py) and paired;
#  3. one preference fit of both rounds' pairs with the keep term (keep/train_keep.py), half and full dose, int8 directions.
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
[ -d /srv/axonry/repos/WSP-667dfec ] && export PYTHONPATH=/srv/axonry/repos/WSP-667dfec${PYTHONPATH:+:$PYTHONPATH}   # rehearsal box only; the sandbox has wsp installed
[ -d /srv/axonry/models ] && [ -z "${WSP_MODELS_DIR:-}" ] && export WSP_MODELS_DIR=/srv/axonry/models
[ -d /usr/share/nltk_data ] && export NLTK_DATA=${NLTK_DATA:-/usr/share/nltk_data}
P=0009576150-0005; OUT=/workspace/work/w2; D2=keep; E=/workspace/work/cases/mp/edits
ALL="temporal_sequences,hyperbaton,geometric_shapes,object_counting,tracking_shuffled_objects_three_objects,tracking_shuffled_objects_five_objects,tracking_shuffled_objects_seven_objects,logical_deduction_three_objects,logical_deduction_five_objects,logical_deduction_seven_objects,navigate,boolean_expressions,multistep_arithmetic_two,web_of_lies,date_understanding,penguins_in_a_table,reasoning_about_colored_objects,formal_fallacies,word_sorting"
say() { echo "[run] $(date -u +%H:%M:%S) $*"; }
cd /workspace/work; mkdir -p $OUT/logs $OUT/keep
# ---- 1. keep rows on pr128
[ -f cases/w2/case.json ] || python -m wsp.case init w2 --profile qwen35-0.8b-think --parent $P/edit/edit.json \
  --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800 2>&1 | tail -1
[ -f $OUT/req/req.jsonl ] || python $D2/reqgen.py --out-dir $OUT/req 2>&1 | tail -1 | sed "s/^/[run] reqgen /"
M=$(python -c "from wsp import case; print(case.model_path('w2', 'base'))" | tail -1); say "start model $M"
if [ ! -f $OUT/items_all.jsonl ]; then
  REQ_ITEMS= python $D2/practice_puzzles.py --per-task 600 --seed 41 --prefix pzc --set-to 96 --out $OUT/pz_fit.jsonl 2>&1 | tail -1 | sed "s/^/[run] /"
  cat $OUT/req/req.jsonl $OUT/pz_fit.jsonl > $OUT/items_all.jsonl
fi
export REQ_ITEMS=$OUT/items_all.jsonl
[ -f $OUT/keep/keep_ids.json ] || python $D2/keep_rows.py pick --case w2 --req $OUT/req --puzzles $OUT/pz_fit.jsonl --held $ALL --out $OUT/keep 2>&1 | tail -1 | cut -c1-200 | sed "s/^/[run] /"
say "keep rollout start"
[ -f $OUT/keep/keep_other.jsonl ] || { python $D2/x_rollouts.py run --model $M --ids $OUT/keep/keep_ids.json --out $OUT/keep/roll --gpus 0,1,2,3 \
    --caps req=12288,pz=20480,gsm8k=12288,math=20480,mmlupro=12288 --default-cap 20480 --first 1 --more 0 > $OUT/logs/keep.log 2>&1
  python $D2/keep_rows.py rows --rows $OUT/keep/roll/rows.jsonl --out $OUT/keep 2>&1 | tail -2 | cut -c1-240 | sed "s/^/[run] keep /"; }
if [ ! -f $OUT/START.json ]; then   # pr130's absolute units only (set_rows / set_cols); its 4,594 scale_rows gain ops are relative and compose with ours
  mkdir -p $OUT/p_units && cp $P/edit/edit.npz $OUT/p_units/edit.npz
  python -c "
import json; e = json.load(open('$P/edit/edit.json')); e['ops'] = [o for o in e['ops'] if o['op'] in ('set_rows', 'set_cols')]
json.dump(e, open('$OUT/p_units/edit.json', 'w')); print('[run] absolute ops of the start:', len(e['ops']))"
  python $D2/start_neurons.py --edit $OUT/p_units/edit.json --out $OUT/START.json 2>&1 | tail -1 | cut -c1-160 | sed "s/^/[run] /"
fi
wc -l $OUT/keep/keep_other.jsonl $OUT/keep/keep_held.jsonl 2>/dev/null | sed "s/^/[run] /"
# ---- 2. rounds 2 and 3 (our instruction-round pipeline, case mp on pr128)
[ -f cases/mp/case.json ] || python -m wsp.case init mp --profile qwen35-0.8b-think --parent $P/edit/edit.json \
  --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800 2>&1 | tail -1
mkdir -p mp/req mp3/req
[ -f mp/req/train.json ] || python derive/mkreq.py --case mp --out /workspace/work/mp/req/train.json --n-rep 420 --n-word 300 --n-tulu 300 --n-freq 540 2>&1 | tail -1 | cut -c1-200 | sed "s/^/[run] r2 /"
[ -f mp3/req/train.json ] || python derive/mkreq.py --case mp --out /workspace/work/mp3/req/train.json --seed 3 --n-rep 300 --rep-wt 1 --p-first 0.7 --n-word 0 --n-tulu 0 --n-freq 0 --n-lett 300 --n-long 150 2>&1 | tail -1 | cut -c1-200 | sed "s/^/[run] r3 /"
say "round 2 data start"
[ -f mp/branch/pairs.jsonl ] || python derive/pipeline.py --case mp --out /workspace/work/mp --skip C D --stop-after pairs > mp_pipeline.log 2>&1
grep -h "mended pairs" mp/logs/mend.log 2>/dev/null | cut -c1-240 | sed "s/^/[run] r2 /"
say "round 3 data start"
[ -f mp3/branch/pairs.jsonl ] || python derive/pipeline.py --case mp --out /workspace/work/mp3 --skip C D --stop-after pairs > mp3_pipeline.log 2>&1
grep -h "mended pairs" mp3/logs/mend.log 2>/dev/null | cut -c1-240 | sed "s/^/[run] r3 /"
[ -s mp/branch/mpairs.jsonl ] && [ -s mp3/branch/mpairs.jsonl ] || { say "no pairs"; tail -4 mp_pipeline.log mp3_pipeline.log | cut -c1-200; exit 1; }
# ---- 3. one keep-protected fit of both rounds
python - <<'PY'
import json, random
OUT = "/workspace/work/w2"
with open(f"{OUT}/items_r.jsonl", "w") as f:
    f.write(open(f"{OUT}/items_all.jsonl").read())
    for p in ("/workspace/work/mp/req/train.json", "/workspace/work/mp3/req/train.json"):
        for it in json.load(open(p)): f.write(json.dumps(it) + "\n")
held = [l for l in open(f"{OUT}/keep/keep_held.jsonl") if l.strip()]
random.Random(0).shuffle(held)
n = 0
with open(f"{OUT}/data_r.jsonl", "w") as f:
    for p in ("mp/branch/mpairs", "mp/branch/pairs", "mp3/branch/mpairs", "mp3/branch/pairs"):
        for l in open(f"/workspace/work/{p}.jsonl"):
            if l.strip(): f.write(l); n += 1
    f.write(open(f"{OUT}/keep/keep_other.jsonl").read()); f.write("".join(held[:900]))
print(f"[run] pairs of rounds 2 + 3: {n}; keep rows {sum(1 for _ in open(OUT + '/keep/keep_other.jsonl'))} + {min(900, len(held))} puzzles")
PY
export REQ_ITEMS=$OUT/items_r.jsonl
MP=$(python -c "from wsp import case; print(case.model_path('mp', 'base'))" | tail -1)
PREF=$(python -c "
import os
from wsp import case as C
print(os.path.relpath(C.cdir('mp') / C.conf('mp')['parent'], C.cdir('mp') / 'edits'))" | tail -1)
say "fit start (4 GPUs)"
[ -f $E/KR.json ] || torchrun --standalone --nproc_per_node 4 $D2/train_keep.py --keep 100 --model $MP --data $OUT/data_r.jsonl \
  --parent-ref $PREF --objective dpo --select margin --epochs 2 --accum 1 --max-len 20480 --rejected-window 20480 \
  --protect $OUT/keep/keep_other.jsonl --protect-n 20000 --protect-examples 256 --exclude-neurons-from $OUT/START.json \
  --out-spec $E/KR.json --seed 0 > $OUT/logs/fit_KR.log 2>&1
say "fit rc=$? $(grep -E 'val loss: start|wrote|examples:' $OUT/logs/fit_KR.log | tail -3 | cut -c1-200 | tr '\n' ' ')"
grep -E "Traceback|Error" -A3 $OUT/logs/fit_KR.log | tail -5 | cut -c1-200
[ -f $E/KR.json ] || { say "no KR.json"; exit 1; }
python $D2/dose_sparse.py --spec $E/KR.json --start $MP --k 0.5 --out $E/KRh.json 2>&1 | tail -1 | cut -c1-200
for v in KRh KR; do python $D2/encode_dirs.py --start $MP --spec $E/$v.json --out $E/${v}D.json 2>&1 | tail -1 | cut -c1-220 | sed "s/^/[run] $v /"; done
say "r128 done"
"""

if __name__ == "__main__":
    sys.exit(subprocess.run(["bash", "-c", SCRIPT]).returncode)
