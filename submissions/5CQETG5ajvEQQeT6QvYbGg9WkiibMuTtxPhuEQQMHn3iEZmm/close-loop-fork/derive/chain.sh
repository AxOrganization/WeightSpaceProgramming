#!/bin/bash
# The whole GPU work of this method in one detached, resumable chain (run from the work directory).
#   bash submission/derive/chain.sh start          start or resume the chain in its own session; returns at once
#   bash submission/derive/chain.sh status [MIN]   last status lines, steps done, uploads, minutes left
#                                                  (with MIN: first wait up to MIN minutes for a new status line)
#   bash submission/derive/chain.sh stop           stop the chain (an upload in progress is finished first)
#   bash submission/derive/chain.sh upload NAME    package, check and upload one edit of the case (CPU, a few minutes)
# Every step writes state/<step>.done when it has finished; `start` again skips the steps that are done.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
W=${WORK:-$(pwd)}
CASE=${CHAIN_CASE:-k1}
GPUS=${CHAIN_GPUS:-0,1,2,3}   # four GPUs for every generation; the first alone for the directions
# ---- the recipe, fixed before the run (METHOD.md sections 1-3)
CANDS="AB AB1"                # conclude at 0.75 / at 1 (loop at 1 in both); screened after the start, in this order
L=$W/logs; S=$W/state; RD=$W/runs
mkdir -p "$L" "$S" "$RD"
cd "$W"
if ! command -v python > /dev/null; then  # an image with python3 only
  mkdir -p "$W/.bin" && ln -sf "$(command -v python3)" "$W/.bin/python" && export PATH="$W/.bin:$PATH"
fi

deadline(){
  if [ -s "$S/deadline" ]; then cat "$S/deadline"; return; fi
  local v="" t
  [ -n "${DL:-}" ] && v=$DL
  if [ -z "$v" ] && [ -r /etc/axonry-deadline ]; then
    t=$(head -1 /etc/axonry-deadline | tr -d '[:space:]')
    if [[ "$t" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then v=${t%%.*}; else v=$(date -d "$(head -1 /etc/axonry-deadline)" +%s 2>/dev/null); fi
  fi
  [ -z "$v" ] && v=$(( $(date +%s) + 5 * 3600 + 1800 ))   # no deadline found: assume 5.5 h from the first call
  echo "$v" > "$S/deadline"; echo "$v"
}
left(){ echo $(( ($(deadline) - $(date +%s)) / 60 )); }
st(){ echo "[$(date -u +%H:%M:%S) L=$(left)] $*" >> "$L/status.log"; }
done_(){ [ -f "$S/$1.done" ]; }
mark(){ touch "$S/$1.done"; }
stopped(){ [ -f "$S/STOP" ]; }
mp(){ python -c "from wsp import case; print(case.model_path('$CASE', '$1'))" 2>>"$L/build_$1.log" | tail -1; }
gen(){  # gen MODELNAME SET N SALT MAXT  -> runs/MODELNAME__SET.jsonl (one retry if a generation stops)
  local m=$1 s=$2 n=$3 salt=$4 maxt=$5 out=$RD/$1__$2.jsonl path
  [ -s "$out" ] && return 0
  path=$(mp "$m") || return 1
  for try in 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUS timeout 2400 python -m wsp.gen --profile qwen35-0.8b-think --model "$path" \
        --ids "$W/sets/$s.json" --n "$n" --salt "$salt" --max-tokens "$maxt" --out "$out.tmp" > "$L/gen_${m}_$s.log" 2>&1 \
      && mv "$out.tmp" "$out" && return 0
    st "generation $m on $s stopped (try $try): $(tail -1 "$L/gen_${m}_$s.log" | cut -c1-160)"
  done
  return 1
}
stats(){ python - "$@" <<'PY'
import json, sys, collections
c = collections.defaultdict(lambda: [0, 0, 0])
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l); s = c[r["source"]]; s[0] += 1; s[1] += bool(r["correct"]) and not r["cut"]; s[2] += bool(r["cut"])
print(" ".join(f"{k}: n {v[0]}, acc {100*v[1]/v[0]:.1f}%, cut {100*v[2]/v[0]:.1f}%" for k, v in sorted(c.items())))
PY
}

auto_report(){  # the report uploaded until REPORT.md exists: the method, every status line, the screen table
  { echo "# Report (automatic, from derive/chain.sh while the run goes on)"; echo
    echo "Method: submission/METHOD.md. Status lines:"; echo; echo '```'; cat "$L/status.log"; echo '```'; echo
    echo "Paired screens against the start (same problems and seeds), points:"; echo; echo '```'
    python "$D/pair.py" table "$RD" $CANDS 2>/dev/null; echo '```'
    [ -f "$RD/choice.json" ] && { echo; echo "Choice: $(cat "$RD/choice.json")"; }
  } > "$W/REPORT.auto.md"
}
do_upload(){  # do_upload NAME: package, axonry-check, upload; one upload at a time
  local n=$1 rpt start_arg wk pkg
  exec 9>"$S/upload.lock"; flock 9
  if [ "$(left)" -lt 8 ]; then st "upload $n not started: fewer than 8 minutes left"; flock -u 9; return 1; fi
  auto_report
  rpt=$W/REPORT.md; [ -s "$rpt" ] || rpt=$W/REPORT.auto.md
  st "uploading $n"
  # the toolkit names the start by its sid when the case keeps it under that name (<sid>/edit/); toolkits of other
  # versions want --start <sid> or --start parent. A package is used only if it references the start (its "parent" is
  # the sid, or parent/ in the older layout), never one that carries the start's whole chain from the base.
  local sid ok="" par; sid=$(python "$D/start.py" --sid); : > "$L/upload_$n.log"
  wk=$(python -c "from wsp.common import WORK; print(WORK)")
  for start_arg in "" "--start $sid" "--start parent"; do
    python -m wsp.upload "$CASE" "$n" --report "$rpt" --task "$W/task.json" $start_arg --dry-run >> "$L/upload_$n.log" 2>&1 || continue
    par=$(python -c "import json; print(json.load(open('$wk/deliver/${CASE}__$n/edit/edit.json'))['parent'])" 2>/dev/null)
    if [ "$par" = "$sid" ] || [[ "$par" == parent* ]]; then ok=1; break; fi
    echo "package with '$start_arg' references '$par', not the start: not used" >> "$L/upload_$n.log"
  done
  if [ -z "$ok" ]; then
    st "upload $n: no package references the start (logs/upload_$n.log): nothing uploaded"; flock -u 9; return 1
  fi
  wk=$(python -c "from wsp.common import WORK; print(WORK)")
  pkg=$wk/deliver/${CASE}__$n/edit
  if command -v axonry-check > /dev/null; then
    if ! axonry-check "$pkg" --task "$W/task.json" >> "$L/upload_$n.log" 2>&1; then
      st "upload $n: axonry-check refused the package (logs/upload_$n.log)"; flock -u 9; return 1
    fi
  fi
  if python -m wsp.upload "$CASE" "$n" --report "$rpt" --task "$W/task.json" $start_arg >> "$L/upload_$n.log" 2>&1; then
    echo "$n" > "$S/uploaded.txt"; st "uploaded $n"; flock -u 9; return 0
  fi
  st "upload $n failed: $(tail -1 "$L/upload_$n.log" | cut -c1-200)"; flock -u 9; return 1
}

run(){
  st "chain start: case $CASE, GPUs $GPUS, candidates $CANDS"
  if ! done_ init; then
    if [ ! -f "$(python -c "from wsp import case; print(case.cdir('$CASE'))")/case.json" ]; then
      python -m wsp.case init $CASE --profile qwen35-0.8b-think --parent "$(python "$D/start.py" "$W")" \
          --practice gsm8k_train math_train mmlupro ifsynth > "$L/init.log" 2>&1 || { st "FAILED init (logs/init.log)"; exit 1; }
    fi
    mp base > /dev/null || { st "FAILED building the start (logs/build_base.log)"; exit 1; }
    python "$D/sets.py" $CASE "$W/sets" > "$L/sets.log" 2>&1 || { st "FAILED sets"; exit 1; }
    mark init; st "1 case and sets: $(tr '\n' ';' < "$L/sets.log")"
  fi
  stopped && exit 0
  if ! done_ pool; then
    gen base pool 4 pool 12000 || { st "FAILED pool generation"; exit 1; }
    mark pool; st "2 pool: the start's answers, $(stats "$RD/base__pool.jsonl")"
  fi
  stopped && exit 0
  E=$(python -c "from wsp import case; print(case.cdir('$CASE'))")/edits
  if ! done_ layers; then
    if [ ! -f "$E/AB.json" ]; then
      CUDA_VISIBLE_DEVICES=${GPUS%%,*} python "$D/dirs.py" $CASE --files "$RD/base__pool.jsonl" --name AB > "$L/dirs.log" 2>&1
    fi
    if [ -f "$E/AB.json" ]; then
      st "3 layer A·B: $(grep -E 'pairs from|cosine|left out|dir_ablate ops' "$L/dirs.log" | tr '\n' ';' | cut -c1-500)"
    else
      st "FAILED: layer A·B could not be measured: $(tail -1 "$L/dirs.log" | cut -c1-200)"; exit 1
    fi
    for x in $CANDS; do [ -f "$E/$x.json" ] && { mp $x > "$S/model_$x.txt" & }; done; wait
    BUILT=""; for x in $CANDS; do [ -s "$S/model_$x.txt" ] && BUILT="$BUILT $x"; done
    [ -z "$BUILT" ] && { st "FAILED: no candidate edit was built (logs/build_*.log)"; exit 1; }
    echo $BUILT > "$S/built.txt"; mark layers; st "4 built:$BUILT"
  fi
  BUILT=$(cat "$S/built.txt")
  stopped && exit 0
  first=${BUILT%% *}
  # the first candidate (fewest ops) goes up at once, so a complete upload is in place (tried again before each screen)
  done_ interim || { do_upload $first && mark interim; }
  for m in base $BUILT; do
    stopped && exit 0
    done_ interim || { do_upload $first && mark interim; }
    done_ "scr_$m" && continue
    if [ "$(left)" -lt 60 ]; then st "5 screen of $m skipped: $(left) minutes left"; continue; fi
    gen $m scr_g 4 scrg 32768 && gen $m scr_m 2 scrm 16384 && gen $m guard 2 guard 32768 \
      && { mark "scr_$m"; st "5 screen $m: $(stats "$RD/${m}__scr_g.jsonl" "$RD/${m}__scr_m.jsonl" "$RD/${m}__guard.jsonl")"; } \
      || st "5 screen $m incomplete"
  done
  python "$D/pair.py" table "$RD" $BUILT > "$L/screen_table.log" 2>&1
  st "6 screens (gain +- SE vs the start):"; sed 's/^/    /' "$L/screen_table.log" >> "$L/status.log"
  if ! done_ choice; then
    if python "$D/pair.py" choose "$RD" $BUILT > "$L/choice.log" 2>&1; then
      ch=$(python -c "import json; print(json.load(open('$RD/choice.json'))['choice'])")
      st "7 choice: $(cat "$L/choice.log")"
      if [ "$ch" != "$(cat "$S/uploaded.txt" 2>/dev/null)" ]; then do_upload "$ch" || do_upload "$(cat "$S/uploaded.txt" 2>/dev/null)"; fi
      echo "$ch" > "$S/final.txt"; mark choice
    else
      st "7 no choice possible ($(tail -1 "$L/choice.log")): the interim upload stays"
      cp "$S/uploaded.txt" "$S/final.txt" 2>/dev/null; mark choice
    fi
  fi
  auto_report
  st "DONE: final $(cat "$S/final.txt" 2>/dev/null), uploaded $(cat "$S/uploaded.txt" 2>/dev/null)"
}

running(){ [ -f "$S/chain.pid" ] && kill -0 "$(cat "$S/chain.pid")" 2>/dev/null; }
case "${1:-status}" in
  start)
    deadline > /dev/null
    if running; then echo "chain already running (pid $(cat "$S/chain.pid")): nothing started"; exit 0; fi
    rm -f "$S/STOP"
    setsid nohup bash "$0" run > "$L/chain.log" 2>&1 < /dev/null & echo $! > "$S/chain.pid"
    sleep 2; echo "chain started (pid $(cat "$S/chain.pid")); deadline $(date -u -d @"$(deadline)" +%H:%M) UTC, $(left) minutes left";;
  run) run;;
  status)
    if [ -n "${2:-}" ]; then
      n0=$(wc -l < "$L/status.log" 2>/dev/null || echo 0); end=$(( $(date +%s) + 60 * $2 ))
      while [ "$(date +%s)" -lt $end ] && [ "$(wc -l < "$L/status.log" 2>/dev/null || echo 0)" -eq "$n0" ]; do sleep 15; done
    fi
    tail -14 "$L/status.log" 2>/dev/null
    echo "steps done: $(cd "$S" && ls *.done 2>/dev/null | sed 's/.done//' | tr '\n' ' ')"
    echo "uploaded: $(cat "$S/uploaded.txt" 2>/dev/null || echo none)   final: $(cat "$S/final.txt" 2>/dev/null || echo -)"
    [ -f "$S/upload.lock" ] && ! flock -n "$S/upload.lock" true && echo "uploading now"
    running && echo "chain running, L=$(left) minutes left" || echo "chain NOT running, L=$(left) minutes left";;
  stop)
    touch "$S/STOP"; exec 9>"$S/upload.lock"; flock 9
    running && kill -- -"$(cat "$S/chain.pid")" 2>/dev/null  # the chain's own session: its generations go with it
    echo "chain stopped";;
  upload) do_upload "${2:?edit name}";;
  *) echo "usage: chain.sh start|status [MIN]|stop|upload NAME"; exit 2;;
esac
