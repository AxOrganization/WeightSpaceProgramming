# derive/ — commands, in order (sandbox; WSP toolkit at /workspace/WeightSpaceProgramming)

Setup: `export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/scratch PYTHONPATH=/workspace/WeightSpaceProgramming`,
`C=doubt D=submission/derive`. Run each GPU stage detached (`tmux`/`nohup`), one process per GPU. Every stage skips
work already on disk. Log each finished stage in /workspace/work/PROGRESS.md.

| # | Stage | Command | GPUs | Time on 4090 |
|---|---|---|---|---|
| 0 | Case on the parent | `python -m wsp.case init $C --profile qwen35-0.8b-think --parent parent/edit/edit.json --subset holdout256=holdout:256 --practice gsm8k_train math_train mmlupro ifsynth` | - | 1 min |
| 1 | Parent answers on practice DEV | `python $D/onset.py split $C`, then `python -m wsp.case run $C base dev_<g>` with `CUDA_VISIBLE_DEVICES=<g>`, g = 0..3 | 4 | 10-15 min |
| 2 | Profile | read `cases/$C/results/base__dev_*.json`: cut share, `kinds`, acc when finished. Stop if the failure is absent (METHOD.md) | - | 1 min |
| 3 | Doubt points | `python $D/doubt.py find $C --window 30` | - | 1 min |
| 4 | Branch | `CUDA_VISIBLE_DEVICES=0 python $D/doubt.py branch $C` | 1 | 5-10 min |
| 5 | Pairs | `python $D/doubt.py pairs $C`; normal text: `python -m wsp.case pairs $C --sets dev_0 dev_1 dev_2 dev_3 --out pairs_normal` | - | 1 min |
| 6 | Direction | `python $D/funnel.py extract $C --pairs pairs_doubt --chars 0 --tag doubt` (report the stability per layer) | 1 | 5 min |
| 7 | T0 grid | per GPU g: `python $D/funnel.py grid $C --tag doubt --heldout pairs_doubt_heldout --targets o_proj --layers 15-19 11-15-19 7-11-15-19 15 19 --topk 0 --alpha 1 2 3 --min-stab 0.6 --shard g --shards 4` | 4 | 20-30 min |
| 8 | T2 real answers | the two KEEP edits with the most negative d_bad - d_good (`funnel.py pick`-style order, no other choice): `python -m wsp.case run $C <edit> holdout256 --n 2`, split over GPUs like stage 1, plus `run $C base holdout256 --n 2` | 4 | 60-80 min |
| 9 | Upload | best paired gain (math up, no type down beyond noise): `python -m wsp.case freeze $C <edit>`, REPORT.md, `axonry-check`, `python -m wsp.upload $C <edit> --task task.json` | - | 5 min |

Fallback (stage 6 has no layer >= 0.6, or no KEEP at stage 7): report it, upload nothing.
