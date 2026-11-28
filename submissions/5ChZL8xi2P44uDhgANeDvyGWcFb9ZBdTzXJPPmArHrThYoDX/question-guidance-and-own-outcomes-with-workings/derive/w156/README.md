`derive/` of the finished entry #156 "workings-and-maths-recall-on-130" (0009580621-0022), in the copy #168 (0009588957-0032) ran.
Changes (ours): `chain.py` takes `W_ONLY=1` (skips the maths layer R and the start's check reads; its `edits` step then makes WD and
Fw only); `branch.py`, `rollouts.py`, `train_sparse.py` take their default profile from `WSP_PROFILE` (here qwen35-2b-think)
instead of the 0.8B constant; `train_sparse.py` reads the payload limit as the toolkit now states it (`max_payload_bytes_per_hidden` x the base's width, the old `max_payload_bytes` when present); `chain.py`: with `W_ONLY=1` the neuron choice and the fit get every GPU of the chain, and the keep answers' token caps can be set by `KEEP_CAPS` / `KEEP_CAP`. Nothing else is changed.
