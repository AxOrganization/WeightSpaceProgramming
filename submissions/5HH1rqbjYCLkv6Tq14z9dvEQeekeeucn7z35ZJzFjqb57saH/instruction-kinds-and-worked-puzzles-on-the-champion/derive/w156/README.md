`derive/` of the finished entry #156 "workings-and-maths-recall-on-130" (0009580621-0022), in the copy #168 (0009588957-0032) ran.
Changes made by the published `0009599139-0015` (workings-and-requests-on-173), whose `derive/w156/` this is: `chain.py` takes `W_ONLY=1` (skips the maths layer R and the start's check reads; its `edits` step then makes WD and
Fw only); `branch.py`, `rollouts.py`, `train_sparse.py` take their default profile from `WSP_PROFILE` (here qwen35-2b-think)
instead of the 0.8B constant; `train_sparse.py` reads the payload limit as the toolkit now states it (`max_payload_bytes_per_hidden` x the base's width, the old `max_payload_bytes` when present); `chain.py`: with `W_ONLY=1` the neuron choice and the fit get every GPU of the chain, and the keep answers' token caps can be set by `KEEP_CAPS` / `KEEP_CAP`. Nothing else is changed.

Changes made by this entry, and only these: in `chain.py`, `WORKED` / `HELD` move six kinds (`hyperbaton`, `object_counting`, `web_of_lies`,
`date_understanding`, `boolean_expressions`, `formal_fallacies`) from the held kinds to the kinds that get a working (their writers are
the published ones in `workings.py`, unchanged), and the comment on `HELD` says seven kinds; the three code comments that `0009599139-0015` marked "(ours)" (two in `chain.py`, one in
`train_sparse.py`) now name that entry instead. Every other file is byte-identical to `0009599139-0015`'s `derive/w156/`.
