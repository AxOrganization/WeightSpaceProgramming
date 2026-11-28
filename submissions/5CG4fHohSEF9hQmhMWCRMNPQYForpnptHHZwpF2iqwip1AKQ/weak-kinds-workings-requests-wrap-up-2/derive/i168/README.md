`derive/inst/` of the finished entry #168 "workings-and-requests-on-164" (0009588957-0032): its instruction stage (#168's requests of
its own checkable types, answers, branches after `</think>`, preference fits of unit / head gains with a keep term, export, upload).
Changes by `0009599139-0015` (#179), whose copy this is: the profile from `WSP_PROFILE` (run_i.py, gen1.py, branch2.py, fit_gains.py); `I_TRAIN`, `I_TOK`, `I_SEEDS` and
`PAYLOAD_MAX` from the environment (defaults as published: 1000, 12288, 51-54, 4194304; the driver sets 600, 8192, 51-52 and the
2B limit 8388608); with `FW_WAIT=1` the edit step waits for the puzzle chain's Fw (case pz) while time allows and folds it in, or
exports I alone on the start when Fw is not there (the published step required F); with no start in task.json the local `parent/edit/` (stage A) is the start.
Changes by this entry: in `run_i.py`, the reading of the held-out requests after the upload runs the start's reading and then FI's on each of the two GPUs, waiting for the GPU's memory to be free between them (with two GPUs the four readings at once put two engines on each GPU, and FI's could not start); in `fp_filter.py`, its docstring says "the stage's" requests. First-person words inside these files are their authors' (#168, #179), not this entry's.
