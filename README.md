# WeightSpaceProgramming — improving LLMs by editing their weights directly

Improve small LLMs on benchmarks by editing their weights directly, in the spirit of *Sparks of Weight-Space
Programming* (Takehi, Teoh, Zhang, Khattab, 2026, https://rikiyat.github.io/blog/weight-space-programming/): find
what blocks the model (e.g. reasoning loops), localize the components that cause it, and change a few weights.
The editor is usually an LLM agent following the `weight-edit` skill (`.claude/skills/weight-edit/SKILL.md`), which
starts from `playbook.md` (what works, with the evidence). The repository is also where competition entries are
submitted ("Submitting").

## Layout
```
wsp/            toolkit (python -m wsp.<module> -h for each CLI)
  case.py       case workflow: screen, init, split, adopt, run, branch, pairs, extract, selectivity, edit, compare, freeze,
                report
  fit.py        edits fitted on the model's own samples (passing answers, advantages, passing vs failing pairs)
  profiles.py   base models (HF repo pinned to a revision) and their model-card sampling
  data.py       practice sets (no benchmarks)              grade.py  graders, failure kinds
  puzzles.py    generated reasoning puzzles (practice:bbhsynth)
  gen.py        vLLM generation with per-problem seeds      stats.py  summaries, paired comparisons
  prefix_penalty.py  presence penalty over a continued start (branches, primes)
  probe.py      teacher-forced activation / log-prob reads  directions.py  contrast pairs, directions, writers
  edits.py      declarative edit specs -> model dir + manifest (stacking on a parent)
  upload.py     deliverables -> R2 by the upload protocol   rules.py  loads rules.json, renders "Protocol" below
.claude/skills/weight-edit/SKILL.md   the agent's workflow (Claude; Codex reads it via the .agents/skills symlink)
playbook.md     lessons + evidence
rules.json      the competition protocol                      data/practice/  practice sets (none is a benchmark item)
cases/          worked cases (records protocol: wsp/case.py)
tests/          CPU unit tests                                submissions/    champions' entries (from the first one)
tools/          run the toolkit on a remote GPU box (tools/gpu; settings in tools/.config, not committed)
```

## Install and environment
`pip install -e ".[gen,grade,upload]"` (plus `fast` for linear-attention models, `test` for the tests); install
`vllm` first if a resolver picks an old version when solving all extras at once.
- `WSP_MODELS_DIR`: pre-downloaded base models as `<org>__<name>@<revision>/`; otherwise the pinned revision is
  fetched from the HF hub on first use.
- `WSP_WORK` (default `.work/`): scratch (built models, generations, pairs; benchmark items when supplied). `WSP_CASES` (default
  `cases/`): case records, e.g. outside a read-only checkout.

## Workflow
`python -m wsp.case`, step by step as in the skill. Edits stack: `init <case> --parent <edit.json>` starts from an
existing edit and the case's edits inherit it. Deliver with `python -m wsp.upload <case> <edit> [--task task.json]`.

## License
Code: MIT (`LICENSE`). Practice data comes from GSM8K, MATH and MMLU-Pro (each MIT); `ifsynth` adds
IFEval-style constraints to GSM8K train questions; `tuluif` is 1348 prompts of
[allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) (Tulu 3, ODC-BY, revision 47c03c7), kept
where the constraint means exactly what an official checker tests and the request shares no 8-word span with the
evaluation; `bbhsynth` is reasoning puzzles of BIG-Bench Hard's kinds, written and solved by `wsp/puzzles.py`;
`mmlucf` is 2048 questions of the validation split of
[microsoft/MMLU-CF](https://huggingface.co/datasets/microsoft/MMLU-CF) (CDLA-Permissive-2.0, text in
`data/practice/LICENSE-CDLA-Permissive-2.0.txt`; revision c25b89a), none sharing an 8-word span with MMLU-Pro;
`musr` is all of [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) (Sprague et al. 2024, CC-BY-4.0, revision
7c365b4).

## Repository protocol
Everything here should raise what an editing agent can do; the rest stays out, so the repository does not grow
with use.
- **Code** lives only in `wsp/`, one module per concern; no scripts beside it. Tests cover what other parts rely on
  (exact rebuilds, protocol, grading, statistics).
- **Knowledge** goes into `playbook.md` (merged, not appended) and the skill. A **case** is added only when it
  teaches something new (a base model, a failure type, a method) and keeps only its records (`wsp/case.py`):
  config, splits, edit specs, result summaries, decisions. Runs, pairs, directions and built models are scratch.
  A case enters git once it has a conclusion (a frozen edit, or a negative result merged into the playbook).
- **Generated views**: the README protocol section and the playbook evidence table; nothing generated is edited
  by hand, and other views (`cases/*/RESULTS.md`) are not kept.
- **Data**: only the frozen practice sets. The toolkit holds no benchmark and no loader for one: `bench:<name>` reads
  the items the organizer supplies (`$WSP_WORK/data/bench_<name>.jsonl`).
- **Submissions**: only champions' directories are merged. Once what an entry's code can do for others is in
  `wsp/` (or is not general), its `derive/` and `data/` are removed; its `entry.json` and `METHOD.md` stay, and the
  pull request keeps the original.
- **Tooling** that is generic and configuration-driven lives in `tools/`; credentials and machine settings
  (`tools/.config`) are never committed.

## Submitting
Taking part (registration, on-chain commitment, fees, review, runs and decisions) and the rules in force (base
model, benchmarks, thresholds, the published entries one may name as `parent`) are published by the subnet project,
[axonry](https://github.com/AxOrganization/axonry). This repository fixes what an entry and its edit are ("Protocol"
below):
- A pull request adds exactly one directory `submissions/<hotkey_ss58>/<entry>/` and touches nothing else; only
  champions' directories are merged into `master`.
- Reusable tools and lessons from entries are folded into the toolkit and `playbook.md` by the maintainers from time
  to time, with their source. Report toolkit problems as issues.

## Protocol
<!-- protocol:begin -->
Generated from `rules.json` (`python -m wsp.rules doc`); a competition pins this repository's commit and the file's sha256. Protocol of the weight-editing competition. Checks, admission, evaluation and ranking are done by the subnet project, which reads this file.

**Submission** `submissions/<hotkey_ss58>/<entry>/`. A PR adds exactly one directory and touches nothing else. It contributes an idea for improving the model, grounded in the author's own exploration: what fails, why the model does it, and how to find and test an edit that fixes it. The agent carries the idea out on the base model or an inherited edit, derives the edit and uploads it. What that exploration measured and observed may explain the idea; a directory that only hands over an edit, with no idea for the agent to explore and test, is not a method. The edit is derived by the agent during the run from practice data only: nothing in the directory may let the result come from elsewhere or aim at the evaluation, and nothing about the evaluation (which sets, their items, answers or results) reaches the agent. Never weights.
- `entry.json` (required): {"schema": "axonry.entry/1", "base": <profile name in wsp/profiles.py>, "parent": <sid of a published submission whose edit to inherit, "champion" (the champion when the run starts), or null>}
- `METHOD.md` (required): the idea for the agent: the failure to remove, why the model shows it, how to find and test the edit, what to expect
- `derive/`: method code (Python) the agent runs, using the wsp toolkit
- `data/`: small auxiliary text the method needs (prompts, contrast texts, word lists), readable as it stands; numeric arrays are computed during the run by derive/, never shipped
- file types .json, .jsonl, .py, .md, .txt, .csv; at most 1024 KiB in total

**Edit.** An edit spec (wsp/edits.py) of whitelisted ops, deterministic functions of the starting weights and the spec's arrays; its base equals entry.json's base. The subnet builds the model from it.
- no overfitting: The edit must improve the model's behaviour, not fit the benchmarks: it is derived without the evaluation benchmarks' items, answers or near-duplicates, from the toolkit's practice sets, the submission's data/ and the model's own generations, and its gain must carry over to unseen problems of the same kind; an edit whose gain rests on the benchmarks' specific items is disqualified. Every shipped practice set is allowed: none shares a question with the evaluation (data/bench_fingerprints.txt.gz). Other sets built from the evaluation's sources count as near-duplicates. Normative: the checks enforce only the limits.
- inheritance: The task holds each start it offers in <sid>/edit/ (task.json -> starts: the entry's parent, if any, and the champion at the start of the run, when it differs); the agent starts from one of them or from the base. A spec always says where it starts, in "parent": the sid of one of the task's starts, stacking on that published edit and starting from exactly that model's weights, or the pinned base as "<hf_repo>@<revision>" (wsp/profiles.py; task.json -> base_name); anything else, or no "parent", is refused. The start is not uploaded (it is already published): to build, the subnet places the published chain where each spec points, each published edit/ directory as <sid>/edit/ next to the spec that names it. The limits apply to the spec's own ops, measured against the parent model; neither the accumulated change nor the chain length is limited.
- editable tensors: 2-D weight matrices inside the main decoder layers (not auxiliary stacks such as MTP heads or vision towers); never embeddings, output head, norms, biases or other parameters
- ops: `dir_ablate`, `scale_cols`, `scale_rows`, `set_rows`, `set_cols`, `round`
- payload: bytes of every distinct array the own ops carry (wsp.edits.spec_size): the information budget, the main guard against importing outside weights or memorized answers. Its limit grows with the base's width, so that a base of any size admits the same number of rows, columns or directions: max_payload_bytes_per_hidden x the hidden size of the base (wsp.rules.payload_limit; Qwen3.5-0.8B 4 MiB, Qwen3.5-2B 8 MiB)
- relative change: per tensor: manifest relative_change_vs_parent; global: sqrt(sum (relative_change_vs_parent * parent_norm)^2) / norm of all editable matrices of the base
- **limits**: payload bytes per hidden <= 4096 bytes x the base's hidden size, relative change per tensor <= 50%, global relative change <= 20%

**Upload** (`axonry.upload/1`). When the task is finished, the deliverables go to the task's R2 prefix, files first and manifest last (python -m wsp.upload); nothing else is uploaded. The last complete manifest before the deadline counts (complete = every listed file present with matching sha256).
- `edit/edit.json` (required): the edit spec: {"base", "parent" (always: the sid of one of the task's starts, or the base as "<hf_repo>@<revision>"), "ops"}
- `edit/edit.npz`: the arrays the ops reference, when they reference any
- `edit/REPORT.md` (required): what was found, what was changed and why, controls, what did not work
- manifest.json = {"schema": "axonry.upload/1", "files": [{"path", "size", "sha256"}]}, paths relative to the prefix and starting with edit/
<!-- protocol:end -->
