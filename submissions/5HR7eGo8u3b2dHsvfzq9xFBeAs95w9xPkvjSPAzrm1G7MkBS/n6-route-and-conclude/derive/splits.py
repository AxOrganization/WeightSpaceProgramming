"""Frozen question-level practice splits; importing this module needs only Python.

No questions, answers, generations or model arrays are shipped here. A case's
runtime manifest records IDs and hashes, and is checked before any model work.
"""
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

SOURCES = ("math_train", "gsm8k_train", "mmlupro", "ifsynth")
SPLITS = ("fit", "dev", "holdout")
SCHEMA = "n6.practice-question-splits/1"
GROUPING = ("lowercase ASCII alphanumeric words, ignoring punctuation and whitespace; "
            "ifsynth uses its longest complete GSM8K-question prefix, because "
            "wsp.data.make_ifsynth appends constraints to the original question")


def practice_source(row):
    """Recognize only the four practice namespaces, including their train IDs."""
    if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
        raise ValueError("practice row must have a string id")
    pid = row["id"]
    for source in SOURCES:
        if pid.startswith(source + "_") and len(pid) > len(source) + 1:
            # The shipped train rows retain their original loader's source name.
            aliases = {"math_train": "math", "gsm8k_train": "gsm8k"}
            if "source" in row and row["source"] not in (source, aliases.get(source, source)):
                raise ValueError(f"practice source disagrees with id {pid}")
            if source == "mmlupro" and not re.fullmatch(r"mmlupro_[0-9]+", pid):
                raise ValueError(f"unsupported practice id {pid}; suggested-answer variants are excluded")
            return source
    raise ValueError(f"unsupported/non-practice id {pid}")


def question_keys(index):
    """Map each ID to a normalized original question, including GSM8K/IF pairs.

    The full four-source index is required: an unmatched synthetic IF item is an
    error, since treating its constraints as a new question could leak GSM8K.
    """
    if not isinstance(index, Mapping) or not index:
        raise ValueError("a nonempty four-source practice index is required")
    keys, sources = {}, {}
    for pid, row in index.items():
        source = practice_source(row)
        if not isinstance(pid, str) or pid != row["id"]:
            raise ValueError("practice index key must equal the row id")
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"invalid problem for {pid}")
        key = " ".join(re.findall(r"[a-z0-9]+", problem.lower()))
        if not key:
            raise ValueError(f"empty normalized question for {pid}")
        keys[pid], sources[pid] = key, source
    gsm = sorted({key for pid, key in keys.items() if sources[pid] == "gsm8k_train"},
                 key=lambda key: (-len(key), key))
    for pid, key in list(keys.items()):
        if sources[pid] == "ifsynth":
            match = next((q for q in gsm if key == q or key.startswith(q + " ")), None)
            if match is None:
                raise ValueError(f"{pid} has no underlying GSM8K practice question")
            keys[pid] = match
    return keys


def _sizes(sizes):
    if not isinstance(sizes, Mapping) or set(sizes) != set(SPLITS):
        raise ValueError("sizes must declare exactly fit, dev and holdout")
    result = {name: sizes[name] for name in SPLITS}
    if any(type(n) is not int or n < 1 for n in result.values()):
        raise ValueError("every per-source split size must be a positive integer")
    return result


def split_questions(index, seed, sizes):
    """Select exact per-source counts deterministically, reserving whole questions.

    Source order favors scarce question pools. All three groups of a scarce
    source are assigned before abundant sources can consume their questions;
    already assigned same-group questions are reused before reserving new ones.
    Only one ID per original question and source is selected.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    sizes, keys = _sizes(sizes), question_keys(index)
    pools = {source: {} for source in SOURCES}
    for pid in sorted(index):
        pools[practice_source(index[pid])].setdefault(keys[pid], []).append(pid)
    total = sum(sizes.values())
    for source in SOURCES:
        if len(pools[source]) < total:
            raise ValueError(f"not enough distinct {source} questions: need {total}, have {len(pools[source])}")
    assignment, selected = {}, {name: {source: [] for source in SOURCES} for name in SPLITS}
    priority = sorted(SOURCES, key=lambda source: (len(pools[source]), SOURCES.index(source)))
    for source in priority:
        for name in SPLITS:
            reusable = sorted(q for q in pools[source] if assignment.get(q) == name)
            unused = sorted(q for q in pools[source] if q not in assignment)
            rng = random.Random(f"n6-question-splits-v1:{seed}:{source}:{name}")
            rng.shuffle(reusable)
            rng.shuffle(unused)
            candidates = reusable + unused
            if len(candidates) < sizes[name]:
                raise ValueError(f"not enough question-disjoint {source} questions for {name}")
            for question in candidates[:sizes[name]]:
                assignment[question] = name
                selected[name][source].append(pools[source][question][0])
    result = {name: [pid for source in SOURCES for pid in selected[name][source]] for name in SPLITS}
    validate_splits(result, index)
    return result


def validate_splits(splits, index):
    """Reject aliases, missing IDs, duplicates, imbalance, or cross-group questions."""
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("declare exactly the fixed fit, dev and holdout groups")
    keys, assigned = question_keys(index), {}
    for name in SPLITS:
        ids = splits[name]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"{name} must be a nonempty list of practice ids")
        if any(not isinstance(pid, str) or pid not in index for pid in ids):
            raise ValueError(f"{name} contains a missing/non-practice id")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} contains duplicate ids")
        counts = Counter(practice_source(index[pid]) for pid in ids)
        if set(counts) != set(SOURCES) or len(set(counts.values())) != 1:
            raise ValueError(f"{name} must have equal positive counts for all four practice sources")
        for pid in ids:
            previous = assigned.setdefault(keys[pid], name)
            if previous != name:
                raise ValueError(f"{name} overlaps {previous} by original question at {pid}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def split_manifest(splits, index, seed, sizes):
    """Build an ID/hash-only manifest for runtime case creation and revalidation."""
    sizes = _sizes(sizes)
    validate_splits(splits, index)
    if splits != split_questions(index, seed, sizes):
        raise ValueError("splits differ from the declared seed and per-source sizes")
    keys = question_keys(index)
    dataset = {pid: {"source": practice_source(index[pid]), "row_sha256": _digest(index[pid])}
               for pid in sorted(index)}
    return {"schema": SCHEMA, "seed": seed, "sources": list(SOURCES), "sizes": sizes,
            "grouping": GROUPING, "dataset_sha256": _digest(dataset),
            "counts": {name: len(splits[name]) for name in SPLITS},
            "splits": {name: {"ids": list(splits[name]),
                       "question_sha256": {pid: hashlib.sha256(keys[pid].encode("utf-8")).hexdigest()
                                           for pid in splits[name]},
                       "rows_sha256": {pid: dataset[pid]["row_sha256"] for pid in splits[name]}}
                       for name in SPLITS}}


def load_splits(case_dir, index):
    """Read a prepared case and reject legacy, changed or re-labeled splits.

    The case binds the manifest hash; the manifest binds source data, IDs and
    normalized questions. Re-derivation also checks its declared seed and sizes.
    """
    root = Path(case_dir)
    config = json.loads((root / "case.json").read_text(encoding="utf-8"))
    path = root / "diagnostics" / "split.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("split_schema") != SCHEMA
            or config.get("split_manifest_sha256") != _digest(manifest)):
        raise ValueError("case has no valid frozen split manifest; prepare a new N6 case")
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or config.get("practice") != list(SOURCES)):
        raise ValueError("case does not declare the four-source question split schema")
    if config.get("subsets") != {}:
        raise ValueError("fixed N6 splits do not permit subset aliases")
    if {p.name for p in (root / "sets").iterdir()} != {name + ".json" for name in SPLITS}:
        raise ValueError("sets must contain exactly fit.json, dev.json and holdout.json")
    splits = {name: json.loads((root / "sets" / (name + ".json")).read_text(encoding="utf-8"))
              for name in SPLITS}
    validate_splits(splits, index)
    expected = split_manifest(splits, index, manifest.get("seed"), manifest.get("sizes"))
    if manifest != expected:
        raise ValueError("split manifest or practice data changed; prepare a new N6 case")
    return splits


def validate_rows(rows, allowed_ids, label):
    """Keep every generation sample; fail closed on any invalid or foreign row."""
    allowed = set(allowed_ids)
    if not allowed or any(not isinstance(pid, str) or not pid for pid in allowed):
        raise ValueError(f"{label}: allowed ids must be nonempty practice ids")
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise ValueError(f"{label}: row {i} has no valid id")
        if row["id"] not in allowed:
            raise ValueError(f"{label}: row {i} id {row['id']} is outside the declared group")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{label}: row {i} has empty or invalid text")
        if "k" in row and (type(row["k"]) is not int or row["k"] < 0):
            raise ValueError(f"{label}: row {i} has an invalid sample index k")
        result.append(row)
    if not result:
        raise ValueError(f"{label}: no generation rows")
    return result
