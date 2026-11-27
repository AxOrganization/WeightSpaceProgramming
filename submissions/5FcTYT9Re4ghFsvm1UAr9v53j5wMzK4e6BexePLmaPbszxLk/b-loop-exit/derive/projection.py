"""Fit a loop direction while preserving the direction of successful checking."""

import numpy as np


MAX_COLUMNS = 64
HALF_COSINE = 0.25
CONTROL_RATIO = 0.5
MIN_REMAINDER = 32 * np.finfo(np.float64).eps


def _real_array(values, ndim, name):
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "fiu" or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D real array")
    return array


def _unit(vector, minimum=0.0):
    # Scale first so that a finite vector does not overflow when squared.
    scale = np.max(np.abs(vector))
    if not np.isfinite(scale) or scale <= minimum:
        return None
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _against_check(vector, check):
    raw = _unit(vector)
    if raw is None:
        return None, None
    remainder = raw - np.dot(raw, check) * check
    return raw, _unit(remainder, MIN_REMAINDER)


def _writer_columns(loop, controls, weights, direction):
    alignment = direction @ weights
    aligned_loop = np.mean(loop, axis=0, dtype=np.float64) * alignment
    aligned_controls = np.mean(np.abs(controls, dtype=np.float64), axis=0) * np.abs(alignment)
    score = aligned_loop - aligned_controls
    if not np.isfinite(score).all():
        return None, {"reason": "nonfinite_writer_scores"}

    eligible = np.flatnonzero(score > 0)
    if not eligible.size:
        return None, {"reason": "no_positive_writer", "positive_columns": 0}
    ranked = eligible[np.argsort(-score[eligible], kind="stable")[:MAX_COLUMNS]]
    cumulative = np.cumsum(score[ranked])
    total = np.sum(score[eligible])
    if not np.isfinite(total):
        return None, {"reason": "nonfinite_writer_mass"}
    required = total * 0.5
    evidence = {
        "positive_columns": int(eligible.size),
        "max_columns_mass_share": float(cumulative[-1] / total),
    }
    if cumulative[-1] < required:
        return None, evidence | {"reason": "writer_mass_too_diffuse"}
    count = int(np.searchsorted(cumulative, required, side="left")) + 1
    evidence["selected_columns_mass_share"] = float(cumulative[count - 1] / total)
    return ranked[:count], evidence


def _fit_layer(layer, residuals, writers, weights):
    evidence = {"layer": layer, "eligible": False}
    loop, check, cal_loop, cal_check = (np.asarray(array[:, layer], dtype=np.float64) for array in residuals)
    if not all(np.isfinite(array).all() for array in (loop, check, cal_loop, cal_check)):
        return evidence | {"reason": "nonfinite_residuals"}, None

    checking = _unit(check.mean(axis=0))
    if checking is None:
        return evidence | {"reason": "no_checking_direction"}, None
    unprotected, direction = _against_check(loop.mean(axis=0), checking)
    mid = len(loop) // 2
    _, first = _against_check(loop[:mid].mean(axis=0), checking)
    _, second = _against_check(loop[mid:].mean(axis=0), checking)
    if direction is None or first is None or second is None:
        return evidence | {"reason": "no_independent_loop_direction"}, None

    cosine = float(np.clip(np.dot(first, second), -1, 1))
    evidence["fit_half_cosine"] = cosine
    if cosine < HALF_COSINE:
        return evidence | {"reason": "unstable_fit_halves"}, None

    signed = cal_loop @ direction
    control = cal_check @ direction
    raw_control = cal_check @ unprotected
    loop_abs = float(np.mean(np.abs(signed)))
    control_abs = float(np.mean(np.abs(control)))
    raw_control_abs = float(np.mean(np.abs(raw_control)))
    if not np.isfinite([loop_abs, control_abs, raw_control_abs]).all():
        return evidence | {"reason": "nonfinite_calibration_projection"}, None
    positives = int(np.count_nonzero(signed > 0))
    evidence.update(
        calibration_loop_positive=positives,
        calibration_loop_pairs=len(signed),
        calibration_agreement=positives / len(signed),
        calibration_loop_mean_abs=loop_abs,
        calibration_check_mean_abs=control_abs,
        unprotected_calibration_check_mean_abs=raw_control_abs,
    )
    if positives * 3 < 2 * len(signed):
        return evidence | {"reason": "insufficient_loop_agreement"}, None
    if control_abs > CONTROL_RATIO * loop_abs:
        return evidence | {"reason": "checking_signal_too_large"}, None
    if control_abs >= raw_control_abs - MIN_REMAINDER:
        return evidence | {"reason": "checking_projection_not_reduced"}, None

    loop_units, check_units = (array[:, layer] for array in writers)
    if not all(np.isfinite(array).all() for array in (loop_units, check_units, weights)):
        return evidence | {"reason": "nonfinite_writer_data"}, None
    cols, writer_evidence = _writer_columns(loop_units, check_units, weights, direction)
    evidence.update(writer_evidence)
    if cols is None:
        return evidence, None
    evidence["eligible"] = True
    evidence["selected_columns"] = len(cols)
    return evidence, {
        "layer": layer,
        "cols": cols,
        "direction": direction.astype(np.float32),
        "unprotected_direction": unprotected.astype(np.float32),
    }


def choose_edit(fit_loop, fit_check, cal_loop, cal_check, down_weights):
    """Select one non-final MLP down projection using paired BAD-minus-GOOD differences.

    Every split supplies residuals shaped [pairs, layers, hidden]. The fit
    splits also supply MLP activations shaped [pairs, layers, intermediate].
    Down-projection matrices have shape [hidden, intermediate] in layer order.
    The last matrix may be omitted because the last layer is never edited.
    """
    splits = {"fit_loop": fit_loop, "fit_check": fit_check, "cal_loop": cal_loop, "cal_check": cal_check}
    try:
        residuals = [_real_array(split["resid"], 3, f"{name}.resid") for name, split in splits.items()]
        writers = [_real_array(splits[name]["mlp"], 3, f"{name}.mlp") for name in ("fit_loop", "fit_check")]
    except (KeyError, TypeError) as exc:
        raise ValueError("Every split needs resid, and fit splits also need mlp") from exc
    layers, hidden = residuals[0].shape[1:]
    if layers < 2 or len(residuals[0]) < 2 or any(a.shape[1:] != (layers, hidden) for a in residuals):
        raise ValueError("Residual splits need common layer and hidden sizes, and at least two fit-loop pairs")

    intermediate = writers[0].shape[-1]
    if any(writer.shape != (*resid.shape[:2], intermediate) for writer, resid in zip(writers, residuals[:2], strict=True)):
        raise ValueError("Fit MLP splits must match their residual pairs and share an intermediate size")

    if len(down_weights) not in (layers - 1, layers):
        raise ValueError("Down-projection matrices must match the candidate layers")
    matrices = [_real_array(down_weights[layer], 2, f"down_weights[{layer}]") for layer in range(layers - 1)]
    if any(matrix.shape != (hidden, intermediate) for matrix in matrices):
        raise ValueError("Down-projection matrices need the same hidden and intermediate sizes as the probe")

    diagnostics = {
        "pairs": {name: len(array) for name, array in zip(splits, residuals, strict=True)},
        "ignored_final_layer": layers - 1,
        "layers": [],
    }
    candidates = []
    for layer, weights in enumerate(matrices):
        evidence, choice = _fit_layer(layer, residuals, writers, weights)
        diagnostics["layers"].append(evidence)
        if choice is not None:
            rank = (-evidence["calibration_loop_positive"], evidence["calibration_check_mean_abs"], layer)
            candidates.append((rank, choice))
    if not candidates:
        reasons = ", ".join(f"{row['layer']}={row['reason']}" for row in diagnostics["layers"])
        raise ValueError(f"No layer passed the protected-loop requirements ({reasons})")
    return min(candidates, key=lambda entry: entry[0])[1] | {"diagnostics": diagnostics}
