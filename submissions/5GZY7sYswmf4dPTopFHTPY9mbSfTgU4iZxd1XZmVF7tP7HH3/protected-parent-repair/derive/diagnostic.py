"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def protected_direction(gradient, protected, rtol=1e-7):
    """Project a proposed descent direction out of protected gradient row-space."""
    g = np.asarray(gradient, dtype=np.float64)
    a = np.asarray(protected, dtype=np.float64).reshape(-1, g.size)
    if not np.isfinite(g).all() or not np.isfinite(a).all():
        raise ValueError("nonfinite gradient")
    if not len(a):
        return g.copy()
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(singular > rtol * singular[0])) if len(singular) and singular[0] else 0
    basis = vh[:rank]
    return g - basis.T @ (basis @ g)

def analyze(z):
    d = protected_direction(-np.asarray(z['gradient']), z['protected'])
    return {'direction': d.tolist(), 'norm': float(np.linalg.norm(d)),
            'max_protected_dot': float(np.max(np.abs(z['protected'] @ d), initial=0))}

def self_test():
    d = protected_direction([2., 3., 4.], [[1., 0., 0.], [2., 0., 0.]])
    assert np.allclose(d, [0., 3., 4.])
    assert np.allclose(protected_direction([2., 3.], np.eye(2)), 0)
    assert np.allclose(protected_direction([2., 3.], np.empty((0, 2))), [2., 3.])
    result = analyze({'gradient': np.array([2., 3.]), 'protected': np.array([[1., 0.]])})
    assert np.allclose(result['direction'], [0., -3.])

if __name__ == "__main__":
    run_cli('protected-parent-repair', globals())
