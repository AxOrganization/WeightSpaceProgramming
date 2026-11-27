"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def silu(x):
    x = np.asarray(x,dtype=np.float64)
    sigmoid = np.exp(-np.logaddexp(0.,-x))
    return x*sigmoid

def gate_rebalance(gate, up, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('factor must be positive and measured on dev')
    gate,up = np.asarray(gate,dtype=np.float64),np.asarray(up,dtype=np.float64)
    if gate.shape != up.shape: raise ValueError('gate/up shapes differ')
    return silu(gate*factor)*(up/factor)

def analyze(z):
    factor = float(z['factor'])
    before = silu(z['gate'])*z['up']; after = gate_rebalance(z['gate'],z['up'],factor)
    return {'before':before.tolist(),'after':after.tolist(),
            'change_l2':float(np.linalg.norm(after-before)),
            'gate_factor':factor,'up_factor':1/factor}

def self_test():
    g,u = np.array([-2.,0.,2.]),np.array([1.,3.,4.])
    assert np.allclose(gate_rebalance(g,u,1),silu(g)*u)
    assert not np.allclose(gate_rebalance(g,u,2),silu(g)*u)
    assert np.allclose((g*2)*(u/2),g*u)
    assert np.isfinite(silu([-1000.,1000.])).all()

if __name__ == "__main__":
    run_cli('nonlinear-gate-rebalance', globals())
