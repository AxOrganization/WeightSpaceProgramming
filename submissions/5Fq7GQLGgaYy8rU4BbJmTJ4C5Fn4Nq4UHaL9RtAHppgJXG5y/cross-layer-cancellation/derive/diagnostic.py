"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def cancellation_scores(earlier,later,margin_gradient):
    a,b,g = [np.asarray(x,dtype=np.float64) for x in (earlier,later,margin_gradient)]
    if a.shape != b.shape or a.shape != g.shape or a.ndim != 2:
        raise ValueError('expected aligned [positions, residual_features] arrays')
    pa,pb = np.sum(a*g,axis=1),np.sum(b*g,axis=1)
    score = np.abs(pa)+np.abs(pb)-np.abs(pa+pb)
    return {'score':np.maximum(score,0), 'earlier_margin_component':pa,
            'later_margin_component':pb, 'combined_margin_component':pa+pb}

def analyze(z):
    return {k:v.tolist() for k,v in cancellation_scores(z['earlier'],z['later'],z['margin_gradient']).items()}

def self_test():
    r = cancellation_scores([[2.,0.],[1.,0.]],[[-2.,0.],[2.,0.]],[[1.,0.],[1.,0.]])
    assert np.allclose(r['score'],[4.,0.])
    assert np.allclose(r['combined_margin_component'],[0.,3.])

if __name__ == "__main__":
    run_cli('cross-layer-cancellation', globals())
