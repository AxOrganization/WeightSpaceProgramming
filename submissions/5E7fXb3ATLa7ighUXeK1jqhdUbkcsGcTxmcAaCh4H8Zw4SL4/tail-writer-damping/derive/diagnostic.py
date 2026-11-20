"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def tail_scores(activations, harmful_effect, quantile=.99):
    """Inputs [events, units], and held-out causal loss reduction [units]."""
    a = np.asarray(activations,dtype=np.float64)
    harm = np.asarray(harmful_effect,dtype=np.float64)
    if a.ndim != 2 or harm.shape != (a.shape[1],) or not 0.5 < quantile < 1:
        raise ValueError('invalid shapes or quantile')
    typical = np.median(np.abs(a),axis=0)
    floor = max(float(np.median(typical))*1e-6,1e-10)
    ratio = np.quantile(np.abs(a),quantile,axis=0)/np.maximum(typical,floor)
    return ratio*np.maximum(harm,0),ratio

def analyze(z):
    scores,ratio = tail_scores(z['activations'],z['harmful_effect'])
    return {'scores':scores.tolist(),'tail_ratio':ratio.tolist(),
            'ranked_units':np.argsort(-scores,kind='stable').tolist()}

def self_test():
    a = np.ones((1000,3)); a[980:,0]=100; a[980:,1]=100
    score,ratio = tail_scores(a,[1.,-1.,.1])
    assert score[0]>score[2]>score[1] and score[1]==0 and ratio[0]>ratio[2]

if __name__ == "__main__":
    run_cli('tail-writer-damping', globals())
