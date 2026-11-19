"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def metric_shrinkage(covariance,amount,ridge=1e-5):
    cov = np.asarray(covariance,dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0]!=cov.shape[1] or not 0<=amount<=1 or ridge<=0:
        raise ValueError('invalid covariance, amount or ridge')
    cov = (cov+cov.T)/2
    eig,vec = np.linalg.eigh(cov)
    scale = max(float(np.mean(np.maximum(eig,0))),1e-10)
    inv = 1/np.sqrt(np.maximum(eig/scale,ridge))
    inv /= np.mean(inv)
    factor = (1-amount)+amount*inv
    return (vec*factor)@vec.T

def analyze(z):
    transform = metric_shrinkage(z['covariance'],float(z['amount']))
    return {'candidate_transform':transform.tolist(),
            'distance_to_identity':float(np.linalg.norm(transform-np.eye(len(transform)))),
            'note':'Fit a budgeted sparse q_proj row realization and verify actual attention.'}

def self_test():
    assert np.allclose(metric_shrinkage(np.eye(3),.5),np.eye(3))
    assert np.allclose(metric_shrinkage(np.diag([1.,9.]),0),np.eye(2))
    t = metric_shrinkage(np.diag([1.,9.]),.5)
    assert t[0,0]>t[1,1]>0 and np.isclose(np.trace(t),2)

if __name__ == "__main__":
    run_cli('query-metric-shrinkage', globals())
