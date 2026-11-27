"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def choose_coordinate(gradients,curvature,task_scale,step):
    g,h = np.asarray(gradients,dtype=np.float64),np.asarray(curvature,dtype=np.float64)
    scales=np.asarray(task_scale,dtype=np.float64)
    if g.ndim!=2 or h.shape!=g.shape or scales.shape!=(g.shape[0],):
        raise ValueError('expect tasks x coordinates and one scale per task')
    if step<=0 or not np.isfinite(step) or np.any(h<0) or np.any(scales<=0):
        raise ValueError('step/scales must be positive and curvature nonnegative')
    if not all(np.isfinite(x).all() for x in (g,h,scales)):
        raise ValueError('nonfinite local loss model')
    best=None
    for j in range(g.shape[1]):
        for sign in (-1.,1.):
            delta=sign*step
            change=(g[:,j]*delta+.5*h[:,j]*delta*delta)/scales
            # Strict shared improvement; no trade-off is hidden by averaging.
            improvement=float(np.min(-change))
            if improvement>0 and (best is None or improvement>best['worst_task_improvement']):
                best={'coordinate':j,'delta_log_scale':delta,
                      'worst_task_improvement':improvement,'predicted_task_change':change.tolist()}
    return best

def analyze(z):
    return {'candidate':choose_coordinate(z['gradients'],z['curvature'],z['task_scale'],float(z['step']))}

def self_test():
    r=choose_coordinate([[-2.,1.],[-1.,-1.]],np.zeros((2,2)),[1.,1.],.1)
    assert r['coordinate']==0 and r['delta_log_scale']>0
    assert choose_coordinate([[1.],[-1.]],np.zeros((2,1)),[1.,1.],.1) is None
    assert choose_coordinate([[-1.],[-1.]],[[100.],[100.]],[1.,1.],1.) is None

if __name__ == "__main__":
    run_cli('multiobjective-curvature-scales', globals())
