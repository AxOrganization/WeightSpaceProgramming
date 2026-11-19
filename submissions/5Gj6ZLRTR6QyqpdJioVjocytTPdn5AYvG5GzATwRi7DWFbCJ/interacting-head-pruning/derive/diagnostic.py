"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def head_interaction(parent_loss,loss_drop_a,loss_drop_b,loss_drop_both):
    base,a,b,ab = np.broadcast_arrays(*[np.asarray(x,dtype=np.float64)
                                       for x in (parent_loss,loss_drop_a,loss_drop_b,loss_drop_both)])
    ia,ib,joint = base-a,base-b,base-ab
    return {'individual_a':ia,'individual_b':ib,'joint_improvement':joint,
            'interaction':joint-ia-ib, 'better_than_either':joint>np.maximum(ia,ib)}

def analyze(z):
    r=head_interaction(z['parent_loss'],z['loss_drop_a'],z['loss_drop_b'],z['loss_drop_both'])
    return {k:v.tolist() for k,v in r.items()}

def self_test():
    r=head_interaction([5.,5.],[5.2,4.],[5.1,4.5],[3.,3.5])
    assert r['interaction'][0]>0 and r['joint_improvement'][0]>0
    assert np.isclose(r['interaction'][1],0)
    assert bool(r['better_than_either'][0])

if __name__ == "__main__":
    run_cli('interacting-head-pruning', globals())
