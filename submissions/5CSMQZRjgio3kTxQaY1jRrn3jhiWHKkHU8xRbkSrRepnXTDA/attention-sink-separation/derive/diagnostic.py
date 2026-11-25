"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def sink_direction(sinks, content, protected, rtol=1e-7):
    sinks,content,protected = [np.asarray(x,dtype=np.float64) for x in (sinks,content,protected)]
    if sinks.ndim != 2 or content.ndim != 2 or sinks.shape[1] != content.shape[1]:
        raise ValueError('states must have matching feature dimensions')
    if not len(sinks) or not len(content) or not sinks.shape[1]:
        raise ValueError('sink and content measurements must be nonempty')
    if not all(np.isfinite(x).all() for x in (sinks,content,protected)):
        raise ValueError('nonfinite states')
    raw = sinks.mean(0)-content.mean(0)
    protected = protected.reshape(-1,len(raw))
    if len(protected):
        _,s,vh = np.linalg.svd(protected,full_matrices=False)
        rank = np.count_nonzero(s > rtol*s[0]) if len(s) and s[0] else 0
        basis = vh[:rank]; raw -= basis.T @ (basis @ raw)
    length = float(np.linalg.norm(raw))
    return raw/length if length > 1e-10 else np.zeros_like(raw)

def analyze(z):
    u = sink_direction(z['sinks'],z['content'],z['protected'])
    return {'candidate_key_input_direction':u.tolist(), 'nonzero':bool(np.linalg.norm(u)>0),
            'note':'Causal head selection and dev attenuation are still required.'}

def self_test():
    u = sink_direction([[3.,2.],[3.,2.]],[[0.,0.]],[[1.,0.]])
    assert np.allclose(u,[0.,1.])
    assert np.allclose(sink_direction([[1.,0.]],[[0.,0.]],[[1.,0.]]),0)

if __name__ == "__main__":
    run_cli('attention-sink-separation', globals())
