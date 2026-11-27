"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def compare(x,t,op):
    return {'<':x<t,'<=':x<=t,'>':x>t,'>=':x>=t}[op]

def boundary_items(rng,index):
    t=rng.randrange(-200,500); rows=[]
    strict,inclusive = rng.choice([('<','<='),('>','>=')])
    words={'<':'strictly less than','<=':'less than or equal to',
           '>':'strictly greater than','>=':'greater than or equal to'}
    for op in (strict,inclusive):
        for offset in (-1,0,1):
            x=t+offset
            rows.append({'task':'math','problem':f'Is {x} {words[op]} {t}? '
                'Return 1 if the comparison is true and 0 otherwise.',
                'answer':str(int(compare(x,t,op))),'x':x,'threshold':t,'operator':op,
                'family_key':['boundary',t],
                'variant':f'{op}:{offset}'})
    return rows

make_group=boundary_items

def self_test():
    for t in (-10,0,30):
        assert not compare(t,t,'<') and compare(t,t,'<=')
        assert not compare(t,t,'>') and compare(t,t,'>=')
        for x in (t-1,t+1):
            assert compare(x,t,'<')==compare(x,t,'<=')
            assert compare(x,t,'>')==compare(x,t,'>=')

if __name__ == "__main__":
    run_cli('boundary-triplet-repair', globals())
