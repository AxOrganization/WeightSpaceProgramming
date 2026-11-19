"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def carry_trace(a,b):
    digits = max(len(str(a)),len(str(b))); carry=0; out=[]
    for pos in range(digits):
        da,db = (a//10**pos)%10,(b//10**pos)%10
        total=da+db+carry
        out.append({'position':pos,'a_digit':da,'b_digit':db,'incoming':carry,
                    'out_digit':total%10,'outgoing':total//10})
        carry=total//10
    reconstructed=sum(s['out_digit']*10**s['position'] for s in out)+carry*10**digits
    assert reconstructed==a+b
    return out

def carry_items(rng,index):
    digits=rng.randrange(2,7); mode=index%3
    if mode == 2:
        a=10**digits-rng.randrange(1,10); b=rng.randrange(10,40)
    else:
        aa=[rng.randrange(1,5) for _ in range(digits)]
        bb=[rng.randrange(1,5) for _ in range(digits)]
        if mode == 1:
            pos=rng.randrange(digits);aa[pos]=rng.randrange(6,10);bb[pos]=rng.randrange(6,10)
        a=sum(x*10**i for i,x in enumerate(aa));b=sum(x*10**i for i,x in enumerate(bb))
    trace=carry_trace(a,b)
    return [{'task':'math','problem':f'Compute the exact integer sum {a} + {b}.',
             'answer':str(a+b),'a':a,'b':b,'carry_trace':trace,
             'family_key':['carry',sorted([a,b])],'variant':'symbolic'},
            {'task':'math','problem':f'A depot counts {a} identical parts in one crate and {b} in a second crate. '
             'What is the total number of parts?', 'answer':str(a+b),'a':a,'b':b,
             'carry_trace':trace,'family_key':['carry',sorted([a,b])],'variant':'worded'}]

make_group=carry_items

def self_test():
    t=carry_trace(999,1)
    assert [x['outgoing'] for x in t]==[1,1,1]
    assert all(x['outgoing']==0 for x in carry_trace(123,234))
    for i in range(50):carry_items(random.Random(i),i)

if __name__ == "__main__":
    run_cli('carry-state-repair', globals())
