"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def memory_items(rng,index):
    names = rng.sample(['moss','reed','clay','opal','ash','fern','slate','pine'],5)
    values = rng.sample(range(100,1000),24)
    query = names[0]; initial = [(names[j],values[j]) for j in range(5)]
    distract = [(rng.choice(names[1:]),v) for v in values[5:5+rng.randrange(4,16)]]
    old,new = values[0],values[-1]
    sequences = [('short',initial,old),('interference',initial+distract,old),
                 ('explicit-replacement',initial+[(query,new)]+distract,new)]
    rows = []
    for variant,seq,gold in sequences:
        state = dict(seq); assert state[query]==gold
        problem = 'A log stores integer values under keys. Each assignment replaces the earlier value of that key. '
        problem += ' '.join(f'Set {key} to {value}.' for key,value in seq)
        problem += f' What is the latest stored value of {query}?'
        rows.append({'task':'math','problem':problem,'answer':str(gold),'query':query,
                     'assignments':seq,'family_key':['memory',initial], 'variant':variant})
    return rows

make_group = memory_items

def self_test():
    for i in range(30):
        rows = memory_items(random.Random(i),i)
        assert rows[0]['answer']==rows[1]['answer']!=rows[2]['answer']
        for r in rows: assert str(dict(r['assignments'])[r['query']])==r['answer']

if __name__ == "__main__":
    run_cli('recurrent-write-interference', globals())
