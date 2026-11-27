"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def unit_items(rng, index):
    large, small, scale = rng.choice([('meters','centimeters',100), ('minutes','seconds',60),
                                     ('kilometers','meters',1000), ('hours','minutes',60)])
    a, b = rng.sample(range(3, 80), 2); total = a+b
    rows = [
      {'problem':f"Two segments measure {a} {large} and {b} {large}. What is their total in {small}?",
       'answer':str(total*scale), 'output_unit':small, 'to_smallest':1, 'variant':'large-input-small-output'},
      {'problem':f"Two segments measure {a*scale} {small} and {b*scale} {small}. What is their total in {small}?",
       'answer':str(total*scale), 'output_unit':small, 'to_smallest':1, 'variant':'small-input-small-output'},
      {'problem':f"Two segments measure {a*scale} {small} and {b*scale} {small}. What is their total in {large}?",
       'answer':str(total), 'output_unit':large, 'to_smallest':scale, 'variant':'small-input-large-output'}]
    for r in rows:
        r.update(task='math', physical_total_smallest=total*scale, conversion_scale=scale,
                 family_key=['unit',sorted([a,b]),large,small])
        if large in ('minutes','hours'):
            r['problem'] = r['problem'].replace('segments measure', 'nonoverlapping activities last')
    return rows

make_group = unit_items

def self_test():
    for i in range(40):
        rows = unit_items(random.Random(i), i)
        assert len({int(r['answer'])*r['to_smallest'] for r in rows}) == 1
        assert int(rows[0]['answer']) == rows[0]['physical_total_smallest']

if __name__ == "__main__":
    run_cli('unit-equivariance-repair', globals())
