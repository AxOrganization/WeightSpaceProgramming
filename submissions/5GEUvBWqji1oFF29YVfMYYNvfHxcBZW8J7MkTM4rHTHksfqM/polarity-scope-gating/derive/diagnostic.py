"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def truth(expr, a, b):
    if expr == 'a': return a
    if expr == 'b': return b
    op,*xs = expr
    vals = [truth(x,a,b) for x in xs]
    return not vals[0] if op == 'not' else all(vals) if op == 'and' else any(vals)

def render(expr):
    if isinstance(expr,str): return expr.upper()
    if expr[0] == 'not': return f'(NOT {render(expr[1])})'
    return f'({render(expr[1])} {expr[0].upper()} {render(expr[2])})'

def polarity_items(rng, index):
    a,b = bool(rng.randrange(2)),bool(rng.randrange(2))
    def subtree(depth):
        if depth == 0: return rng.choice(['a','b'])
        e=(rng.choice(['and','or']),subtree(depth-1),subtree(depth-1))
        return ('not',e) if rng.randrange(2) else e
    left,right=subtree(2),subtree(2)
    forms = [('not',('and',left,right)), ('or',('not',left),('not',right)),
             ('and',('not',left),right), ('not',('not',('and',left,right)))]
    return [{'task':'math', 'problem':f"A is {a} and B is {b}. "
             f"Using Boolean logic, evaluate {render(e)}. Return 1 for true or 0 for false.",
             'answer':str(int(truth(e,a,b))), 'expr':e, 'a':a, 'b':b,
             'family_key':['polarity',a,b,forms],
             'variant':f'scope-{j}'} for j,e in enumerate(forms)]

make_group = polarity_items

def self_test():
    for a in (False,True):
        for b in (False,True):
            assert truth(('not',('and','a','b')),a,b) == truth(('or',('not','a'),('not','b')),a,b)
            assert truth(('not',('not',('and','a','b'))),a,b) == (a and b)

if __name__ == "__main__":
    run_cli('polarity-scope-gating', globals())
