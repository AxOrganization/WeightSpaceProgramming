"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def execute(program):
    state = {}
    for op, dst, arg in program:
        if op == 'const': state[dst] = arg
        elif op == 'copy': state[dst] = state[arg]
        elif op == 'add': state[dst] += arg
        else: raise ValueError(op)
    return state

def binding_items(rng, index):
    a, b = rng.sample(range(20, 180), 2); delta = rng.randrange(1, 25)
    mode = rng.randrange(3)
    program = [('const',0,a), ('const',1,b)]
    program += ([('copy',2,0), ('copy',0,1), ('copy',1,2)] if mode == 0 else
                [('copy',2,1), ('add',1,delta), ('copy',0,2)] if mode == 1 else
                [('copy',2,0), ('add',0,delta), ('copy',1,0), ('add',2,-delta)])
    target = rng.randrange(3); gold = execute(program)[target]
    rows = []
    for variant, names in enumerate((['amber','birch','cedar'], ['wren','lynx','ibis'])):
        rng.shuffle(names)
        steps = []
        for op,dst,arg in program:
            steps.append(f"Set {names[dst]} to {arg}." if op == 'const' else
                         f"Copy the current value of {names[arg]} into {names[dst]}." if op == 'copy' else
                         f"Add {arg} to the current value of {names[dst]}.")
        rows.append({'task':'math', 'problem':'Execute these assignments in order. '
            + ' '.join(steps) + f" What is the final value of {names[target]}?",
            'answer':str(gold), 'program':program, 'target':target, 'names':names,
            'family_key':['binding',program,target], 'variant':f'renaming-{variant}'})
    return rows

make_group = binding_items

def self_test():
    assert execute([('const',0,3),('copy',1,0),('add',0,2)]) == {0:5,1:3}
    for i in range(30):
        rows = binding_items(random.Random(i), i)
        assert len({r['answer'] for r in rows}) == 1
        assert int(rows[0]['answer']) == execute(rows[0]['program'])[rows[0]['target']]

if __name__ == "__main__":
    run_cli('role-binding-invariance', globals())
