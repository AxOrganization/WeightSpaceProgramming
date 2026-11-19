"""Method-specific local diagnostics; no model weights or precomputed edits."""
import random
import numpy as np
from protocol import run_cli

def permutation_items(rng, index):
    a, b, c = rng.sample(range(11, 150), 3)
    answer = a * b - c
    choices = [answer, answer + a, answer - b, a + b - c]
    if len(set(choices)) != 4:
        return []
    base = list(range(4)); rng.shuffle(base)
    rows = []
    for shift in range(4):
        order = base[shift:] + base[:shift]
        rows.append({'task': 'choice', 'problem':
            f"A workshop makes {a} batches of {b} components and discards {c} components. "
            "How many remain? Select the correct option.",
            'options': '\n'.join(f'{chr(65+j)}. {choices[k]}' for j,k in enumerate(order)),
            'answer': chr(65 + order.index(0)), 'option_values': [choices[k] for k in order],
            'semantic_answer': str(answer), 'family_key':['permutation',sorted([a,b]),c],
            'variant': f'rotation-{shift}'})
    return rows

make_group = permutation_items

def self_test():
    rows = permutation_items(random.Random(17), 0)
    assert len(rows) == 4 and {r['answer'] for r in rows} == set('ABCD')
    for r in rows:
        assert str(r['option_values'][ord(r['answer'])-65]) == r['semantic_answer']
        assert len(set(r['option_values'])) == 4

if __name__ == "__main__":
    run_cli('permutation-label-debias', globals())
