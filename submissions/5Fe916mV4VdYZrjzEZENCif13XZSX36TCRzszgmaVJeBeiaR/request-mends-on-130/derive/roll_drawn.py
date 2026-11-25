"""rollouts.py with the drawn puzzles in the item index (derive/drawn_items.py, DRAWN_ITEMS).

  DRAWN_ITEMS=PZ.jsonl python derive/roll_drawn.py <arguments of rollouts.py ...>
"""
import os

import drawn_items  # noqa: F401  (adds the drawn puzzles; must come first)
import rollouts

if __name__ == "__main__":
    rollouts.__file__ = os.path.abspath(__file__)      # its GPU workers start through this file as well
    rollouts.main()
