"""The published rollouts.py (unchanged) with the generated requests in the item index (derive/xitems.py, REQ_ITEMS).

  REQ_ITEMS=REQ.jsonl python derive/x_rollouts.py <arguments of rollouts.py ...>
"""
import os

import xitems  # noqa: F401  (adds the requests; must come first)
import rollouts

if __name__ == "__main__":
    rollouts.__file__ = os.path.abspath(__file__)      # its GPU workers start through this file as well
    rollouts.main()
