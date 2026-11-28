"""The published branch.py (unchanged) with the generated requests in the item index (derive/xitems.py, REQ_ITEMS).

  REQ_ITEMS=REQ.jsonl python derive/x_branch.py <arguments of branch.py ...>
"""
import os

import xitems  # noqa: F401  (adds the requests; must come first)
import branch

if __name__ == "__main__":
    branch.__file__ = os.path.abspath(__file__)      # its GPU workers start through this file as well
    branch.main()
