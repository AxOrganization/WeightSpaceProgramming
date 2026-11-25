"""The published train_sparse.py (unchanged) with the generated requests in the item index (derive/xitems.py, REQ_ITEMS).

  REQ_ITEMS=REQ.jsonl torchrun --standalone --nproc_per_node 2 derive/x_train.py <arguments of train_sparse.py ...>
"""
import xitems  # noqa: F401  (adds the requests; must come first)
import train_sparse

if __name__ == "__main__":
    train_sparse.main()
