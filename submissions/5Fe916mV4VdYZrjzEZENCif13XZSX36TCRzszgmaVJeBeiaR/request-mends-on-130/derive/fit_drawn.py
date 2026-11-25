"""train_sparse.py with the drawn puzzles in the item index (derive/drawn_items.py, DRAWN_ITEMS).

  DRAWN_ITEMS=PZ.jsonl torchrun --standalone --nproc_per_node 4 derive/fit_drawn.py <arguments of train_sparse.py ...>
"""
import drawn_items  # noqa: F401  (adds the drawn puzzles; must come first)
import train_sparse

if __name__ == "__main__":
    train_sparse.main()
