"""Exit 0 if at least N minutes are left before the run's deadline (the file `axonry-time` reads), else 3: a pipeline
job prefixed with it is not started late. Without a deadline file there is no limit.
  python derive/left.py N"""
import sys
import time
from pathlib import Path

need = float(sys.argv[1])
p = Path("/etc/axonry-deadline")
left = (int(p.read_text().strip()) - time.time()) / 60 if p.exists() else float("inf")
print(f"[left] {left:.0f} min left, {need:.0f} needed")
sys.exit(0 if left >= need else 3)
