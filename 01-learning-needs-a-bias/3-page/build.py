import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_kit"))
import page

ok = page.build(
    Path(__file__).resolve().parents[1],
    more=[{"section": "the-formal-statements", "from": "For the three together, we need a little more care",
           "through": "Events built from disjoint groups of independent bits are independent",
           "button": "Show why the three survival events are independent"}],
)
sys.exit(0 if ok else 1)
