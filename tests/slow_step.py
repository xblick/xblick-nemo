"""The node with NeMO's whole-frame search made slower by SLOW_S seconds (its first argument), for
tests/test_node.py: its queryables answer while the model works. It prints "slow step" when one begins."""

import sys
import time

from xblick_nemo import nemo, node

SLOW_S = float(sys.argv.pop(1))
search = nemo.NeMO.detect


def slow_detect(self, *args, **kwargs):
    print("slow step", flush=True)
    time.sleep(SLOW_S)
    return search(self, *args, **kwargs)


nemo.NeMO.detect = slow_detect
sys.exit(node.main())
