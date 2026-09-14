"""Cost-aware supervised research entry point.

The former TP/SL classifier ignored fees, discarded unresolved trades, and
saved binary logits as if they were a calibrated PPO policy. That transfer
is removed. New artifacts predict net R and declare their purpose.

Usage: python pretrain_supervised.py --output experiments/net-return-v1
See run_entry_research.py for the chronological experimental protocol.
"""
from run_entry_research import main

if __name__ == '__main__':
    main()
