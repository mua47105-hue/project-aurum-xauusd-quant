"""
AURUM v2.0 -- Purged Walk-Forward Splitter (purge.py)
======================================================
Blueprint 2 Section 4.9, verbatim.

Semantics follow Lopez de Prado, Advances in Financial Machine Learning (2018),
ch. 7 (purged k-fold CV with embargo), cross-verified against the open
skfolio reproduction of CombinatorialPurgedCV:

  PURGE : drop from the TRAINING set every sample whose LABEL WINDOW
          (t, t+H] overlaps the label window of any TEST sample.  With label
          horizon H = 24 bars and contiguous yearly test blocks, this removes
          train rows in [test_start - H, test_start).
  EMBARGO: additionally drop train rows immediately FOLLOWING the test block
          (serial correlation of features leaks backward in time):
          [test_end + 1, test_end + H + embargo].
  Constraint (skfolio): purge + embargo < fold size - 1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    label_horizon: int = 24          # H in bars
    embargo: int = 24                # extra bars after test block (24-48 rec.)
    test_block: str = "365D"         # contiguous test block length
    min_train: int = 24 * 252 * 2    # burn-in before first test block

    def splits(self, index: pd.DatetimeIndex):
        """Yield (train_idx, test_idx) position arrays, expanding window."""
        idx = pd.DatetimeIndex(index)
        blocks = pd.date_range(idx[self.min_train], idx[-1],
                               freq=self.test_block, tz=idx.tz)
        for i, bstart in enumerate(blocks):
            bend = blocks[i + 1] if i + 1 < len(blocks) else idx[-1] + pd.Timedelta("1h")
            t_lo = idx.searchsorted(bstart)
            t_hi = idx.searchsorted(bend)
            if t_hi - t_lo < self.label_horizon + 2:
                continue
            test = np.arange(t_lo, t_hi)
            # purge: train labels must not overlap the test label window
            purge_lo = max(0, t_lo - self.label_horizon)
            # embargo after the test block
            emb_hi = min(len(idx), t_hi + self.label_horizon + self.embargo)
            train = np.concatenate([np.arange(0, purge_lo),
                                    np.arange(emb_hi, len(idx))])
            train = train[train >= self.min_train]
            # enforce the skfolio constraint
            if self.label_horizon + self.embargo >= (t_hi - t_lo) - 1:
                raise ValueError("purge + embargo >= fold size - 1; reduce embargo")
            if len(train) > 0:
                yield train, test


def self_test() -> None:
    idx = pd.date_range("2016-01-01", "2026-01-01", freq="h", tz="UTC")
    wf = PurgedWalkForward(label_horizon=24, embargo=24, min_train=24 * 252 * 2)
    n = 0
    last_test_end = -1
    prev_pre = 0
    for train, test in wf.splits(idx):
        # test blocks are disjoint and ordered
        assert test.min() > last_test_end
        last_test_end = test.max()
        # no train row within the purge window before the test block
        assert not np.any((train >= test.min() - 24) & (train < test.min()))
        # no train row within the embargo window after the test block
        assert not np.any((train > test.max()) & (train <= test.max() + 24 + 24))
        # expanding window: pre-block train grows over folds
        if n > 0:
            assert (train < test.min()).sum() >= prev_pre
        prev_pre = (train < test.min()).sum()
        n += 1
    assert n >= 6, f"expected >=6 yearly folds, got {n}"
    print(f"PurgedWalkForward self-test passed: {n} folds, purge=24, embargo=24.")
