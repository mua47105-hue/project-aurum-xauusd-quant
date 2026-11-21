#!/usr/bin/env python3
"""Trace one broken partial-TP trade through the M1 walker by hand."""
import numpy as np
import pandas as pd

m1 = pd.read_parquet("data/phase1/m1_xauusd.parquet")
m1 = m1[~m1.index.duplicated()].sort_index()
m1_ts = m1.index.asi8
m1_o, m1_h, m1_l, m1_c = (m1[c].values.astype(np.float64) for c in "ohlc")
hour_ids, first_idx = np.unique(m1_ts // 3_600_000_000_000, return_index=True)

px = pd.read_parquet("data/phase1/h1_xauusd.parquet")
px = px[~px.index.duplicated()].sort_index()
px_ts = px.index.asi8
o, hi, lo, c = (px[k].values.astype(np.float64) for k in "ohlc")

# decision ts 1736856000000000000 = 2025-01-14T12:00Z
dec_ts = 1736856000_000_000_000
px_i = np.searchsorted(px_ts, dec_ts)
assert px_ts[px_i] == dec_ts, "decision bar not found"
print(f"decision bar px_i={px_i} stamp={pd.Timestamp(px_ts[px_i], unit='ns', tz='UTC')}")
entry_i = px_i + 1
print(f"entry bar {entry_i} stamp={pd.Timestamp(px_ts[entry_i], unit='ns', tz='UTC')} open={o[entry_i]}")
print(f"exit_bar_h1=89331 stamp={pd.Timestamp(px_ts[89331], unit='ns', tz='UTC')}")
print(f"partial h1 bar 89320 stamp={pd.Timestamp(px_ts[89320], unit='ns', tz='UTC')} high={hi[89320]:.2f}")

# ATR (ewm alpha=1/14) at decision bar
prev_c = np.roll(c, 1); prev_c[0] = c[0]
tr_ = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_c), np.abs(lo - prev_c)))
atr = pd.Series(tr_).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values.astype(np.float64)
a = atr[px_i]
e = o[entry_i]
# q50 inferred from R_h1: R_gross = q50/2 + 0.5*(t/m) with m=1, t=4, R_gross=3.814
q50 = (3.814 - 2.0) * 2.0
part_lvl = e + q50 * a
print(f"\nATR[px_i]={a:.2f}  entry={e:.2f}  q50(inferred)={q50:.3f}  part_lvl={part_lvl:.2f}")

# M1 window: entry hour = stamp of entry bar
entry_hour = px_ts[entry_i] // 3_600_000_000_000
hiq = np.searchsorted(hour_ids, entry_hour)
print(f"entry_hour={pd.Timestamp(entry_hour * 3_600_000_000_000, unit='ns', tz='UTC')} "
      f"bucket found={hour_ids[hiq] == entry_hour} first_m1_idx={first_idx[hiq]}")
start = int(first_idx[hiq])
print(f"first M1 bar: stamp={pd.Timestamp(m1_ts[start], unit='ns', tz='UTC')} open={m1_o[start]} (H1 open {o[entry_i]})")

end_ns = px_ts[px_i + 24] + 3_600_000_000_000
print(f"window end_ns -> {pd.Timestamp(end_ns, unit='ns', tz='UTC')}")
win = (m1_ts >= m1_ts[start]) & (m1_ts < end_ns)
print(f"M1 bars in window: {win.sum()}")
wm_h = m1_h[win]
print(f"M1 window max high = {wm_h.max():.2f}  vs part_lvl = {part_lvl:.2f}  -> partial triggers: {wm_h.max() >= part_lvl}")
# H1 check: bars px_i+1..px_i+24 highs
h1_win_h = hi[entry_i:px_i + 25]
print(f"H1 window max high = {h1_win_h.max():.2f}")
# where is the H1 max?
j = int(np.argmax(h1_win_h))
print(f"H1 max-high bar = {entry_i + j} stamp={pd.Timestamp(px_ts[entry_i + j], unit='ns', tz='UTC')}")
# M1 bars in that hour
that_hour = px_ts[entry_i + j] // 3_600_000_000_000
in_h = (m1_ts // 3_600_000_000_000) == that_hour
print(f"M1 bars in that hour: {in_h.sum()}, max high = {m1_h[in_h].max():.2f}")
print(f"M1 count total span check: first bar of window stamp={pd.Timestamp(m1_ts[start], unit='ns', tz='UTC')}, "
      f"last < end: {pd.Timestamp(m1_ts[win][-1], unit='ns', tz='UTC')}")
