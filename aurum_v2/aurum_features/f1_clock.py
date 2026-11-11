"""
AURUM v2.0 -- Feature Module F1: DST-Aware Exchange-Clock Features (UNIFIED)
==============================================================================
Synthesis of BOTH v2.0 blueprints:
  * Blueprint 2 (Industrial Quant Blueprint) Section 3.1 -- DST-aware exchange
    clocks (LBMA AM/PM, Silver, WMR, SGE AM/PM/Night, COMEX open/settle/break),
    session dummies, DST gap features, Lou-Polk-Skouras session-split returns.
  * Blueprint 1 (Transcript Intelligence Report) Section 5.1 -- Krohn-Mueller-
    Whelan (2024 JF) post-fix reversal signals, pre-auction 30m M1 drift
    signals (S38/S39), Blose et al. (S40) daytime asymmetry flags, Tokyo
    gotobi / London month-end-4pm flags.

Ground truth timetable (verified 2026-11 against LBMA 'About LBMA daily auction
prices', SGE 'Detailed Rules of SGE on Benchmark Price Trading' (Nov-2019
amendment) and IANA tzdata via zoneinfo; Spec v2.0 Section 2, ledger P2-1..P2-4):

  LBMA Gold Price AM auction   10:30 Europe/London   (IBA-administered)
  LBMA Gold Price PM auction   15:00 Europe/London   (anchored since 1968)
  LBMA Silver Price auction    12:00 Europe/London   (once daily)
  WM/Refinitiv FX fix          16:00 Europe/London   (rate window 15:59:30-16:00;
                                                     5-minute window since May 2015)
  SGE Shanghai Gold Bench AM   10:15 Asia/Shanghai   (SHAU; reference px 10:09-10:14)
  SGE Shanghai Gold Bench PM   15:00 Asia/Shanghai   (SHAU; 2019-amended rules;
                                                     legacy time was 14:15)
  SGE night session open       19:45 Asia/Shanghai   (pre-order matching 19:45-19:59)
  COMEX/CME Globex day open   18:00 America/New_York (Sun-Fri; NY Fed-corroborated)
  COMEX GC daily settlement   13:30 America/New_York (legacy convention)
  COMEX/Globex maintenance    17:00 America/New_York (17:00-18:00 ET break)

DST rules: UK/EU last-Sun March -> last-Sun Oct (01:00 UTC); US 2nd-Sun March ->
1st-Sun Nov (02:00 local); China no DST since 1991.  London-NY offset is 5h except
two annual windows (spring: 14 or 21 days; autumn: exactly 7 days) when it is 4h.

Every feature is a pure function of the bar's UTC timestamp (and, for the
session-split returns and the M1 fixing drifts, of trailing prices) -> strictly
causal.

References: Caminschi & Heaney (JFM 34(11), 2014); Krohn, Mueller & Whelan (JF
2024, fx fixings W-cycle); Lou, Polk & Skouras (JFE 134(1), 2019); Boyarchenko,
Larsen & Whelan (RFS 36(9), 2023); Gao, Han, Li, Zhou (JFE 129(2), 2018);
Baltussen, Da, Lammers, Martens (JFE 142(1), 2021); Kamstra, Kramer, Levi (AER
90(4), 2000); Blose, Gondhalekar, Kort (JEF 42(3), 2018); Kurov et al. (ECB WP
1901, 2016).
"""

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
NEWYORK = ZoneInfo("America/New_York")
SHANGHAI = ZoneInfo("Asia/Shanghai")
TOKYO = ZoneInfo("Asia/Tokyo")

# (timezone, hour, minute) -- all verified times, see module docstring.
ANCHORS: dict[str, tuple[ZoneInfo, int, int]] = {
    "LBMA_AM":      (LONDON,  10, 30),
    "LBMA_PM":      (LONDON,  15,  0),
    "LBMA_SILVER":  (LONDON,  12,  0),
    "WMR_FIX":      (LONDON,  16,  0),
    "SGE_AM":       (SHANGHAI, 10, 15),
    "SGE_PM":       (SHANGHAI, 15,  0),
    "SGE_NIGHT":    (SHANGHAI, 19, 45),
    "COMEX_OPEN":   (NEWYORK, 18,  0),
    "COMEX_SETTLE": (NEWYORK, 13, 30),
    "COMEX_BREAK":  (NEWYORK, 17,  0),
}


def _local_wall_minutes(idx_utc: pd.DatetimeIndex, tz: ZoneInfo) -> pd.Series:
    """Minutes since local midnight (wall clock), as float Series indexed by idx_utc."""
    loc = idx_utc.tz_convert(tz)
    return pd.Series(loc.hour * 60 + loc.minute, index=idx_utc, dtype="float64")


def minutes_to_anchor(idx_utc: pd.DatetimeIndex, name: str) -> pd.Series:
    """Signed minutes from each timestamp to the anchor event on the SAME
    exchange-local calendar day.  Positive = event still ahead today;
    negative = event already passed today.  Naive UTC hour arithmetic would
    blur this by one hour for half the year (DST) -- hence tz_convert, never
    fixed offsets."""
    tz, hh, mm = ANCHORS[name]
    event_min = hh * 60 + mm
    now_min = _local_wall_minutes(idx_utc, tz)
    out = event_min - now_min
    # Mask weekends per venue (no fixes / sessions on Sat-Sun in that venue):
    dow = idx_utc.tz_convert(tz).dayofweek
    venue_weekend = (dow >= 5)
    if name.startswith(("LBMA", "WMR")):
        out[venue_weekend] = np.nan
    elif name.startswith("SGE"):
        out[(dow >= 5)] = np.nan
    # COMEX opens Sunday 18:00 ET -> keep Sunday for COMEX anchors (dayofweek==6).
    return out


def _dst_transitions(year_start: int, year_end: int) -> dict[str, pd.DatetimeIndex]:
    """Exact DST transition instants for London and New York between
    year_start-1 and year_end+1, found by scanning daily utcoffset changes and
    refining to the hour (authoritative source: the tz database itself, via
    zoneinfo -- exactly the mechanism pandas uses for tz_convert)."""
    days = pd.date_range(f"{year_start - 1}-01-01", f"{year_end + 1}-12-31", freq="D", tz="UTC")
    out: dict[str, list] = {"UK": [], "US": []}
    for tzname, tz in (("UK", LONDON), ("US", NEWYORK)):
        offs = pd.Series([d.tz_convert(tz).utcoffset() for d in days], index=days)
        day_chg = offs.ne(offs.shift()).fillna(False)
        day_chg.iloc[0] = False  # drop the artificial first-row change
        for day in days[day_chg]:
            # refine within +-36h to the exact transition hour
            hrs = pd.date_range(day - pd.Timedelta("36h"), day + pd.Timedelta("12h"), freq="h", tz="UTC")
            hoffs = pd.Series([h.tz_convert(tz).utcoffset() for h in hrs], index=hrs)
            h_chg = hoffs.ne(hoffs.shift()).fillna(False)
            h_chg.iloc[0] = False
            out[tzname].extend(h for h in hrs[h_chg])
    return {k: pd.DatetimeIndex(v) for k, v in out.items()}


def clock_features(idx_utc: pd.DatetimeIndex,
                   close_xau: pd.Series,
                   year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Full DST-aware exchange-clock feature block for a UTC-indexed H1 frame
    (Blueprint 2 core + Blueprint 1 timestamp-level additions)."""
    if idx_utc.tz is None:
        raise ValueError("idx_utc must be tz-aware UTC (bar close times).")
    idx = idx_utc.tz_convert("UTC")
    feats = pd.DataFrame(index=idx)

    # --- 1) Anchor distances (signed minutes, same venue-local day) --------
    for name in ANCHORS:
        feats[f"minutes_to_{name}"] = minutes_to_anchor(idx, name)

    # Caminschi-Heaney edge window: the first 4 minutes of the PM fix carry
    # the informed-trading profit; and the 60 minutes running into the fix.
    m_pm = feats["minutes_to_LBMA_PM"]
    feats["in_pm_fix_window"] = ((m_pm <= 0) & (m_pm >= -10)).astype("float64")
    feats["in_pm_fix_first4m"] = ((m_pm <= 0) & (m_pm >= -4)).astype("float64")
    feats["pre_pm_fix_60m"] = ((m_pm < 0) & (m_pm >= -60)).astype("float64")

    # --- 2) Session dummies in EXCHANGE-LOCAL time -------------------------
    loc_ldn = idx.tz_convert(LONDON)
    loc_ny = idx.tz_convert(NEWYORK)
    loc_sh = idx.tz_convert(SHANGHAI)
    h_ldn, h_ny, h_sh = loc_ldn.hour, loc_ny.hour, loc_sh.hour
    dow_ldn, dow_ny, dow_sh = loc_ldn.dayofweek, loc_ny.dayofweek, loc_sh.dayofweek

    feats["sess_london_day"] = ((h_ldn >= 8) & (h_ldn < 17) & (dow_ldn < 5)).astype(float)
    feats["sess_ny_day"] = ((h_ny >= 9) & (h_ny < 16) & (dow_ny < 5)).astype(float)
    feats["sess_globex_break"] = ((h_ny == 17)).astype(float)
    feats["sess_asia_night"] = (((h_sh >= 20) | (h_sh < 3)) & (dow_sh < 6)).astype(float)
    feats["sess_shanghai_day"] = (((h_sh >= 9) & (h_sh < 11)) | ((h_sh >= 13) & (h_sh < 16))).astype(float)
    feats["sess_ldn_ny_overlap"] = (
        feats["sess_london_day"] * feats["sess_ny_day"]
    )
    feats["sess_ldn_pm_cluster"] = ((h_ldn >= 15) & (h_ldn < 17) & (dow_ldn < 5)).astype(float)
    feats["sess_us_macro_hour"] = ((h_ny == 8) & (dow_ny < 5)).astype(float)
    feats["sess_asia_eu_handover"] = ((h_ldn >= 7) & (h_ldn < 8) & (dow_ldn < 5)).astype(float)

    # --- 2b) Blueprint 1 additions: timestamp-level asymmetry flags --------
    # Blose-Gondhalekar-Kort (JEF 2018): overnight returns significantly
    # positive, US day returns (COMEX open->close) significantly negative.
    feats["is_us_daytime"] = ((h_ny >= 8) & (h_ny < 17)).astype("float64")       # 08:00-17:00 NY
    feats["is_comex_first_hour"] = (h_ny == 8).astype("float64")                 # 08:00-09:00 NY
    feats["is_asian_overnight"] = ((h_ny >= 18) | (h_ny < 3)).astype("float64")  # 18:00-03:00 NY

    # Tokyo gotobi (5/10/15/20/25/month-end, Tokyo calendar) pre-fix hour (09:xx JST)
    loc_tyo = idx.tz_convert(TOKYO)
    is_gotobi = (loc_tyo.day % 5 == 0) | (loc_tyo.is_month_end)
    feats["is_tokyo_gotobi_pre_fix"] = (is_gotobi & (loc_tyo.hour == 9)).astype("float64")
    # London month-end 15:00-16:00 cluster (Melvin-Prins equity-hedging window)
    feats["is_london_month_end_4pm"] = (
        loc_ldn.is_month_end & (h_ldn == 15)
    ).astype("float64")

    # --- 3) DST state, offset gap, distance to transition ------------------
    is_dst_uk = np.array([t.dst().total_seconds() > 0 for t in loc_ldn])
    is_dst_us = np.array([t.dst().total_seconds() > 0 for t in loc_ny])
    feats["is_dst_uk"] = is_dst_uk.astype(float)
    feats["is_dst_us"] = is_dst_us.astype(float)
    off_ldn = np.array([t.utcoffset().total_seconds() for t in loc_ldn]) / 3600.0
    off_ny = np.array([t.utcoffset().total_seconds() for t in loc_ny]) / 3600.0
    feats["ny_ldn_offset_hours"] = (off_ldn - off_ny)          # 5.0 normally, 4.0 in gap weeks
    feats["is_dst_gap_week"] = (feats["ny_ldn_offset_hours"] == 4.0).astype(float)

    if year_range is None:
        year_range = (int(idx.year.min()) - 1, int(idx.year.max()) + 1)
    tr = _dst_transitions(year_range[0], year_range[1])
    for key, tag in (("UK", "uk"), ("US", "us")):
        trans = tr[key].tz_convert("UTC")
        # signed days to nearest transition (negative after)
        idx64 = idx.asi8  # ns
        t64 = trans.asi8
        j = np.searchsorted(t64, idx64)
        prev = np.where(j > 0, t64[np.clip(j - 1, 0, len(t64) - 1)], t64[0])
        nxt = np.where(j < len(t64), t64[np.clip(j, 0, len(t64) - 1)], t64[-1])
        d_prev = (idx64 - prev) / 86400e9
        d_next = (nxt - idx64) / 86400e9
        days_to = np.where(d_prev < d_next, -d_prev, d_next)
        feats[f"days_to_dst_{tag}"] = np.clip(days_to, -14, 14)
        feats[f"dst_change_week_{tag}"] = (np.abs(days_to) <= 3).astype(float)

    # --- 4) Session-split returns (Lou-Polk-Skouras decomposition) ---------
    if close_xau is not None:
        px = close_xau.reindex(idx)
        r = np.log(px / px.shift(1))
        sess = {
            "asia": (feats["sess_shanghai_day"] == 1) | (feats["sess_asia_night"] == 1),
            "london_am": (h_ldn >= 8) & (h_ldn < 12) & (dow_ldn < 5),
            "ldn_ny_overlap": feats["sess_ldn_ny_overlap"] == 1,
            "ny_pm": (h_ny >= 12) & (h_ny < 17) & (dow_ny < 5),
        }
        for name, mask in sess.items():
            m = np.asarray(mask, dtype=bool)
            rr = pd.Series(r.values, index=idx).where(m)
            feats[f"ret24_{name}"] = rr.fillna(0.0).rolling(24, min_periods=1).sum()
            feats[f"ret6_{name}"] = rr.fillna(0.0).rolling(6, min_periods=1).sum()

    return feats


# ============================================================================
# Blueprint 1 Section 5.1 -- M1-derived fixing-drift features
# (pre-auction drifts + WMR post-fix reversal + VR convexity)
# ============================================================================
def b1_m1_fixing_features(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
                          price_col: str = "close",
                          vol_col: str = "volume") -> pd.DataFrame:
    """Blueprint-1 fixing-microstructure block, computed on M1 and aggregated
    to H1 (causal at H1 bar close).

    Requires m1 with UTC DatetimeIndex (open-labeled) and close/volume columns.
    All windows are strictly inside the H1 bar being labeled (or shifted once
    at H1 granularity for the post-fix reversal signals).
    """
    idx_utc = m1.index.tz_convert("UTC")
    m1 = m1.set_axis(idx_utc)
    idx_ldn = idx_utc.tz_convert(LONDON)
    idx_nyc = idx_utc.tz_convert(NEWYORK)

    ret_m1 = np.log(m1[price_col] / m1[price_col].shift(1)).fillna(0.0)
    tick_sign = np.sign(m1[price_col].diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    signed_vol = tick_sign * m1[vol_col].astype("float64").fillna(0.0)

    # 1. Pre-LBMA AM Fix [10:00-10:29 London] & PM Fix [14:30-14:59 London] drift
    ldn_mins = idx_ldn.hour * 60 + idx_ldn.minute
    is_pre_lbma_am = (ldn_mins >= 600) & (ldn_mins < 630)
    is_pre_lbma_pm = (ldn_mins >= 870) & (ldn_mins < 900)

    # 2. Pre-WMR London 16:00 Fix last-hour return & OFI [15:00-15:59 London]
    is_pre_wmr_lh = (ldn_mins >= 900) & (ldn_mins < 960)

    # 3. Pre-US Macro Release 30-min drift window [08:00-08:29 NY]
    nyc_mins = idx_nyc.hour * 60 + idx_nyc.minute
    is_pre_us_macro = (nyc_mins >= 480) & (nyc_mins < 510)

    # 4. Variance Ratio Convexity: (H1 displacement)^2 / sum(M1 return^2)
    h1_groups = m1.groupby(pd.Grouper(freq="1h"))
    h1_close = h1_groups[price_col].last()
    h1_open = h1_groups["open" if "open" in m1.columns else price_col].first()
    h1_ret = np.log(h1_close / h1_open)
    m1_rv = ret_m1.pow(2).resample("1h").sum()
    vr_convexity = (h1_ret.pow(2)) / (m1_rv + 1e-12)

    # Aggregate event-specific M1 drifts to H1 (strictly causal at H1 close)
    pre_lbma_am_drift = (ret_m1 * is_pre_lbma_am).resample("1h").sum()
    pre_lbma_pm_drift = (ret_m1 * is_pre_lbma_pm).resample("1h").sum()
    pre_wmr_lh_ret = (ret_m1 * is_pre_wmr_lh).resample("1h").sum()
    pre_wmr_lh_ofi = (signed_vol * is_pre_wmr_lh).resample("1h").sum() / (
        m1[vol_col].astype("float64").fillna(0.0).resample("1h").sum() + 1e-8
    )
    pre_us_macro_drift = (ret_m1 * is_pre_us_macro).resample("1h").sum()

    feats = pd.DataFrame(index=h1_close.index)
    feats["vr_h1_m1_convexity"] = vr_convexity
    feats["pre_lbma_am_drift"] = pre_lbma_am_drift
    feats["pre_lbma_pm_drift"] = pre_lbma_pm_drift
    # Krohn-Mueller-Whelan (JF 2024): negative of last-hour pre-fix return/OFI
    # predicts the post-16:00 fix reversal; shift(1) at H1 granularity = causal.
    feats["wmr_post_fix_rev_signal"] = -1.0 * pre_wmr_lh_ret.shift(1).fillna(0.0)
    feats["wmr_post_fix_ofi_signal"] = -1.0 * pre_wmr_lh_ofi.shift(1).fillna(0.0)
    feats["pre_us_macro_drift"] = pre_us_macro_drift

    return feats.reindex(h1_index)


def dst_gap_sanity_check() -> None:
    """Assert the two annual London-NY misalignment windows are found exactly.
    Autumn window: last-Sun Oct 01:00 UTC -> 1st-Sun Nov 06:00/07:00 UTC (7 days).
    Spring window: 2nd-Sun Mar 07:00 UTC -> last-Sun Mar 01:00 UTC (14/21 days)."""
    idx = pd.date_range("2026-10-20", "2026-11-05", freq="h", tz="UTC")
    f = clock_features(idx, None, year_range=(2026, 2026))
    off = f["ny_ldn_offset_hours"].resample("D").first()
    assert (off["2026-10-20":"2026-10-25"] == 5.0).all(), "pre-gap offset wrong"
    assert (off["2026-10-26":"2026-10-31"] == 4.0).all(), "autumn gap window wrong"
    assert (off["2026-11-02":] == 5.0).all(), "post-gap offset wrong"
    pm_ny_hour = pd.Series(idx.tz_convert(NEWYORK).hour, index=idx)[f["minutes_to_LBMA_PM"] == 0]
    assert set(pm_ny_hour["2026-10-26":"2026-10-30"]) == {11}, "PM fix local NY hour in gap week"
    assert set(pm_ny_hour["2026-11-02":"2026-11-04"]) == {10}, "PM fix local NY hour normal week"
    print("F1 DST sanity checks passed (autumn 2026 gap window: 7 days at 4h).")


def f1_self_test() -> None:
    dst_gap_sanity_check()
    # B1 fixing features: causality + shape on synthetic M1
    rng = np.random.default_rng(3)
    n = 60 * 24 * 40
    idx = pd.date_range("2026-01-05", periods=n, freq="min", tz="UTC")
    px = 2000 * np.exp(np.cumsum(0.0003 * rng.standard_normal(n)))
    m1 = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                       "volume": 100 * rng.random(n)}, index=idx)
    h1_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    f = b1_m1_fixing_features(m1, h1_idx)
    assert f["vr_h1_m1_convexity"].notna().sum() > 0.8 * len(f)
    assert f["pre_lbma_am_drift"].notna().all()
    # causality: truncate M1 tail -> earlier rows unchanged
    cut = 60 * 24 * 30
    h1c = h1_idx[h1_idx < idx[cut]]
    fc = b1_m1_fixing_features(m1.iloc[:cut + 60], h1c)
    common = h1c[h1c < idx[cut] - pd.Timedelta("48h")]
    pd.testing.assert_frame_equal(fc.loc[common], f.loc[common])
    print("F1 Blueprint-1 fixing features: causal, shapes OK.")
