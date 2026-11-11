"""
AURUM v2.0 -- Feature Store Builder (orchestrator, UNIFIED)
============================================================
Synthesizes the feature engines of BOTH blueprints into one strictly causal
H1 feature frame over the existing Kaggle dataset ``tskdev/aurum-phase1-data``
(M1 + H1 parquets for XAUUSD/XAGUSD/EURUSD/USDJPY + FRED macro embedded in
h1_features.parquet as mi_* columns), plus the v2.0 continuous label targets.

Blueprint 2 Section 3.7 builder + Blueprint 1 additions wired in:
  * f1_clock.b1_m1_fixing_features  (pre-auction drifts, WMR post-fix
    reversal, VR convexity) -- Blueprint 1 Section 5.1
  * f6_trend.compute_cfm_and_ahl_trend_features + fast_causal_changepoint_features
    (CFM tanh, AHL damped MACD composite, multi-scale norm returns, CPD
    severity/location at 21/126) -- Blueprint 1 Section 5.2
  * v1.2 era-safe FRED macro innovations (mi_*) carried over verbatim from
    h1_features.parquet so no macro context is lost.

IO CONTRACT (adapted to the actual on-disk layout of tskdev/aurum-phase1-data:
M1/H1 parquet column names are o/h/l/c/v, open-labeled UTC indices):

  input : /kaggle/input/tskdev/aurum-phase1-data/{m1,h1}_{xauusd,...}.parquet
          /kaggle/input/tskdev/aurum-phase1-data/h1_features.parquet (mi_*)
  output: aurum_v2_features.parquet   (one row per H1 bar, all features)

Label targets (v2.0 -- continuous, cost-aware, no 48-cell grid):
  y_ret24_norm  : next-24-bar vol-normalized XAUUSD return (primary aux target)
  y_dir24       : sign of next-24-bar return (binary aux target)
  y_ret1_norm   : next-1-bar vol-normalized return (fast aux target)
  The PRIMARY training signal is the differentiable net-Sharpe loss over the
  model's continuous position (Spec Section 4) -- these targets only steer
  auxiliary heads.
"""

import numpy as np
import pandas as pd
from f1_clock import clock_features, b1_m1_fixing_features
from f2_realized import realized_block_m1, range_block_h1, variance_ratio_block, roughness_block, har_aggregates
from f3_signatures import signature_features
from f4_toxicity import vpin_block, kyle_lambda_block, amihud_clv_block
from f5_residual import residual_channels
from f6_trend import trend_ladder, compute_cfm_and_ahl_trend_features, fast_causal_changepoint_features


# ----------------------------------------------------------------------------
# IO contract -- the only lines adapted to the on-disk layout
# ----------------------------------------------------------------------------
M1_PATHS = {
    "xau": "m1_xauusd.parquet",
    "xag": "m1_xagusd.parquet",
    "eur": "m1_eurusd.parquet",
    "jpy": "m1_usdjpy.parquet",
}
H1_PATHS = {k: p.replace("m1_", "h1_") for k, p in M1_PATHS.items()}
H1_FEATURES_PATH = "h1_features.parquet"     # v1.2 frame carrying mi_* columns
OUTPUT_PATH = "aurum_v2_features.parquet"

HORIZON = 24           # primary label horizon (bars)
SIG_SPAN = 96          # vol-normalization span for targets

# canonical column mapping (v1.x layout o/h/l/c/v -> blueprint layout)
_COLMAP = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df = df.sort_index()
    df = df.rename(columns={c: _COLMAP.get(c, c) for c in df.columns})
    return df


def build_v2_features(m1: dict[str, pd.DataFrame], h1: dict[str, pd.DataFrame],
                      depth3: bool = False,
                      mi_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the full v2.0 feature frame. m1/h1: {'xau': df, 'xag': df,
    'eur': df, 'jpy': df} with UTC DatetimeIndex and open/high/low/close/volume.
    mi_frame: optional v1.2 macro-innovation frame (mi_* columns) aligned to
    the H1 index -- carried over verbatim when provided."""
    # H1 master frame: XAUUSD bars define the index; context closes joined
    base = h1["xau"].copy()
    for k in ("xag", "eur", "jpy"):
        base[f"close_{k}"] = h1[k]["close"].reindex(base.index)
    h1_index = base.index

    blocks: list[pd.DataFrame] = []

    # ---- F1: DST-aware exchange clock (B2) + B1 timestamp flags ----------
    blocks.append(clock_features(h1_index, base["close"]))
    # ---- F1b: Blueprint-1 M1 fixing features (pre-auction drift etc.) -----
    blocks.append(b1_m1_fixing_features(m1["xau"], h1_index))

    # ---- F2: realized geometry (XAU + context) ---------------------------
    for k in ("xau", "xag", "eur", "jpy"):
        rb = realized_block_m1(m1[k], h1_index).add_suffix(f"_{k}")
        blocks.append(rb)
    rng_xau = range_block_h1(base[["open", "high", "low", "close"]])
    blocks.append(rng_xau.add_suffix("_xau"))
    r_xau = np.log(base["close"] / base["close"].shift(1))
    blocks.append(variance_ratio_block(r_xau).add_suffix("_xau"))
    rv = realized_block_m1(m1["xau"], h1_index)["RV_1h"]
    blocks.append(roughness_block(rv.reindex(h1_index)).to_frame("H_rough_vol_xau"))
    # HAR aggregates of the core XAU realized block
    rb_xau = realized_block_m1(m1["xau"], h1_index)
    blocks.append(har_aggregates(rb_xau[["RV_1h", "BV_1h", "SJ_1h", "RVnegShare_1h",
                                         "jumpShare_1h"]]))

    # ---- F3: multi-asset signatures ---------------------------------------
    m1_multi = m1["xau"][["close"]].copy()
    m1_multi["close_xag"] = m1["xag"]["close"].reindex(m1_multi.index).ffill()
    m1_multi["close_eur"] = m1["eur"]["close"].reindex(m1_multi.index).ffill()
    m1_multi["close_jpy"] = m1["jpy"]["close"].reindex(m1_multi.index).ffill()
    blocks.append(signature_features(
        m1_multi, h1_index,
        {"xau": "close", "xag": "close_xag", "eur": "close_eur", "jpy": "close_jpy"},
        depth3=depth3))

    # ---- F4: toxicity / liquidity ----------------------------------------
    vol_col = "volume" if "volume" in m1["xau"].columns else None
    if vol_col:
        blocks.append(vpin_block(m1["xau"], h1_index).add_suffix("_xau"))
        blocks.append(kyle_lambda_block(m1["xau"], h1_index).add_suffix("_xau"))
        blocks.append(amihud_clv_block(m1["xau"], base, vol_col=vol_col).add_suffix("_xau"))

    # ---- F5: online cross-asset residualization --------------------------
    blocks.append(residual_channels(
        base, {"xau": "close", "jpy": "close_jpy", "eur": "close_eur", "xag": "close_xag"}))

    # ---- F6: saturating trend ladder (B2) + B1 regime/trend/CPD ----------
    blocks.append(trend_ladder(base["close"]).add_suffix("_xau"))
    blocks.append(compute_cfm_and_ahl_trend_features(base["close"]).add_suffix("_xau"))
    blocks.append(fast_causal_changepoint_features(r_xau).add_suffix("_xau"))

    # ---- v1.2 era-safe FRED macro innovations (carried over verbatim) ----
    if mi_frame is not None:
        mi = mi_frame.reindex(h1_index)
        blocks.append(mi.add_prefix("") if any(c.startswith("mi_") for c in mi.columns)
                      else mi)

    feats = pd.concat(blocks, axis=1)

    # ---- targets (strictly forward-looking; drop at train time by shift) --
    sig = r_xau.ewm(span=SIG_SPAN, min_periods=24).std()
    fwd24 = base["close"].shift(-HORIZON) / base["close"] - 1.0
    fwd1 = base["close"].shift(-1) / base["close"] - 1.0
    feats["y_ret24_norm"] = fwd24 / (sig * np.sqrt(HORIZON) + 1e-12)
    feats["y_dir24"] = np.sign(fwd24)
    feats["y_ret1_norm"] = fwd1 / (sig + 1e-12)
    feats["sig_ret_xau_target"] = sig
    return feats


def feature_groups(feats: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Semantic group slices for ContinuumNet's VSN, aligned to the built
    feature frame (Spec Section 3.0 families, minus the label columns)."""
    cols = [c for c in feats.columns if not c.startswith(("y_", "sig_ret_xau_target"))]
    prefixes = [
        ("minutes_to_", "clock"), ("in_pm_fix", "clock"), ("pre_pm", "clock"),
        ("sess_", "sessions"), ("is_", "sessions"), ("days_to_dst", "clock"),
        ("dst_change", "clock"), ("ret24_", "sessions"), ("ret6_", "sessions"),
        ("ny_ldn", "clock"), ("vr_h1_m1", "clock"), ("pre_lbma", "clock"),
        ("pre_us_macro", "clock"), ("wmr_post", "clock"),
        ("RV_", "realized"), ("BV_", "realized"), ("MedRV_", "realized"),
        ("MinRV_", "realized"), ("jump", "realized"), ("logRV", "realized"),
        ("SJ_", "realized"), ("parkinson", "realized"), ("garman", "realized"),
        ("rogers", "realized"), ("gap_", "realized"), ("YZ_", "realized"),
        ("gk_minus", "realized"), ("VR", "realized"), ("hurst", "realized"),
        ("H_rough", "realized"), ("_har", "realized"),
        ("sig1_", "signatures"), ("sig2_", "signatures"), ("levy_", "signatures"),
        ("ll_", "signatures"),
        ("vpin", "toxicity"), ("kyle_", "toxicity"), ("amihud_", "toxicity"),
        ("clv", "toxicity"), ("tick_volume", "toxicity"),
        ("eps", "residual"), ("pred", "residual"), ("R2_ew", "residual"),
        ("beta_", "residual"), ("sig_eps", "residual"), ("res_mom", "residual"),
        ("spread_", "residual"), ("ou_", "residual"), ("sig_eq", "residual"),
        ("q_", "residual"), ("chB", "residual"), ("conflict", "residual"),
        ("ewmac", "trend"), ("cfm_", "trend"), ("F_blend", "trend"), ("fdm", "trend"),
        ("ahl_macd", "trend"), ("norm_ret_", "trend"), ("ex_ante_vol", "trend"),
        ("cpd_", "regime"),
        ("mi_", "macro"),
    ]
    groups: dict[str, list[str]] = {}
    for c in cols:
        placed = False
        for pref, gname in prefixes:
            if c.startswith(pref):
                groups.setdefault(gname, []).append(c)
                placed = True
                break
        if not placed:
            groups.setdefault("other", []).append(c)
    slices = []
    start = 0
    # preserve a deterministic order: clock, sessions, realized, signatures,
    # toxicity, residual, trend, regime, macro, other
    order = ["clock", "sessions", "realized", "signatures", "toxicity",
             "residual", "trend", "regime", "macro", "other"]
    ordered_cols = []
    for g in order:
        ordered_cols.extend(sorted(groups.get(g, [])))
    # NOTE: group slices refer to the FULL frame column order; the caller must
    # reorder feats to ordered_cols before training. We return slices into the
    # REORDERED frame.
    for g in order:
        members = [c for c in ordered_cols if c in set(groups.get(g, []))]
        if members:
            slices.append((start, start + len(members), g))
            start += len(members)
    return slices, ordered_cols


def builder_integration_test() -> None:
    """End-to-end on synthetic 4-asset data: shapes, NaN discipline, causality
    of the full assembled frame, and feature-count report."""
    rng = np.random.default_rng(42)
    n = 60 * 24 * 90            # 90 days of M1
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f1 = rng.standard_normal(n) * 0.0009
    f2 = rng.standard_normal(n) * 0.0006
    common = 0.5 * f1 + 0.3 * f2
    px = {
        "xau": 2000 * np.exp(np.cumsum(0.8 * common + rng.standard_normal(n) * 0.0004)),
        "xag": 24 * np.exp(np.cumsum(1.9 * common + rng.standard_normal(n) * 0.0012)),
        "eur": 1.1 * np.exp(np.cumsum(0.4 * f2 + rng.standard_normal(n) * 0.0002)),
        "jpy": 150 * np.exp(np.cumsum(-0.6 * f1 + rng.standard_normal(n) * 0.0002)),
    }
    m1 = {}
    for k, p in px.items():
        m1[k] = pd.DataFrame(
            {"open": p, "high": p, "low": p, "close": p,
             "volume": 100 + rng.random(n) * 50}, index=idx)
    h1 = {}
    for k, df in m1.items():
        agg = df.resample("h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        h1[k] = agg
    feats = build_v2_features(m1, h1)
    T = len(feats)
    print(f"integration: {T} H1 bars x {feats.shape[1]} features")
    nan_frac = feats.isna().mean()
    warm_ok = (nan_frac.iloc[2:] < 0.9).mean()
    print(f"features with >90% NaN: {(nan_frac > 0.9).sum()} "
          f"(expected: only deep-warmup HAR/YZ columns); median NaN {nan_frac.median():.3f}")
    assert warm_ok > 0.9, "too many all-NaN feature columns"
    assert feats["y_ret24_norm"].notna().sum() > 0.8 * T
    # causality of the FULL frame: truncate M1 tail, earlier rows identical
    cut = 60 * 24 * 60
    h1c = {k: df.iloc[: df.index.searchsorted(idx[cut])] for k, df in h1.items()}
    m1c = {k: df.iloc[: cut + 60] for k, df in m1.items()}
    feats_c = build_v2_features(m1c, h1c)
    common_idx = feats_c.index[(feats_c.index < idx[cut] - pd.Timedelta("48h")) &
                               (feats_c.index > idx[60 * 24 * 10])]
    a = feats.loc[common_idx].dropna(axis=1)
    b = feats_c.loc[common_idx].dropna(axis=1)
    shared = a.columns.intersection(b.columns)
    pd.testing.assert_frame_equal(a[shared], b[shared])
    print(f"full-pipeline causality check passed on {len(common_idx)} bars "
          f"({len(shared)} shared columns)")
    print("feature-group counts:")
    for pref, n_pat in [("minutes_to", 0), ("sess_", 0), ("sig", 0), ("levy", 0),
                        ("ll_", 0), ("RV", 0), ("ewmac", 0), ("beta", 0),
                        ("q_", 0), ("vpin", 0), ("cpd_", 0), ("ahl_", 0), ("y_", 0)]:
        cnt = sum(c.startswith(pref) for c in feats.columns)
        if cnt:
            print(f"  {pref:<12} {cnt}")
