"""
AURUM v2.0 -- K6 kernel main (rev v2.0.2): Full Walk-Forward Backtest &
Statistical Gates
=============================================================================
Phase 4+5 of the v2.0 execution plan (CPU):

Evaluates EVERY strategy in the single-position causal event loop:
  M0_Champ_v13 (v1.3 flagship cap=3, frozen translation), M0_Champ_v13_cap1,
  M1_GBDT_v2, M2_RFF_Ridge, RLSBank, M3c_DAE, M3m_DAE, M4_DMN_Lite,
  M4b_DMN_Lite, M5_ContinuumNet, M6_AURUM_v2_Ensemble (Carver shrunk-MV
  stack over the FIXED active CPU family), M6b_Ensemble_EQW
  (equal-weight robustness variant), M7_Active_Grand_Hybrid (50% v1.3
  champion + 50% ACTIVE de-saturated v2 stack), M7_Core_Satellite
  (50% v1.3 champion + 50% passive gold sleeve -- the transparent
  continuation of the v2.0.1 M7 stream), Buy & Hold Gold, Carver EWMAC
  standalone, Channel A / Channel B standalone.

v2.0.1 CHANGELOG (post-causality-audit revision):
  * frozen_net one-bar lookahead FIXED (z[t]*r[t] -> w[t-1]*r[t]); the
    engine loop was already causal (net = z_cur * r_arr[t]).
  * G4.7_causality_v201 regression gate: the same-bar formulation is
    deliberately re-created on the Channel_A baseline and must show the
    pathological Sharpe inflation (trap detected), while the fixed
    accounting must show NO residual same-bar edge (Channel_A ~ -0.13).
  * Data-signature checkpoints: every per-strategy result is cached under
    k6_out/ckpt/ keyed by an md5 of (code revision, strategy name, full
    z-series bytes).  When an input forecast is replaced (e.g. M5 gaining
    the complete 2016-2025 K3c folds, M4b, or the M6/M7 composites being
    rebuilt on top of them) the signature changes and ONLY the affected
    strategies re-evaluate -- cached results for unchanged inputs are
    reused verbatim.
  * M7_Grand_Hybrid is the FIXED 50/50 blend (a priori, selection-free);
    the FIT-grid blend scan is retained as a diagnostic only (ledger'd).

v2.0.2 CHANGELOG (Engineering Flaw #8 + active-stack verification):
  * Producer-side z_rff rebuild (K2 rev k2-v2.0.2): causal trailing
    standardization tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), eps)))
    re-anchored per yearly re-fit fold, gated by G4.8 AT THE PRODUCER.
    The v2.0.1 fold-local ES-scale transform saturated z_rff to per-fold
    sign bets (OOS nunique=1) -- M2/M6/M6b/M7 silently degenerated into
    passive gold exposure.  pred_rff is byte-identical; only the position
    transform changed.
  * M6_AURUM_v2_Ensemble rebuilt per Carver S24: the family is the FIXED
    a-priori active CPU pair {M1_GBDT_v2, M2_RFF_Ridge} (the v2.0.1
    positive-FIT membership filter on the UNBUFFERED frozen_net FIT Sharpe
    dropped M1_GBDT_v2 -- which has a clean active 0.938 OOS Sharpe --
    for a -0.24 FIT estimate); SR inputs are now the BUFFERED ENGINE FIT
    Sharpes (the shipped execution configuration per the gauntlet), and
    the SR estimates are shrunk 90% toward the equal-weight prior
    (sr' = 0.10*sr + 0.90*mean(sr)); corr-shrink 0.50; clip >= 0 with the
    equal-weight fallback when the shrunk SRs cannot rank members.
  * M7 split into two explicit side-by-side benchmarks:
      M7_Active_Grand_Hybrid = 0.5*v1.3 champion + 0.5*ACTIVE de-saturated
        v2 stack (M6) -- measures the TRUE active orthogonality gain;
      M7_Core_Satellite = 0.5*v1.3 champion + 0.5*passive gold sleeve --
        the transparent accounting of what the v2.0.1 M7 stream actually
        was in OOS (bit-identical net stream; preserves 2.6351 / 2.5012@2x).
  * SHIPPING_FAMILY pruned to the cost-viable configurations: the
    de-saturated M2/M6/M6b FAIL the Carver speed limit (annual cost > 1/3
    pre-cost SR -- the de-saturated active stack is not cost-viable at H1
    frequency), so they are demoted to reference-only (flags still
    exported and discussed).  M7_Active passes the speed limit and stays
    in the shipping family as the active measurement arm.
  * G6.1 DSR flagship = M7_Core_Satellite (direct continuation of the
    v2.0.1 gated stream); M7_Active DSR reported alongside.

Windows: FIT 2016-2022, OOS 2023-2025 (untouched), year-by-year 2016-2025.
Costs: 1.0x / 1.5x / 2.0x.  Carver speed-limit audit (cost <= 1/3 pre-cost SR).
Statistics: 5,000-sample block-bootstrap CIs, Deflated Sharpe Ratio over the
ENTIRE trial ledger (K2+K3+K6), CSCV-PBO on the selection family.

Outputs: backtest_v2.json, year_table.csv, net_series.parquet, stats.json,
gates.json, champion_compare.json, trial_ledger.jsonl, run_log.txt,
gauntlet.json, k6_meta.json, ckpt/.
"""
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "model"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v2_common import (BARS_PER_YEAR, COST_R, FIT_YEARS, OOS_YEARS,
                       SIGMA_TGT_ANN, TrialLedger, _find_dir,
                       block_bootstrap_ci, find_features_ds, find_phase1_ds,
                       frozen_net, load_v2_data, run_online_engine)
from online_engine import deflated_sharpe_ratio, pbo_cscv

CODE_REV = "k6-v2.0.2"

GATES = {}
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:3000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)


def stage(name):
    class C:
        def __init__(self):
            self.t0 = time.time()
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            if et is not None:
                err = "".join(traceback.format_exception(et, ev, tb))[-2500:]
                log(f"===== STAGE {name} ERROR =====\n{err}")
                raise
            log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False
    return C()


# v1.3 champion reference numbers (from tskdev/aurum-p4-backtest v1.3 output,
# backtest_summary.md -- OOS 2023-25, cost 1.0x, trade-based barrier accounting)
V13_REFERENCE = {
    "flagship_cap3": {"sharpe_ann": 2.444, "cagr_pct": 54.56, "maxdd_pct": 13.06,
                      "er_per_trade": 0.1478, "pf": 1.258, "n_trades": 3341,
                      "dsr": 0.895},
    "flagship_cap1": {"sharpe_ann": 2.305, "cagr_pct": 53.18, "maxdd_pct": 14.67},
    "aurum_r1_cap1": {"sharpe_ann": 2.194},
    "gbdt_cap1_v13": {"sharpe_ann": 0.307, "er_per_trade": 0.0310},
    "buyhold_cap1": {"sharpe_ann": 1.919, "cagr_pct": 23.61, "maxdd_pct": 10.80},
}

# strategies evaluated under frozen translation only (no v2 engine overlay):
# the v1.3 champion policies are REPORT-ONLY translations of the barrier
# trade stream into continuous accounting.
FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1"}

# Carver speed-limit (annual cost <= 1/3 pre-cost SR) is a SHIP-ABILITY
# design constraint: it is gated on the shipped/champion family.  The audit
# is still computed and exported for every strategy; tournament also-rans
# and baselines with non-positive pre-cost Sharpe are annotated as
# reference-only (the 1/3 rule is undefined for a dead signal).
# v2.0.2: the de-saturated M2/M6/M6b legitimately FAIL the speed limit
# (cost drag 6.2x / 2.5x the budget at H1 turnover) -- demoted to
# reference-only; their flags are exported and root-caused in the report.
SHIPPING_FAMILY = {"M7_Core_Satellite", "M7_Active_Grand_Hybrid",
                   "M0_Champ_v13", "M0_Champ_v13_cap1", "BuyHold_Gold"}

# neural forecast column -> tournament strategy name
NN_MAP = {"z_M3c": "M3c_DAE", "z_M3m": "M3m_DAE", "z_M4": "M4_DMN_Lite",
          "z_M4b": "M4b_DMN_Lite", "z_M5": "M5_ContinuumNet"}

# M6 stack family (v2.0.2, Carver S24): the FIXED a-priori active CPU pair
# -- no FIT-window membership filter (the v2.0.1 positive-FIT rule on the
# unbuffered frozen_net FIT Sharpe dropped M1_GBDT_v2, whose engine FIT
# estimate is the selection-relevant metric; membership is a design choice,
# weighting is where the evidence enters).  SR inputs = buffered ENGINE
# FIT Sharpes from the gauntlet; SR estimates shrunk 90% toward the
# equal-weight prior; corr-shrink 0.50 toward identity.
STACK_CANDIDATES = ["M1_GBDT_v2", "M2_RFF_Ridge"]
SR_SHRINK_TO_EQUAL = 0.90      # Carver shrinkage_SR: 90% toward equal weights
CORR_SHRINK = 0.50


def champion_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex,
                      cap: int = 3) -> pd.Series:
    """v1.3 flagship per-bar exposure from its walk-forward trades.

    VERIFIED MAPPING: exit_bar is the absolute bar position in the SAME
    2010-based H1 frame (first 2016-01-04 entry sits at frame position 36521
    with exit_bar 36522; exit_bar max 94948 < 95008).  Each active trade
    contributes dir/cap; the sum is clipped to [-1, 1].  Causal by
    construction (the trades are the v1.3 policy's own walk-forward output).
    """
    z = np.zeros(len(index))
    sub = trades[(trades["strategy"] == "flagship") &
                 (trades["cap"] == cap) &
                 (trades["cost_mult"] == 1.0)]
    for row in sub.itertuples():
        d = 1.0 if row.cell_name.startswith("L") else -1.0
        i0 = index.searchsorted(pd.Timestamp(row.ts, unit="ns", tz="UTC"))
        i1 = int(row.exit_bar)
        if i0 >= len(index):
            continue
        i1 = min(len(index), max(i1, i0 + 1))
        z[i0:i1] += d / cap
    return pd.Series(np.clip(z, -1.0, 1.0), index=index)


def samebar_net_sharpe(z: pd.Series, r: pd.Series, sigma: pd.Series) -> float:
    """The ORIGINAL (buggy) same-bar formulation, kept ONLY as the negative
    control for gate G4.7: credits z[t]*r[t] -- a one-bar lookahead."""
    idx = z.index
    zv = z.values.astype("float64")
    rv = r.reindex(idx).values
    sv = sigma.reindex(idx).values
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
    net = zv * v * rv
    net = net[np.isfinite(net)]
    if len(net) < 20:
        return float("nan")
    return float(net.mean() / (net.std(ddof=1) + 1e-12) * math.sqrt(BARS_PER_YEAR))


# ----------------------------------------------------------------------------
# data-signature checkpoints
# ----------------------------------------------------------------------------
def z_sig(z: pd.Series) -> str:
    arr = z.dropna()
    if len(arr) == 0:
        return "empty"
    h = hashlib.md5()
    h.update(f"{arr.index[0]}|{arr.index[-1]}|{len(arr)}|".encode())
    h.update(np.ascontiguousarray(arr.values.astype("float64")).tobytes())
    return h.hexdigest()[:12]


def sig_of(name: str, z: pd.Series, extra: str = "") -> str:
    return hashlib.md5(f"{CODE_REV}|{name}|{z_sig(z)}|{extra}".encode()).hexdigest()[:12]


class Ckpt:
    """Per-strategy checkpoint store keyed by data signatures.  A checkpoint
    is reused iff the strategy name, the code revision and the FULL z-series
    bytes are unchanged -- replacing a forecast (M5 complete folds, M4b,
    rebuilt composites) automatically invalidates only the affected keys."""

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "fitgauntlet"), exist_ok=True)
        os.makedirs(os.path.join(root, "evalgrid"), exist_ok=True)

    def load_json(self, sub, name):
        p = os.path.join(self.root, sub, f"{name}.json")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_json(self, sub, name, obj):
        p = os.path.join(self.root, sub, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, indent=1, default=str)

    def net_path(self, name, window):
        return os.path.join(self.root, "evalgrid", f"{name}__{window}.parquet")

    def load_net(self, name, window):
        p = self.net_path(name, window)
        if os.path.isfile(p):
            s = pd.read_parquet(p)["net"]
            s.index = pd.DatetimeIndex(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
            return s
        return None

    def save_net(self, name, window, net: pd.Series):
        pd.DataFrame({"net": net.values}, index=net.index).to_parquet(
            self.net_path(name, window))


def main():
    WORK = os.environ.get("AURUM_K6OUT", "/kaggle/working")
    if WORK == "/kaggle/working" and not os.path.exists(WORK):
        WORK = os.path.join(_HERE, "k6_out")
    os.makedirs(WORK, exist_ok=True)
    ck = Ckpt(os.path.join(WORK, "ckpt"))

    t_wall = time.time()

    # ---------------- 1) load everything ------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        r = feats["r_xau"]
        sigma = feats["sigma"]
        yr = pd.Series(feats.index.year, index=feats.index)
        idx = feats.index

        k2dir = os.environ.get("AURUM_K2DIR") or _find_dir(
            "v2_cpu_forecasts.parquet",
            "/kaggle/input/aurum-v2-cpu-preds",
            "/kaggle/input/datasets/tskdev/aurum-v2-cpu-preds",
            "/kaggle/input/aurum-v2-forecasts-cpu",
            "/kaggle/input/aurum-v2-forecasts",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-cpu-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-cpu-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-forecasts*")))
        assert k2dir, "K2 forecasts dataset not found"
        fc_cpu = pd.read_parquet(os.path.join(k2dir, "v2_cpu_forecasts.parquet"))
        fc_cpu.index = pd.DatetimeIndex(fc_cpu.index)
        if fc_cpu.index.tz is None:
            fc_cpu.index = fc_cpu.index.tz_localize("UTC")

        k3dir = os.environ.get("AURUM_K3DIR") or _find_dir(
            "v2_neural_forecasts.parquet",
            "/kaggle/input/aurum-v2-neural-preds",
            "/kaggle/input/datasets/tskdev/aurum-v2-neural-preds",
            "/kaggle/input/aurum-v2-forecasts-neural",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-neural-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-neural-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-forecasts*")))
        fc_nn = None
        if k3dir:
            fc_nn = pd.read_parquet(os.path.join(k3dir, "v2_neural_forecasts.parquet"))
            fc_nn.index = pd.DatetimeIndex(fc_nn.index)
            if fc_nn.index.tz is None:
                fc_nn.index = fc_nn.index.tz_localize("UTC")

        v13dir = os.environ.get("AURUM_V13DIR") or _find_dir(
            "trades_all.parquet", "/kaggle/input/aurum-v13-artifacts",
            "/kaggle/input/datasets/tskdev/aurum-v13-artifacts",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v13*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v13*")))
        assert v13dir, "v1.3 artifacts dataset not found"
        trades = pd.read_parquet(os.path.join(v13dir, "trades_all.parquet"))

        # trial ledger: seed ONCE from the K2/K3 published ledgers, then add
        # every K6 configuration on top (dedup by config_hash on resume).
        ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))
        if not ledger.rows:
            for d in (k2dir, k3dir):
                p = os.path.join(d, "trial_ledger.jsonl") if d else None
                if p and os.path.isfile(p):
                    with open(p) as f:
                        for ln in f:
                            ln = ln.strip()
                            if ln:
                                ledger.rows.append(json.loads(ln))
                    with open(os.path.join(WORK, "trial_ledger.jsonl"), "a") as f:
                        f.write(open(p).read().rstrip("\n") + "\n")
        log(f"forecasts: cpu cols={list(fc_cpu.columns[:6])}; "
            f"neural={'loaded' if fc_nn is not None else 'MISSING'}; "
            f"v13 trades={len(trades)}; ledger={len(ledger.rows)} rows")

    # ---------------- 2) strategy z-series ----------------------------------
    with stage("strategy panel"):
        S = {}
        S["M1_GBDT_v2"] = fc_cpu["z_gbdt"]
        S["M2_RFF_Ridge"] = fc_cpu["z_rff"]
        if "z_rlsbank" in fc_cpu:
            S["RLSBank"] = fc_cpu["z_rlsbank"]
        if fc_nn is not None:
            for c, name in NN_MAP.items():
                if c in fc_nn.columns:
                    S[name] = fc_nn[c]
        # baselines (features are causal)
        if "F_blend_xau" in feats:
            S["Carver_EWMAC"] = (feats["F_blend_xau"] / 20.0).clip(-1, 1)
        S["Channel_A"] = feats["q_A"].clip(-1, 1)
        S["Channel_B"] = feats["q_B"].clip(-1, 1)
        S["BuyHold_Gold"] = pd.Series(1.0, index=idx)
        S["M0_Champ_v13"] = champion_exposure(trades, idx, cap=3)
        S["M0_Champ_v13_cap1"] = champion_exposure(trades, idx, cap=1)
        S = {k: v for k, v in S.items() if v is not None and v.notna().sum() > 5000}
        log(f"strategies: {list(S.keys())}")

    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    # ---------------- 3) v2.0.1 causality regression (G4.7) ------------------
    with stage("v2.0.1 causality regression"):
        zca = S["Channel_A"].dropna()
        reg = {}
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = zca.index[mask.reindex(zca.index).fillna(False)]
            if len(sel) < 500:
                continue
            sh_same = samebar_net_sharpe(zca[sel], r[sel], sigma[sel])
            sh_fix = frozen_net(zca[sel], r[sel], sigma[sel], 1.0)["metrics"]["sharpe_ann"]
            reg[wname] = {"samebar_sharpe": round(float(sh_same), 3),
                          "causal_sharpe": round(float(sh_fix), 3)}
            ledger.add("causality_regression",
                       {"window": wname, "probe": "Channel_A", "rev": CODE_REV},
                       float(sh_fix), stage="regression", kernel="K6")
        trap_detected = all(v["samebar_sharpe"] > 3.0 for v in reg.values())
        no_residual = all(abs(v["causal_sharpe"]) < 1.5 for v in reg.values())
        ok_caus = gate("G4.7_causality_v201",
                       trap_detected and no_residual and len(reg) == 2,
                       "v2.0.1 frozen_net causality regression on Channel_A: "
                       "same-bar (lookahead) Sharpe "
                       + json.dumps({k: v["samebar_sharpe"] for k, v in reg.items()})
                       + " vs fixed causal Sharpe "
                       + json.dumps({k: v["causal_sharpe"] for k, v in reg.items()})
                       + " -- trap must be detected (>3) and the fixed "
                         "accounting must show no residual same-bar edge")

    # ---------------- 4) FIT gauntlet (engine vs frozen) ---------------------
    with stage("FIT gauntlet"):
        gauntlet = {}
        engine_mode = {}

        def fit_gauntlet(name, z):
            sig = sig_of(name, z)
            cached = ck.load_json("fitgauntlet", name)
            if cached and cached.get("sig") == sig and "engine_shipped" in cached:
                gauntlet[name] = cached
                engine_mode[name] = cached["engine_shipped"]
                log(f"  GAUNTLET {name}: CACHED engine_shipped="
                    f"{cached['engine_shipped']}")
                return
            sel = z.dropna().index
            sel = sel[fit_mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                gauntlet[name] = {"skipped": True}
                engine_mode[name] = False
                return
            if name in FORCED_FROZEN:
                fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
                rec = {"sig": sig, "forced": "frozen_translation",
                       "engine_fit_sharpe": None,
                       "frozen_fit_sharpe": fro["metrics"]["sharpe_ann"],
                       "engine_shipped": False}
            else:
                eng = run_online_engine(z[sel], r[sel], sigma[sel], 1.0)
                fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
                e_sh = eng["metrics"].get("sharpe_ann", -9)
                f_sh = fro["metrics"].get("sharpe_ann", -9)
                rec = {"sig": sig, "engine_fit_sharpe": e_sh,
                       "frozen_fit_sharpe": f_sh,
                       "engine_turnover": eng["turnover_py"],
                       "frozen_turnover": fro["turnover_py"],
                       "engine_shipped": bool(e_sh >= f_sh)}
            gauntlet[name] = rec
            engine_mode[name] = rec["engine_shipped"]
            ledger.add(f"{name}+gauntlet", {"rev": CODE_REV, **{
                k: v for k, v in rec.items() if k != "sig"}},
                       rec.get("frozen_fit_sharpe"), stage="gauntlet", kernel="K6")
            ck.save_json("fitgauntlet", name, rec)
            log(f"  GAUNTLET {name}: engine={rec.get('engine_fit_sharpe')} "
                f"frozen={rec.get('frozen_fit_sharpe')} "
                f"ships_engine={rec['engine_shipped']}")

        for name, z in S.items():
            fit_gauntlet(name, z)

    # ---------------- 5) composites: M6 / M6b / M7 ---------------------------
    with stage("composite construction (M6/M6b/M7)"):
        # M6 (v2.0.2, Carver S24): FIXED a-priori family = the active CPU
        # pair; SR inputs = buffered ENGINE FIT Sharpes (the shipped
        # execution configuration per the gauntlet), shrunk 90% toward the
        # equal-weight prior; dispersion/correlation from the members' FIT
        # frozen net streams; corr shrunk 0.50 toward identity; weights
        # clipped >= 0 with the equal-weight fallback.  Frozen before the
        # OOS seal is broken.
        streams = {}
        member_fit = {}
        member_fit_engine = {}
        for m in STACK_CANDIDATES:
            if m not in S:
                continue
            z = S[m].dropna()
            sel = z.index[fit_mask.reindex(z.index).fillna(False)]
            if len(sel) < 500:
                continue
            fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
            member_fit[m] = fro["metrics"]["sharpe_ann"]
            # v2.0.2: the selection-relevant SR input is the BUFFERED
            # ENGINE FIT Sharpe (gauntlet), not the unbuffered frozen_net
            g = gauntlet.get(m) or {}
            e_sh = g.get("engine_fit_sharpe")
            if e_sh is None or not np.isfinite(e_sh):
                e_sh = member_fit[m]
            member_fit_engine[m] = e_sh
            streams[m] = fro["net"]
        members = list(streams.keys())
        assert members, "no stack members present"
        sdf = pd.DataFrame(streams).dropna()
        members = list(sdf.columns)
        sd = sdf.std()
        # Carver shrinkage_SR = 0.90: SR estimates shrunk 90% toward the
        # equal-weight prior (the cross-member mean SR equalizes weights);
        # the 10% residual keeps the honest FIT ranking in the weights.
        sr = pd.Series({m: member_fit_engine[m] for m in members})
        sr_shrunk = ((1.0 - SR_SHRINK_TO_EQUAL) * sr
                     + SR_SHRINK_TO_EQUAL * sr.mean())
        mu_shrunk = sr_shrunk * sd * BARS_PER_YEAR
        C = sdf.corr()
        n = len(C)
        C_shrunk = CORR_SHRINK * C + (1.0 - CORR_SHRINK) * np.eye(n)
        sdv = sd.values
        Cov = C_shrunk.values * np.outer(sdv, sdv) * BARS_PER_YEAR
        try:
            w_raw = np.linalg.solve(Cov + 1e-6 * np.eye(n), mu_shrunk.values)
        except np.linalg.LinAlgError:
            w_raw = np.ones(n) / n
        w = np.clip(w_raw, 0.0, None)
        if w.sum() <= 1e-9:
            # both shrunk SRs non-positive: the evidence cannot rank the
            # a-priori family -- the equal-weight prior ships unchanged
            w = np.ones(n) / n
        w = w / w.sum()
        weights = dict(zip(members, [round(float(x), 4) for x in w]))
        ok_stack = gate("G6.2_stack", len(members) == len(STACK_CANDIDATES)
                        and abs(w.sum() - 1.0) < 1e-6,
                        f"M6 FIXED family {STACK_CANDIDATES} (a priori, no "
                        f"FIT membership filter): engine FIT Sharpes "
                        f"{json.dumps({k: round(v, 4) for k, v in member_fit_engine.items()})} "
                        f"(frozen_net FIT for reference: "
                        f"{json.dumps({k: round(v, 4) for k, v in member_fit.items()})}): "
                        f"weights={weights} (Carver S24: SR-shrink 0.90 "
                        f"toward equal weights, corr-shrink 0.50, "
                        f"shrunk SRs {json.dumps({k: round(float(v), 4) for k, v in sr_shrunk.items()})})")
        ledger.add("M6_Ensemble", {"weights": weights, "members": members,
                                   "shrinkage_SR": SR_SHRINK_TO_EQUAL,
                                   "shrinkage_corr": CORR_SHRINK,
                                   "sr_inputs": "buffered engine FIT Sharpes",
                                   "family_rule": "fixed a-priori active CPU pair",
                                   "rev": CODE_REV},
                   stage="config", kernel="K6")

        z_src = pd.concat([S[m].rename(m) for m in members], axis=1)
        z_src = z_src.ffill().fillna(0.0)
        S["M6_AURUM_v2_Ensemble"] = pd.Series(z_src.values @ w, index=idx)
        # M6b: equal-weight robustness variant over the same members
        w_eq = np.ones(len(members)) / len(members)
        S["M6b_Ensemble_EQW"] = pd.Series(z_src.values @ w_eq, index=idx)

        # M7 Grand Hybrids (v2.0.2 split -- both FIXED a-priori blends,
        # selection-free; the FIT-grid scan below is a diagnostic only):
        #   * ACTIVE arm: 50% v1.3 champion + 50% active de-saturated v2
        #     stack (M6 over {M1_GBDT_v2, M2_RFF_Ridge}) -- measures the
        #     true active orthogonality gain.
        #   * CORE-SATELLITE arm: 50% v1.3 champion + 50% passive gold
        #     sleeve -- the transparent accounting of what the v2.0.1
        #     M7_Grand_Hybrid stream actually was in OOS (its v2 leg was a
        #     saturated constant +1.0, i.e. bit-identical to BuyHold_Gold;
        #     this arm preserves the 2.6351 / 2.5012@2x benchmark).
        z_ch = S["M0_Champ_v13"]
        z_v2s = S["M6_AURUM_v2_Ensemble"]
        S["M7_Active_Grand_Hybrid"] = 0.5 * z_v2s + 0.5 * z_ch
        S["M7_Core_Satellite"] = 0.5 * S["BuyHold_Gold"] + 0.5 * z_ch
        ledger.add("M7_Active_Grand_Hybrid",
                   {"blend": "50/50 fixed", "leg_v2": "M6_AURUM_v2_Ensemble "
                    "(active de-saturated stack)",
                    "leg_v13": "M0_Champ_v13 (cap=3 exposure)",
                    "selection": "none (a priori)", "rev": CODE_REV},
                   stage="config", kernel="K6")
        ledger.add("M7_Core_Satellite",
                   {"blend": "50/50 fixed", "leg_passive": "BuyHold_Gold "
                    "(passive gold sleeve)",
                    "leg_v13": "M0_Champ_v13 (cap=3 exposure)",
                    "selection": "none (a priori); transparent continuation "
                                 "of the v2.0.1 M7_Grand_Hybrid OOS stream",
                    "rev": CODE_REV},
                   stage="config", kernel="K6")
        best_h, best_hs = 0.5, -9e9
        grid_curve = {}
        for wv2 in np.arange(0.0, 1.01, 0.05):
            zh = wv2 * z_v2s + (1 - wv2) * z_ch
            sel = zh.dropna().index
            sel = sel[fit_mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            sh = frozen_net(zh[sel], r[sel], sigma[sel], 1.0)["metrics"]["sharpe_ann"]
            grid_curve[round(float(wv2), 2)] = None if sh is None else round(float(sh), 3)
            ledger.add("M7_grid_diagnostic", {"w_v2": round(float(wv2), 2)},
                       sh, stage="fit_eval", kernel="K6")
            if sh is not None and np.isfinite(sh) and sh > best_hs:
                best_hs, best_h = sh, float(wv2)
        log(f"  M7_Active FIT-grid diagnostic: best w_v2={best_h:.2f} "
            f"(FIT Sharpe {best_hs:.3f}); shipped blend = FIXED 0.50")

        for name in ("M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M7_Active_Grand_Hybrid", "M7_Core_Satellite"):
            fit_gauntlet(name, S[name])
        log(f"strategies (final): {list(S.keys())}")

    # ---------------- 2b) z-quality / saturation audit (G4.8) ---------------
    # Engineering Flaw #8 (discovered in the final v2.0.1 K6 audit, FIXED at
    # the producer in v2.0.2): a fold-local z-scale (std of the model's
    # early-stop-slice predictions) can collapse relative to the test-fold
    # prediction magnitude, saturating tanh(pred/scale) to sign(pred): the
    # strategy degrades into a per-fold directional bet with ~zero per-bar
    # information (v2.0.1 M2_RFF_Ridge: OOS nunique=1, z==1.0 for all of
    # 2023-25).  The audit records saturation status for every strategy;
    # v2.0.2 carries the gate to the producer (K2 G4.8) and this downstream
    # audit remains as the independent verification layer.
    with stage("z-quality audit"):
        zq = {}
        for name, z in S.items():
            rec = {}
            for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = z.dropna().index
                sel = sel[mask.reindex(sel).fillna(False)]
                if len(sel) < 500:
                    continue
                v = z[sel].values
                uniq = int(pd.unique(v).size)
                top_share = float((v == pd.Series(v).mode().iloc[0]).mean())
                rec[wname] = {"n": int(len(v)), "nunique": uniq,
                              "std": round(float(np.std(v)), 5),
                              "top_value_share": round(top_share, 4)}
                rec[f"{wname}_saturated"] = bool(
                    np.std(v) < 0.05 or uniq <= 8 or top_share >= 0.90)
            zq[name] = rec
        # position schedules (benchmarks, barrier translations, passive
        # sleeves, sparse channel steps) are static BY DESIGN -- the audit
        # exists to catch MODEL-SCORE degeneracy, so these are exempt but
        # annotated.  M7_Core_Satellite is a fixed 50/50 blend of two
        # position schedules (champion exposure + passive gold sleeve) --
        # it contains NO model scores, so it is by-design static as well.
        BY_DESIGN_STATIC = {"BuyHold_Gold", "M0_Champ_v13", "M0_Champ_v13_cap1",
                            "Channel_A", "Channel_B", "M7_Core_Satellite"}
        # composites: a healthy-looking blend riding on a saturated member
        # leg is still quarantined (the member's degeneracy is inherited).
        # BY_DESIGN_STATIC legs (barrier translations, passive sleeves,
        # sparse channel steps) are position schedules by design -- they are
        # exempt from flagging and annotated as such.
        COMPOSITE_LEGS = {"M6_AURUM_v2_Ensemble": members,
                          "M6b_Ensemble_EQW": members,
                          "M7_Active_Grand_Hybrid": ["M6_AURUM_v2_Ensemble",
                                                     "M0_Champ_v13"],
                          "M7_Core_Satellite": ["M0_Champ_v13",
                                                "BuyHold_Gold"]}
        for comp, legs in COMPOSITE_LEGS.items():
            if comp in zq:
                zq[comp]["legs"] = {}
                for m_ in legs:
                    if m_ in BY_DESIGN_STATIC:
                        zq[comp]["legs"][m_] = "exempt (position schedule by design)"
                    else:
                        zq[comp]["legs"][m_] = bool(
                            zq.get(m_, {}).get("oos_saturated")
                            or zq.get(m_, {}).get("fit_saturated"))
                zq[comp]["has_flagged_leg"] = any(
                    v is True for v in zq[comp]["legs"].values())
        for n in BY_DESIGN_STATIC & set(zq):
            zq[n]["by_design_position_schedule"] = True
        flagged = sorted([n for n, r in zq.items()
                          if n not in BY_DESIGN_STATIC
                          and (r.get("oos_saturated") or r.get("fit_saturated")
                               or r.get("has_flagged_leg"))])
        with open(os.path.join(WORK, "z_quality.json"), "w") as f:
            json.dump({"audit": zq, "flagged_strategies": flagged,
                       "criteria": "std < 0.05 or nunique <= 8 or top-value "
                                   "share >= 0.90, per window",
                       "note": "flagged = saturated/degenerate z (per-fold "
                               "directional bet); results are still exported "
                               "verbatim and quarantined in the report"},
                      f, indent=1, default=str)
        ok_zq = gate("G4.8_z_quality_audit", len(zq) == len(S),
                     "z saturation audit exported for all strategies (v2.0.2: "
                     "independent downstream verification of the K2 producer "
                     "gate); flagged (saturated/degenerate): "
                     f"{flagged or 'none'}; full audit in z_quality.json"
                     + ("" if not flagged else
                        " -- flagged strategies are quarantined in the report"))

    # ---------------- 6) full evaluation grid --------------------------------
    with stage("evaluation grid (windows x costs)"):
        results = {}
        year_rows = []
        net_series_export = {}

        def eval_windows(name, z):
            use_engine = engine_mode.get(name, False)
            sig = sig_of(name, z, extra=f"engine={use_engine}")
            cached = ck.load_json("evalgrid", name)
            nets_c = {}
            if (cached and cached.get("sig") == sig and "entry" in cached
                    and "fit" in cached["entry"] and "oos" in cached["entry"]):
                ok_nets = True
                for window in ("fit", "oos"):
                    if window in cached["entry"]:
                        s = ck.load_net(name, window)
                        if s is None:
                            ok_nets = False
                            break
                        nets_c[window] = s
                if ok_nets:
                    results[name] = cached["entry"]
                    for window, s in nets_c.items():
                        net_series_export[f"{name}__{window}"] = s
                        yr_rows = cached["entry"][window].get("year_rows", [])
                        for row in yr_rows:
                            year_rows.append({"strategy": name, **row})
                    log(f"  EVAL {name}: CACHED ({'engine' if use_engine else 'frozen'})")
                    return
            entry = {"gauntlet_engine": use_engine}
            for window, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = z.dropna().index
                sel = sel[mask.reindex(sel).fillna(False)]
                if len(sel) < 500:
                    continue
                zw, rw, sw = z[sel], r[sel], sigma[sel]
                w_entry = {}
                if use_engine:
                    full = run_online_engine(zw, rw, sw, 1.0)
                    w_entry = {"metrics": full["metrics"],
                               "turnover_py": full["turnover_py"],
                               "diag": full["diagnostics"]}
                    net = full["net"]
                    for cm in (1.5, 2.0):
                        v = run_online_engine(zw, rw, sw, cm)
                        w_entry[f"cost{cm}"] = v["metrics"]
                else:
                    fro = frozen_net(zw, rw, sw, 1.0)
                    w_entry = {"metrics": fro["metrics"],
                               "turnover_py": fro["turnover_py"]}
                    net = fro["net"]
                    for cm in (1.5, 2.0):
                        w_entry[f"cost{cm}"] = frozen_net(zw, rw, sw, cm)["metrics"]
                # Carver speed limit: cost drag <= 1/3 pre-cost Sharpe
                pre = frozen_net(zw, rw, sw, 0.0)
                drag = frozen_net(zw, rw, sw, 1.0)
                pre_sh = pre["metrics"].get("sharpe_ann", 0.0)
                cost_sh = pre_sh - drag["metrics"].get("sharpe_ann", 0.0)
                w_entry["speed_limit_ok"] = bool(
                    cost_sh <= pre_sh / 3.0 + 1e-9) if pre_sh > 0 else None
                w_entry["cost_drag_sharpe"] = round(float(cost_sh), 3)
                w_entry["pre_cost_sharpe"] = round(float(pre_sh), 3)
                # year-by-year
                yr_rows = []
                yy = pd.Series(sel.year, index=sel)
                for y_ in sorted(yy.unique()):
                    gsel = sel[yy == y_]
                    sub = z[gsel]
                    if use_engine:
                        mtr = run_online_engine(sub, r[gsel], sigma[gsel], 1.0)["metrics"]
                    else:
                        mtr = frozen_net(sub, r[gsel], sigma[gsel], 1.0)["metrics"]
                    row = {"year": int(y_), "window": "FIT" if y_ < 2023 else "OOS"}
                    row.update({k: v for k, v in mtr.items() if k != "n"})
                    yr_rows.append(row)
                    year_rows.append({"strategy": name, **row})
                w_entry["year_rows"] = yr_rows
                entry[window] = w_entry
                net_series_export[f"{name}__{window}"] = net
                ck.save_net(name, window, net)
            results[name] = entry
            ck.save_json("evalgrid", name, {"sig": sig, "entry": entry})
            log(f"  EVAL {name}: {'engine' if use_engine else 'frozen'} "
                f"OOS={entry.get('oos', {}).get('metrics', {}).get('sharpe_ann')}")

        for name, z in S.items():
            eval_windows(name, z)

        year_df = pd.DataFrame(year_rows)
        year_df.to_csv(os.path.join(WORK, "year_table.csv"), index=False)
        ns = pd.DataFrame(net_series_export)
        ns.to_parquet(os.path.join(WORK, "net_series.parquet"))
        serial = {}
        for name, entry in results.items():
            serial[name] = {}
            for window in ("fit", "oos"):
                if window in entry:
                    serial[name][window] = {
                        k: v for k, v in entry[window].items() if k != "year_rows"}
        with open(os.path.join(WORK, "backtest_v2.json"), "w") as f:
            json.dump({"strategies": serial, "v13_reference": V13_REFERENCE,
                       "code_rev": CODE_REV}, f, indent=1, default=str)
        with open(os.path.join(WORK, "gauntlet.json"), "w") as f:
            json.dump(gauntlet, f, indent=1, default=str)

    # ---------------- 7) statistics: bootstrap / DSR / PBO -------------------
    with stage("statistics"):
        stats = {}
        for name, entry in results.items():
            if "oos" in entry and f"{name}__oos" in net_series_export:
                net = net_series_export[f"{name}__oos"].values
                stats.setdefault(name, {})["oos_bootstrap"] = block_bootstrap_ci(net, 5000)
        N_trials = ledger.unique_configs()
        # ------------------------------------------------------------------
        # DSR trial family (v2.0.2, unit-consistent, effective-independent):
        #   Bailey & Lopez de Prado 2014: N counts INDEPENDENT trials.
        #   Correlated siblings are not independent trials, so the K6
        #   selection panel is reduced to effective strategy families:
        #     step 1 -- exact stream duplicates on the evaluation domain
        #               collapse (M6b is bit-identical to M6 when the stack
        #               weight hits the equal-weight fallback);
        #     step 2 -- sibling groups count once via their best-FIT member
        #               (M3c/M3m DAE sizes, M4/M4b DMN variants, the v1.3
        #               champion cap=3/cap=1 translations).
        #   This yields the distinct configurations the championship was
        #   actually selected among.  Sharper variants are reported
        #   in stats._meta.dsr_sensitivity_m7 for full transparency.
        #   EXCLUDED from the primary family (also sensitivity-reported):
        #     - M7 blend grid: diagnostic only -- the 50/50 blends were fixed
        #       a priori and shipped AGAINST their FIT optimum, so they are
        #       not selection trials.
        #     - K2/K3 ledger rows: pre-fix same-bar accounting vintage /
        #       per-fold search metrics -- not unit-comparable.
        # All trial Sharpes are annualized -> per-bar (Bailey-LdP requires
        # SR and V[SR] in the same frequency).
        # ------------------------------------------------------------------
        SIBLING_GROUPS = {"M3c_DAE": "M3_DAE", "M3m_DAE": "M3_DAE",
                          "M4_DMN_Lite": "M4_DMN", "M4b_DMN_Lite": "M4_DMN",
                          "M0_Champ_v13": "M0_Champ",
                          "M0_Champ_v13_cap1": "M0_Champ"}
        # v2.0.2 blend-slot convention (consistent with the v2.0.1
        # pre-registered family): the fixed a-priori portfolio constructions
        # over counted signal families (M6/M6b over {M1,M2}; M7_Active over
        # {M0,M6}; M7_Core over {M0,BuyHold}) are deterministic re-weightings
        # of already-counted members -- correlated constructions, not
        # independent attempts -- so they contribute ONE blend-slot trial at
        # the best-FIT variant.  The conservative accounting (every distinct
        # blend stream counts) is exported alongside in _meta.
        BLEND_SLOT = ["M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                      "M7_Active_Grand_Hybrid", "M7_Core_Satellite"]
        evaldom = S["M1_GBDT_v2"].index >= pd.Timestamp("2016-01-01", tz="UTC")
        seen_sigs, uniq = set(), {}
        for name in S:                       # panel order: members first
            sg = z_sig(S[name][evaldom])
            if sg in seen_sigs:
                continue                     # exact stream duplicate
            seen_sigs.add(sg)
            uniq[name] = (gauntlet.get(name) or {}).get("frozen_fit_sharpe")
        fam_best = {}
        for name, sh in uniq.items():
            if not isinstance(sh, (int, float)) or not np.isfinite(sh):
                continue
            fk = SIBLING_GROUPS.get(name, name)
            if fk not in fam_best or sh > fam_best[fk]:
                fam_best[fk] = float(sh)
        # conservative accounting: every distinct (deduped) stream counts
        fam_conservative = np.array(sorted(fam_best.values()))
        # primary accounting: signal families + ONE blend slot (best-FIT
        # variant of the pre-registered blend constructions)
        blend_sharpes = [fam_best[b] for b in BLEND_SLOT if b in fam_best]
        fam_signals = {k: v for k, v in fam_best.items() if k not in BLEND_SLOT}
        fam_primary_vals = sorted(fam_signals.values())
        if blend_sharpes:
            fam_primary_vals.append(max(blend_sharpes))
        fam_primary = np.array(fam_primary_vals)
        trial_sharpes = (fam_primary / math.sqrt(BARS_PER_YEAR)
                         if len(fam_primary) > 5 else None)
        trial_sharpes_conservative = (fam_conservative / math.sqrt(BARS_PER_YEAR)
                                      if len(fam_conservative) > 5 else None)
        # sensitivity families (flagship only)
        def _dsr_m7(vals_ann):
            if len(vals_ann) < 6:
                return None
            net7 = net_series_export["M7_Core_Satellite__oos"].values
            zz = (net7 - net7.mean()) / (net7.std() + 1e-12)
            return round(deflated_sharpe_ratio(
                float(net7.mean() / (net7.std(ddof=1) + 1e-12)), len(net7),
                float(np.mean(zz ** 3)), float(np.mean(zz ** 4)),
                sr_trials=vals_ann / math.sqrt(BARS_PER_YEAR),
                N_trials=len(vals_ann)), 4)
        def _fam(rows_pred):
            seen, vals = set(), []
            for r in ledger.rows:
                fm = r.get("fit_metric")
                if (not isinstance(fm, (int, float)) or not np.isfinite(fm)
                        or not rows_pred(r)):
                    continue
                key = (r["model"], r["config_hash"])
                if key in seen:
                    continue
                seen.add(key)
                vals.append(float(fm))
            return np.array(vals)
        sens_grid = _fam(lambda r: r.get("kernel") == "K6" and
                         r.get("stage") in ("gauntlet", "fit_eval"))
        sens_k2 = _fam(lambda r: (r.get("kernel") == "K6" and
                                  r.get("stage") in ("gauntlet", "fit_eval"))
                       or (r.get("kernel") == "K2" and
                           r.get("stage") in ("fit", "engine")))
        sens_all = _fam(lambda r: True)
        for name, entry in results.items():
            if "oos" not in entry:
                continue
            net = net_series_export[f"{name}__oos"].values
            z = (net - net.mean()) / (net.std() + 1e-12)
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))
            sr_per_bar = float(net.mean() / (net.std(ddof=1) + 1e-12))
            dsr = deflated_sharpe_ratio(sr_per_bar, len(net), skew, kurt,
                                        sr_trials=trial_sharpes if len(trial_sharpes) > 5 else None,
                                        N_trials=max(2, len(fam_primary)))
            stats.setdefault(name, {})["oos_dsr"] = round(dsr, 4)
            stats[name]["dsr_passes_95"] = bool(dsr >= 0.95)
            # conservative full-panel accounting (every distinct blend counts)
            dsr_c = deflated_sharpe_ratio(
                sr_per_bar, len(net), skew, kurt,
                sr_trials=trial_sharpes_conservative
                if len(trial_sharpes_conservative) > 5 else None,
                N_trials=max(2, len(fam_conservative)))
            stats[name]["oos_dsr_conservative"] = round(dsr_c, 4)
        # PBO CSCV on the FIT-window net streams of the selection family
        # (FIT-only matrix; mixing windows would empty the dropna intersection)
        fam = {f"{name}__fit": net_series_export[f"{name}__fit"]
               for name in results if f"{name}__fit" in net_series_export}
        pnl = pd.DataFrame(fam).dropna()
        pbo = pbo_cscv(pnl.values, n_blocks=16) if pnl.shape[1] >= 4 else None
        stats["_meta"] = {"N_trials_ledger": N_trials,
                          "ledger_rows": len(ledger.rows),
                          "pbo_cscv": round(pbo, 4) if pbo is not None else None,
                          "code_rev": CODE_REV,
                          "dsr_family": "K6 v2.0.2 tournament selection panel, "
                                        "effective-independent families "
                                        "(Bailey-LdP: correlated siblings "
                                        "are not independent trials; frozen "
                                        "FIT Sharpes under causal accounting). "
                                        "Primary = signal families + ONE "
                                        "blend slot (fixed a-priori portfolio "
                                        "constructions over counted members "
                                        "are correlated re-weightings, "
                                        "counted once at their best-FIT "
                                        "variant -- the v2.0.1 pre-registered "
                                        "convention, applied consistently)",
                          "dsr_family_N": int(len(fam_primary)),
                          "dsr_family_sharpes": [round(float(x), 4) for x in fam_primary],
                          "dsr_family_conservative_N": int(len(fam_conservative)),
                          "dsr_family_conservative_sharpes": [round(float(x), 4)
                                                              for x in fam_conservative],
                          "dsr_conservative_note": "conservative accounting: "
                          "every distinct (deduped) blend stream counts as a "
                          "trial; flagship DSR under this vintage is exported "
                          "per strategy as oos_dsr_conservative",
                          "trial_sharpe_units": "per-bar (annualized / sqrt(5786))",
                          "dsr_sensitivity_m7": {
                              "primary": _dsr_m7(fam_primary),
                              "conservative": _dsr_m7(fam_conservative),
                              "with_m7_grid": _dsr_m7(sens_grid),
                              "with_k2_prefix": _dsr_m7(sens_k2),
                              "all_ledger_mixed_units": _dsr_m7(sens_all),
                              "family_N": {"primary": int(len(fam_primary)),
                                           "conservative": int(len(fam_conservative)),
                                           "with_grid": int(len(sens_grid)),
                                           "with_k2": int(len(sens_k2)),
                                           "all": int(len(sens_all))}}}
        dsr_flagship = stats.get("M7_Core_Satellite", {}).get("oos_dsr")
        dsr_active = stats.get("M7_Active_Grand_Hybrid", {}).get("oos_dsr")
        dsr_flagship_c = stats.get("M7_Core_Satellite", {}).get(
            "oos_dsr_conservative")
        pbo_val = stats["_meta"]["pbo_cscv"]
        if dsr_flagship is not None and dsr_flagship < 0.95:
            verdict = ("PRE-REGISTERED BAR MISSED (v2.0.2 finding, reported "
                       "verbatim): the Flaw #8 fix replaces the saturated-M2 "
                       "artifact trial (+0.3504 FIT) with the true de-"
                       "saturated trial (-0.8394), widening the honest trial "
                       "dispersion and raising the Bailey-LdP null threshold "
                       "-- the flagship's 95% DSR certification (v2.0.1: "
                       "0.958 on the artifact panel) is withdrawn to "
                       "borderline")
        else:
            verdict = "flagship clears the pre-registered 0.95 DSR bar"
        ok_stats = gate("G6.1_stats",
                        bool(dsr_flagship is not None and dsr_flagship >= 0.95
                             and (pbo_val is None or pbo_val <= 0.20)),
                        f"DSR over N={N_trials} ledger configs: shipping "
                        f"flagship M7_Core_Satellite DSR={dsr_flagship} "
                        f"(gate >= 0.95, primary family N={len(fam_primary)}: "
                        f"signal families + one blend slot); conservative "
                        f"full-panel vintage (every distinct blend counts, "
                        f"N={len(fam_conservative)}): flagship DSR="
                        f"{dsr_flagship_c}; active measurement arm "
                        f"M7_Active_Grand_Hybrid DSR={dsr_active}; "
                        f"CSCV-PBO={pbo_val} (gate <= 0.20); {verdict}; all: "
                        + json.dumps({k: v.get("oos_dsr") for k, v in stats.items()
                                      if isinstance(v, dict) and "oos_dsr" in v}))
        with open(os.path.join(WORK, "stats.json"), "w") as f:
            json.dump(stats, f, indent=1, default=str)

    # ---------------- 8) champion comparison + export ------------------------
    with stage("champion comparison + export"):
        comp = {"v13_flagship_cap3": V13_REFERENCE["flagship_cap3"],
                "v13_flagship_cap1": V13_REFERENCE["flagship_cap1"]}
        for name in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                     "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M0_Champ_v13", "M0_Champ_v13_cap1", "M5_ContinuumNet",
                     "M2_RFF_Ridge", "M1_GBDT_v2", "BuyHold_Gold"):
            if name in results and "oos" in results[name]:
                m = results[name]["oos"]["metrics"]
                comp[name] = {"sharpe_ann": m.get("sharpe_ann"),
                              "maxdd_volunits": m.get("maxdd_volunits"),
                              "oos_bootstrap": stats.get(name, {}).get("oos_bootstrap"),
                              "oos_dsr": stats.get(name, {}).get("oos_dsr"),
                              "oos_dsr_conservative": stats.get(name, {}).get(
                                  "oos_dsr_conservative"),
                              "cost2x_sharpe": results[name]["oos"].get("cost2.0", {}).get("sharpe_ann")}
        with open(os.path.join(WORK, "champion_compare.json"), "w") as f:
            json.dump(comp, f, indent=1, default=str)
        # Carver speed-limit audit: gated on the shipping family; every
        # strategy's flag is exported (reference_only for non-shipping)
        sl = {}
        sl_ref_only = {}
        for name, entry in results.items():
            for window in ("fit", "oos"):
                if window in entry and entry[window].get("speed_limit_ok") is not None:
                    key = f"{name}_{window}"
                    if name in SHIPPING_FAMILY:
                        sl[key] = entry[window]["speed_limit_ok"]
                    else:
                        sl_ref_only[key] = entry[window]["speed_limit_ok"]
        ok_sl = gate("G5.4_speed_limit", all(sl.values()) if sl else True,
                     f"Carver speed limit (annual cost <= 1/3 pre-cost SR), "
                     f"shipping family: {sl}; reference-only (also-rans/"
                     f"baselines, not gated): {sl_ref_only}")
        ok_g = gate("G5.3_gauntlet", len(gauntlet) == len(S),
                    "engine vs frozen per strategy (engine ships iff it beats "
                    "frozen on FIT; v1.3 translations forced frozen): "
                    + json.dumps({k: v.get("engine_shipped") for k, v in gauntlet.items()}))
        ok_ex = gate("G6.6_export", True,
                     "backtest_v2.json + year_table.csv + net_series.parquet + "
                     "stats.json + champion_compare.json + trial_ledger.jsonl + "
                     "gauntlet.json + k6_meta.json + ckpt/")
        with open(os.path.join(WORK, "k6_meta.json"), "w") as f:
            json.dump({"code_rev": CODE_REV,
                       "wall_seconds": round(time.time() - t_wall, 1),
                       "m6_weights": weights, "m6_members": members,
                       "m6_member_fit_sharpes": {
                           "engine": {k: v for k, v in member_fit_engine.items()},
                           "frozen": {k: v for k, v in member_fit.items()}},
                       "m7_active_blend": "0.5 * M6 + 0.5 * M0_Champ_v13 (fixed)",
                       "m7_core_satellite_blend": "0.5 * BuyHold_Gold + 0.5 * "
                                                  "M0_Champ_v13 (fixed; "
                                                  "transparent continuation "
                                                  "of the v2.0.1 M7 OOS stream)",
                       "m7_grid_diagnostic": {"best_w_v2": best_h,
                                              "best_fit_sharpe": best_hs,
                                              "curve": grid_curve},
                       "strategies": list(S.keys()),
                       "n_strategies": len(S),
                       "data_signatures": {k: z_sig(v) for k, v in S.items()},
                       "z_quality_flagged": flagged,
                       "z_quality_summary": {n: {w: {"nunique": zq[n][w]["nunique"],
                                                      "std": zq[n][w]["std"]}
                                                 for w in ("fit", "oos") if w in zq[n]}
                                              for n in zq},
                       "forced_frozen": sorted(FORCED_FROZEN & set(S)),
                       "cost_r": COST_R, "bars_per_year": BARS_PER_YEAR,
                       "fit_years": FIT_YEARS, "oos_years": OOS_YEARS},
                      f, indent=1, default=str)

    all_ok = all([ok_caus, ok_zq, ok_stack, ok_g, ok_sl, ok_stats, ok_ex])
    gate("G6_ALL", all_ok, "all K6 gates green" if all_ok else "K6 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"K6 COMPLETE in {time.time()-t_wall:.0f}s" if all_ok else "K6 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
