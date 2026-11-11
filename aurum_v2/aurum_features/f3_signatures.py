"""
AURUM v2.0 -- Feature Module F3: Depth-2/3 Multi-Asset Path Signatures
======================================================================
Blueprint 2 Section 3.3, verbatim (no causality defects in the original).

Theory (Chevyrev & Kormilitzin, "A primer on the signature method in machine
learning", arXiv:1603.03788; Morrill et al. arXiv:2006.00873; Reizenstein &
Graham arXiv:1802.08252):

  Signature of a d-dim path X over [a,b]:  S^{i}_{a,t} = X^i_t - X^i_a,
  S^{i,j}_{a,t} = int_a^t S^i_{a,s} dX^j_s,  S^{i_1..i_k} = iterated integrals
  over the simplex -> the ORDER OF EVENTS is encoded (cross terms).

  Levy area (2D):  A^{ij} = 1/2 ( S^{i,j} - S^{j,i} ) = 1/2 int (X^i dX^j - X^j dX^i).
  For a closed counter-clockwise loop, A = enclosed area.  A > 0 iff channel i
  "leads" the co-movement with channel j in the CCW sense; sign flips under
  path reversal -- exactly the who-moved-first detector OHLC aggregates destroy.

  Time-augmentation is MANDATORY (otherwise the signature is invariant to
  time-reparametrisation and blind to speed): channel 0 of the path is
  normalized time.

  Exact streaming recursion for the piecewise-linear interpolated path
  (Chen's identity on each linear segment, tensor-exponential increment):
    S1_i  <- S1_i  + dx_i
    S2_ij <- S2_ij + S1_i * dx_j + 0.5 * dx_i * dx_j
    S3_ijk<- S3_ijk+ S2_ij * dx_k + 0.5 * S1_i * dx_j * dx_k + dx_i dx_j dx_k / 6
  (level-3 term uses the PRE-update S1, S2 -- order matters.)

  Term counts (incl. the constant 1, which we drop): depth-N full signature
  has (d^(N+1)-1)/(d-1) terms; d=5 depth 2 -> 30, depth 3 -> 155; log-signature
  depth 2 -> d + d(d-1)/2 = 15 (increments + Levy areas).

Channels (default): normalized time + per-symbol z-scored M1 log-price
increments for XAUUSD, XAGUSD, EURUSD, USDJPY  => d=5.  Windows: the 60 M1
bars inside each H1 bar.  Missing M1 bars are forward-filled (zero increments)
-- documented convention; coverage statistics are emitted alongside.

The depth-3 block is optional (155 features); default output is the depth-2
full signature (30) + the 10 pairwise Levy areas + lead-lag pair areas
(Flint-Lyons-Turner lead-lag transform for (xau,jpy) and (xau,xag)).
"""

import numpy as np
import pandas as pd

STEPS_PER_HOUR = 60


def _hourly_path_tensor(m1: pd.DataFrame, channels: dict[str, str],
                        h1_index: pd.DatetimeIndex,
                        zscore_span: int = 1440) -> np.ndarray:
    """Build the (T_h1, 60, d) path tensor with channel 0 = normalized time.

    Parameters
    ----------
    m1 : M1 frame, UTC DatetimeIndex (open-labeled), column names given by
         ``channels`` mapping channel-name -> m1 column (PRICE levels).
    zscore_span : trailing M1 bars for per-channel increment z-scoring (24h).
    """
    cols = []
    for name in channels:
        if name == "time":
            continue
        px = m1[channels[name]].astype("float64")
        inc = np.log(px / px.shift(1))
        mu = inc.rolling(zscore_span, min_periods=60).mean()
        sd = inc.rolling(zscore_span, min_periods=60).std()
        cols.append(((inc - mu) / (sd + 1e-12)).rename(name))
    incs = pd.concat(cols, axis=1)

    # full hourly x 60-minute grid; missing minutes are ZERO increments
    # (documented Morrill et al. convention -- path does not move when
    #  no data arrives; the blueprint's literal .ffill() repeated the
    #  last increment through gaps, caught by K1's G1.5 unit check)
    full = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    grid = incs.reindex(full).fillna(0.0)
    # hours present in h1_index only
    T = len(h1_index)
    d = len(channels)  # includes time channel
    X = np.zeros((T, STEPS_PER_HOUR, d), dtype=np.float32)
    # normalized time channel 0: k/59 in [0,1] per hour (fixed grid)
    X[:, :, 0] = np.linspace(0.0, 1.0, STEPS_PER_HOUR, dtype=np.float32)[None, :]
    for j, name in enumerate([n for n in channels if n != "time"]):
        vals = grid[name].values
        # for each H1 label t, the 60 M1 bars are [t, t+59min]
        starts = h1_index.asi8 // 60_000_000_000  # ns -> minutes since epoch
        full_starts = full.asi8 // 60_000_000_000
        pos = np.searchsorted(full_starts, starts)
        ok = (pos >= 0) & (pos + STEPS_PER_HOUR <= len(vals))
        pos_c = np.clip(pos, 0, len(vals) - STEPS_PER_HOUR)
        raw = np.lib.stride_tricks.sliding_window_view(vals, STEPS_PER_HOUR)
        X[ok, :, 1 + j] = raw[pos_c][ok].astype(np.float32)
    return X


def signature_depth2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact depth-2 signature of the cumulative path of per-step increments.

    Returns (S2, levycli) with S2 (T, d + d*d) = [level-1 terms | level-2
    terms flattened (i-major)] and the (T, d, d) level-2 tensor (for areas).
    """
    T, n, d = X.shape
    S1 = np.zeros((T, d), dtype=np.float64)
    S2 = np.zeros((T, d, d), dtype=np.float64)
    for k in range(n):
        dx = X[:, k, :].astype(np.float64)                     # (T, d)
        S2 += np.einsum("ti,tj->tij", S1, dx, optimize=True)   # S1_{k-1} dx_k
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx, optimize=True)
        S1 += dx
    flat = np.concatenate([S1, S2.reshape(T, d * d)], axis=1)
    return flat.astype(np.float32), S2


def signature_depth3(X: np.ndarray) -> np.ndarray:
    """Exact depth-3 signature (adds d^3 third-level terms).  Memory: T x d^3
    float64 -> use float32 and chunk if T is large."""
    T, n, d = X.shape
    S1 = np.zeros((T, d), dtype=np.float64)
    S2 = np.zeros((T, d, d), dtype=np.float64)
    S3 = np.zeros((T, d, d, d), dtype=np.float64)
    for k in range(n):
        dx = X[:, k, :].astype(np.float64)
        S3 += np.einsum("tij,tl->tijl", S2, dx, optimize=True)
        S3 += 0.5 * np.einsum("ti,tj,tl->tijl", S1, dx, dx, optimize=True)
        S3 += (1.0 / 6.0) * np.einsum("ti,tj,tl->tijl", dx, dx, dx, optimize=True)
        S2 += np.einsum("ti,tj->tij", S1, dx, optimize=True)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx, optimize=True)
        S1 += dx
    return np.concatenate([S1, S2.reshape(T, d * d), S3.reshape(T, d ** 3)],
                          axis=1).astype(np.float32)


def levy_areas(S2: np.ndarray) -> np.ndarray:
    """Pairwise Levy areas A^{ij} = (S^{ij} - S^{ji})/2, upper triangle (i<j)."""
    d = S2.shape[1]
    iu = np.triu_indices(d, k=1)
    return 0.5 * (S2[:, iu[0], iu[1]] - S2[:, iu[1], iu[0]])


def leadlag_pair_areas(inc_a: np.ndarray, inc_b: np.ndarray) -> np.ndarray:
    """Levy areas of the lead-lag transform for one pair of increment series
    (T, n) each -- the explicit 'did b move before a' detector
    (Flint-Lyons-Turner lead-lag encoding via Morrill et al. arXiv:2006.00873).
    Returns (T, 6): the four depth-2 terms of the 2-channel lead-lag pair plus
    the two cross Levy areas."""
    T, n = inc_a.shape
    S1 = np.zeros((T, 2)); S2 = np.zeros((T, 2, 2))
    for k in range(n):
        da = inc_a[:, k]; db = inc_b[:, k]
        # first move: lead goes da, lag stays
        dx = np.stack([da, np.zeros_like(da)], axis=1)
        S2 += np.einsum("ti,tj->tij", S1, dx)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx)
        S1 += dx
        # second move: lag catches up db
        dx = np.stack([np.zeros_like(db), db], axis=1)
        S2 += np.einsum("ti,tj->tij", S1, dx)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx)
        S1 += dx
    A = 0.5 * (S2[:, 0, 1] - S2[:, 1, 0])
    return np.stack([S1[:, 0], S1[:, 1], S2[:, 0, 0], S2[:, 1, 1],
                     A, S2[:, 0, 1] + S2[:, 1, 0]], axis=1)


def signature_features(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
                       price_cols: dict[str, str],
                       depth3: bool = False,
                       leadlag_pairs: tuple[tuple[str, str], ...] = (("xau", "jpy"), ("xau", "xag")),
                       ) -> pd.DataFrame:
    """Full signature feature block.

    Parameters
    ----------
    price_cols : channel name -> M1 price column, e.g.
        {'xau': 'close', 'xag': 'close_xag', 'eur': 'close_eur', 'jpy': 'close_jpy'}.
    depth3 : include the 125 extra depth-3 terms (d=5).
    leadlag_pairs : symbol pairs for explicit lead-lag area features.
    """
    channels = {"time": "time", **price_cols}
    X = _hourly_path_tensor(m1, channels, h1_index)
    flat, S2 = signature_depth2(X)
    d = X.shape[2]
    chan_all = list(channels.keys())                       # ['time', 'xau', ...]
    chan_px = [c for c in chan_all if c != "time"]
    names = [f"sig1_{c}" for c in chan_px] + \
            [f"sig2_{c1}_{c2}" for c1 in chan_all for c2 in chan_all]
    vals = np.concatenate([flat[:, 1:d], flat[:, d:]], axis=1)
    blocks = [pd.DataFrame(vals, index=h1_index, columns=names)]
    # pairwise Levy areas for all price-channel pairs (drop time channel)
    chan_names = [c for c in channels if c != "time"]
    iu = np.triu_indices(d, k=1)
    keep = [(a, b) for a, b in zip(iu[0], iu[1]) if a > 0 and b > 0]
    area_cols = {}
    for a, b in keep:
        area_cols[f"levy_{chan_names[a - 1]}_{chan_names[b - 1]}"] = \
            0.5 * (S2[:, a, b] - S2[:, b, a])
    blocks.append(pd.DataFrame(area_cols, index=h1_index))

    # lead-lag pair features
    incs = {}
    for name in price_cols:
        px = m1[price_cols[name]].astype("float64")
        inc = np.log(px / px.shift(1))
        mu = inc.rolling(1440, min_periods=60).mean()
        sd = inc.rolling(1440, min_periods=60).std()
        incs[name] = ((inc - mu) / (sd + 1e-12))
    full_idx = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    for a, b in leadlag_pairs:
        A = incs[a].reindex(full_idx).fillna(0.0).values
        B = incs[b].reindex(full_idx).fillna(0.0).values
        starts = h1_index.asi8 // 60_000_000_000
        full_starts = full_idx.asi8 // 60_000_000_000
        pos = np.clip(np.searchsorted(full_starts, starts), 0, len(A) - STEPS_PER_HOUR)
        ia = np.lib.stride_tricks.sliding_window_view(A, STEPS_PER_HOUR)[pos]
        ib = np.lib.stride_tricks.sliding_window_view(B, STEPS_PER_HOUR)[pos]
        ll = leadlag_pair_areas(ia.astype(np.float64), ib.astype(np.float64))
        ll_cols = {f"ll_{a}_{b}_{nm}": ll[:, j]
                   for j, nm in enumerate(["lead", "lag", "d11", "d22", "area", "sym"])}
        blocks.append(pd.DataFrame(ll_cols, index=h1_index))

    if depth3:
        f3sig = signature_depth3(X)
        d3names = [f"sig3_{c1}_{c2}_{c3}" for c1 in chan_all for c2 in chan_all for c3 in chan_all]
        blocks.append(pd.DataFrame(f3sig[:, d + d * d: d + d * d + d ** 3],
                                   index=h1_index, columns=d3names))
    return pd.concat(blocks, axis=1)


def f3_self_test() -> None:
    """Known-geometry checks: (1) Levy area of the CCW unit right triangle
    (0,0)->(1,0)->(1,1)->(0,0) equals the enclosed area 0.5; (2) reversal
    flips the sign; (3) diagonal level-2 term of a straight line = dx^2/2;
    (4) lead-lag area sign encodes who moved first; (5) Chen depth-3 identity."""
    tri = np.array([[[1, 0], [0, 1], [-1, -1]]], dtype=np.float64)   # 3 steps
    _, S2 = signature_depth2(tri)
    A = 0.5 * (S2[0, 0, 1] - S2[0, 1, 0])
    assert abs(A - 0.5) < 1e-12, f"triangle Levy area {A} != 0.5"
    tri_rev = np.array([[[-1, -1], [0, 1], [1, 0]]], dtype=np.float64)
    _, S2r = signature_depth2(tri_rev)
    Ar = 0.5 * (S2r[0, 0, 1] - S2r[0, 1, 0])
    assert abs(Ar + 0.5) < 1e-12, f"reversal did not flip area: {Ar}"
    line = np.array([[[0.3, 0.0], [0.3, 0.0]]], dtype=np.float64)
    _, S2l = signature_depth2(line)
    assert abs(S2l[0, 0, 0] - 0.5 * 0.6 ** 2) < 1e-12, "diagonal term != dx^2/2"
    # lead-lag: a moves first, b follows one step later -> area > 0
    T, n = 1, 60
    ia = np.zeros((T, n)); ib = np.zeros((T, n))
    ia[0, 10] = 1.0; ib[0, 20] = 1.0
    ll = leadlag_pair_areas(ia, ib)
    assert ll[0, 4] > 0, "lead-lag area should be positive when a leads"
    ia2 = np.zeros((T, n)); ib2 = np.zeros((T, n))
    ia2[0, 20] = 1.0; ib2[0, 10] = 1.0
    ll2 = leadlag_pair_areas(ia2, ib2)
    assert ll2[0, 4] < 0, "lead-lag area should be negative when b leads"
    # depth-3 Chen identity spot check on a single linear segment:
    seg = np.array([[[0.5, -0.25]]], dtype=np.float64)
    f3 = signature_depth3(seg)          # flat: [S1(d=2) | S2(4) | S3(8)]
    i_s3_000 = 2 + 4 + 0                # S3[0,0,0] position in the S3 block
    assert abs(f3[0, i_s3_000] - 0.5 ** 3 / 6.0) < 1e-12, "depth-3 linear-step term wrong"
    print("F3 self-tests passed: triangle area, reversal, diagonal, lead-lag, Chen.")
