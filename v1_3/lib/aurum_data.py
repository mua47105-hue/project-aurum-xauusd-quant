# ============================================================================
# AURUM data ingestion: HistData M1 (primary), Dukascopy gap-fill (fallback),
# FRED daily macro (DTWEXBGS/DGS10/DGS2/T10Y2Y/VIXCLS). Free sources only.
# ============================================================================
import io, os, re, time, zipfile, requests
import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/zip,*/*"}

FRED_SERIES = {"dxy": "DTWEXBGS", "y10": "DGS10", "y2": "DGS2", "spread": "T10Y2Y", "vix": "VIXCLS"}

# ---------------------------------------------------------------------------
# HistData: token-flow downloader for yearly ASCII M1 archives
# ---------------------------------------------------------------------------

def histdata_year_url(pair, year):
    return f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"

def histdata_download_year(pair, year, out_dir, session=None, tries=4, sleep_s=3.0, log=print):
    """Downloads the full-year ASCII M1 zip for pair (e.g. 'xauusd'). Returns
    path to zip or None. Parses the actual `file_down` form fields (tk token
    + platform/timeframe/fxpair controls) and POSTs them to /get.php."""
    path = os.path.join(out_dir, f"{pair.upper()}_M1_{year}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    s = session or requests.Session()
    s.headers.update(UA)
    url = histdata_year_url(pair, year)
    for attempt in range(tries):
        try:
            r = s.get(url, timeout=60)
            if r.status_code != 200:
                log(f"  histdata {pair} {year}: GET {r.status_code}")
                time.sleep(sleep_s * (attempt + 1)); continue
            html = r.text
            form = re.search(r'<form id="file_down".*?</form>', html, re.S)
            if not form:
                log(f"  histdata {pair} {year}: no file_down form (len {len(html)})")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload = {}
            for tag in re.findall(r"<input[^>]*>", form.group(0)):
                nm = re.search(r'name="([^"]*)"', tag)
                vl = re.search(r'value="([^"]*)"', tag)
                if nm:
                    payload[nm.group(1)] = vl.group(1) if vl else ""
            if "tk" not in payload or not payload["tk"]:
                log(f"  histdata {pair} {year}: no tk token in form")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload["datemonth"] = str(year)   # full year
            pr = s.post("https://www.histdata.com/get.php", data=payload,
                        headers={"Referer": url}, timeout=600)
            ct = pr.headers.get("content-type", "")
            if pr.status_code == 200 and (pr.content[:2] == b"PK") and len(pr.content) > 10_000:
                os.makedirs(out_dir, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(pr.content)
                log(f"  histdata {pair} {year}: OK {len(pr.content)/1e6:.1f} MB")
                return path
            log(f"  histdata {pair} {year}: POST st={pr.status_code} ct={ct} len={len(pr.content)}")
        except Exception as e:
            log(f"  histdata {pair} {year}: exc {type(e).__name__}: {str(e)[:120]}")
        time.sleep(sleep_s * (attempt + 1))
    return None

def parse_histdata_zip(zip_path, log=print):
    """Returns M1 DataFrame (UTC tz-aware index, o/h/l/c/v) or None.
    Handles both flat csv-in-zip and nested monthly zips."""
    frames = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for nm in names:
                if nm.lower().endswith(".csv"):
                    frames.append(_read_hist_csv(z.read(nm)))
                elif nm.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(z.read(nm))) as z2:
                        for nm2 in z2.namelist():
                            if nm2.lower().endswith(".csv"):
                                frames.append(_read_hist_csv(z2.read(nm2)))
    except Exception as e:
        log(f"  parse {os.path.basename(zip_path)}: {type(e).__name__} {str(e)[:150]}")
        return None
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _read_hist_csv(raw):
    txt = raw.decode("latin-1").replace("\x00", "")
    # header: GMT time;Open;High;Low;Close;Volume, rows "20240101 180000;2062.598;..."
    df = pd.read_csv(io.StringIO(txt), sep=";", header=0, dtype=str,
                     usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ["ts", "o", "h", "l", "c", "v"]
    for col in ("o", "h", "l", "c"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S", utc=True)
    df.index = ts
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[(df["o"] > 0) & (df["h"] >= df[["o", "c"]].min(axis=1) - 1e-9)
            & (df["l"] <= df[["o", "c"]].max(axis=1) + 1e-9)]
    return df[["o", "h", "l", "c", "v"]]

# ---------------------------------------------------------------------------
# Dukascopy fallback (gap filler): per-day BID M1 bi5 (LZMA) files
# ---------------------------------------------------------------------------

def dukascopy_day_url(pair, y, m, d):
    return f"https://datafeed.dukascopy.com/datafeed/{pair.upper()}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"

def parse_bi5(raw):
    try:
        import lzma
        data = lzma.decompress(raw)
    except Exception:
        return None
    n = len(data) // 24
    if n == 0:
        return None
    arr = np.frombuffer(data[:n * 24], dtype=">i4").reshape(n, 6).astype(np.float64)
    # cols: sec-offset(ms?), open, close, low, high, volume  (point-scaled)
    # Dukascopy M1: [time_ms_offset, open, close, low, high, volume], prices *1e5 (fx) or *1e3 (metals)
    return arr

# ---------------------------------------------------------------------------
# FRED (fredgraph.csv endpoints - no API key needed)
# ---------------------------------------------------------------------------

def fred_fetch(series_code, log=print):
    """FRED daily series via fredgraph.csv (no API key). requests first,
    curl-subprocess fallback (some networks throttle python-requests UAs)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    txt = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200 and "observation_date" in r.text[:200]:
                txt = r.text
                break
            log(f"  FRED {series_code}: st={r.status_code}")
        except Exception as e:
            log(f"  FRED {series_code}: exc {type(e).__name__}")
        time.sleep(4 * (attempt + 1))
    if txt is None:
        try:
            import subprocess
            p = subprocess.run(["curl", "-sS", "--max-time", "90", url],
                               capture_output=True, text=True, timeout=120)
            if p.returncode == 0 and "observation_date" in p.stdout[:200]:
                txt = p.stdout
                log(f"  FRED {series_code}: OK via curl fallback")
        except Exception as e:
            log(f"  FRED {series_code}: curl exc {type(e).__name__}")
    if txt is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.set_index("date")["value"].sort_index()
        log(f"  FRED {series_code}: OK {len(s)} rows ({s.index.min().date()}..{s.index.max().date()})")
        return s
    except Exception as e:
        log(f"  FRED {series_code}: parse exc {type(e).__name__}")
        return None

# ---------------------------------------------------------------------------
# UTC-offset estimation for HistData clocks (their ASCII export has run in a
# server-local clock historically). We correlate the hourly volatility profile
# against a canonical forex session template and pick the best integer shift.
# ---------------------------------------------------------------------------

def estimate_utc_offset(m1, log=print, candidates=(-3, -2, -1, 0, 1, 2, 3)):
    """m1: M1 df with tz-UTC index (as-parsed clock). Returns (offset_hours, corr)."""
    r = np.log(m1["c"] / m1["c"].shift(1)).abs()
    prof = r.groupby(m1.index.hour).mean().reindex(range(24), fill_value=0.0).values
    prof = (prof - prof.mean()) / (prof.std() + 1e-12)
    # canonical template: London open 07-08 UTC, NY open 12-13 UTC, overlap 15-16
    tpl = np.zeros(24)
    tpl[[7, 8, 12, 13, 15, 16]] = 1.6
    tpl[[3, 4, 5, 6, 9, 10, 17, 18]] = 0.8
    tpl[[21, 22, 23, 0, 1, 2]] = 0.4
    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
    best_off, best_c = 0, -2
    for off in candidates:
        c = np.corrcoef(np.roll(prof, off), tpl)[0, 1]
        if c > best_c:
            best_c, best_off = c, off
    c0 = np.corrcoef(prof, tpl)[0, 1]
    if best_c < c0 * 1.2 or best_c < 0.35:
        best_off, best_c = 0, c0   # ambiguous -> keep as-is
    log(f"  utc-offset est: offset={best_off:+d}h corr={best_c:.2f} (null corr={c0:.2f})")
    return best_off, float(best_c)

def apply_utc_offset(df, offset_hours):
    if offset_hours == 0:
        return df
    return df.set_index(df.index + pd.Timedelta(hours=offset_hours))

# ---------------------------------------------------------------------------
# H1 aggregation + coverage report
# ---------------------------------------------------------------------------

def to_h1(m1):
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                               c=("c", "last"), v=("v", "sum"))
    return h1.dropna(subset=["o", "c"])

def coverage_report(m1, years=range(2010, 2026)):
    rep = {}
    exp = 365.25 * 24 * 60 * (5 / 7)  # approx tradable minutes per year
    for y in years:
        sub = m1[(m1.index.year == y)]
        got = len(sub)
        rep[str(y)] = {"rows": int(got), "coverage": round(got / exp, 3) if exp else None}
    return rep
