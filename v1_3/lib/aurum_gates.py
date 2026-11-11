# ============================================================================
# AURUM shared helpers: gates, logging, env report
# Spec v1.1 (frozen). All quantity-producing code paths must be dimensionless
# (ATR units / log-returns / z-scores / ratios / codebook ids) - see audit #1.
# ============================================================================
import json, os, sys, time, traceback, random
import numpy as np

GATES = {}
LOG_LINES = []

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

def gate(gid, ok, detail=None, extra=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2000]}
    if extra:
        g.update(extra)
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return ok

def save_gates(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(os.path.dirname(path), "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))

def set_seeds(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def env_report():
    import platform
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["capability"] = str(torch.cuda.get_device_capability(0))
            p = torch.cuda.get_device_properties(0)
            info["total_mem_gb"] = round(p.total_memory / 1e9, 2)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "missing"
    return info

def is_kaggle():
    return os.path.exists("/kaggle/working")

def stage(name):
    """Decorator-ish context: log stage start/end, capture exceptions."""
    class C:
        def __init__(self):
            self.t0 = time.time()
            self.ok = None
            self.err = None
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            self.ok = et is None
            if et is not None:
                self.err = "".join(traceback.format_exception(et, ev, tb))[-3000:]
                log(f"===== STAGE {name} ERROR =====\n{self.err}")
            else:
                log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False  # do not suppress
    return C()

# Frozen architecture constants (Spec v1.1) ---------------------------------
BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]      # x ATR14(H1)
BARRIER_TPS   = [1.5, 2.5, 4.0]           # x ATR14(H1)
BARRIER_HORS  = [24, 48]                  # H1 bars; default vertical barrier H=48
N_CELLS       = 2 * 4 * 3 * 2             # = 48 cells (dir x stop x tp x horizon)
KELLY_FRACTION = 0.25                     # quarter Kelly
HARD_EQUITY_RISK_CAP = 0.01               # 1.0% equity risk per trade
COHORT_K = 200                            # B=200 bootstrap cohorts
COHORT_GAP = 96                           # neighbour must end >=96 bars before target bar
READOUT_NAMES = ["H1_BARRIER", "H2_MTM", "H3_EXCUR", "H4_VOL", "H5_REGIME"]
HEAD_DIMS = {"h1": 48, "h2": 8, "h3": 12, "h4": 8, "h5": 9}  # 85 outputs total

def cell_axes():
    """Flat 48-cell layout: for d in (long,short) for m in stops for t in tps for h in hors."""
    dirs, mss, tps, hrs = [], [], [], []
    for d in ("L", "S"):
        for m in BARRIER_STOPS:
            for t in BARRIER_TPS:
                for h in BARRIER_HORS:
                    dirs.append(d); mss.append(m); tps.append(t); hrs.append(h)
    return dirs, mss, tps, hrs

def cell_names():
    dirs, mss, tps, hrs = cell_axes()
    return [f"{d}_m{m}_t{t}_h{h}" for d, m, t, h in zip(dirs, mss, tps, hrs)]
