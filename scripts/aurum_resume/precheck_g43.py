#!/usr/bin/env python3
"""Pre-push validation with REAL artifacts: load the actual r=3 cache
(ro_hiddens.npy) + real fold/main heads, recompute the fold-cal Brier and
the main-head 2022 Brier, compare to training-time values. This is exactly
what gate G4.3 checks on Kaggle - proving it locally first de-risks the push."""
import importlib.util, json
import numpy as np
import torch

R = "/home/z/my-project/kaggle_work/aurum/results/p2_train_latest"
M = f"{R}/models"
SPEC = importlib.util.spec_from_file_location(
    "p3k", "/home/z/my-project/kaggle_work/aurum/dist/aurum-p3-tables.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

z = np.load(f"{R}/p2_train_data.npz")
lab_years = z["lab_years"]
N = len(lab_years)
years = np.asarray(lab_years)
H = np.load(f"{M}/ro_hiddens.npy", mmap_mode="r")
print(f"cache: {H.shape} {H.dtype}; states={N}")
assert H.shape[0] == N

fold_ref = json.load(open(f"{M}/fold_results.json"))
sb = json.load(open(f"{M}/stageB_result.json"))
sm_h1 = z["sm_h1"]
CAL_FRAC = m.CAL_FRAC

st_main = m._head_stack(1024)
st_main.load_state_dict(torch.load(f"{M}/heads_main.pt", map_location="cpu",
                                   weights_only=True))
st_main.eval()

def brier_on(stack, rows, T_sm):
    X = torch.from_numpy(np.ascontiguousarray(H[rows]).astype(np.float32))
    with torch.no_grad():
        P = torch.sigmoid(stack(X)["h1"]).numpy()
    return float(np.nanmean((P - np.nan_to_num(T_sm, nan=0.5)) ** 2))

# fold heads
ok = 0
for yv in range(2016, 2026):
    st = m._head_stack(1024)
    st.load_state_dict(torch.load(f"{M}/heads_fold{yv}.pt", map_location="cpu",
                                  weights_only=True))
    st.eval()
    ftp = np.flatnonzero(years <= yv - 1)
    n_cal = max(200, int(len(ftp) * CAL_FRAC))
    cal = ftp[-n_cal:]
    b = brier_on(st, cal, sm_h1[cal])
    ref = fold_ref[str(yv)]["brier_sm"]
    match = abs(b - ref) < 0.01
    ok += match
    print(f"fold {yv}: recomputed brier_sm={b:.5f} ref={ref:.5f} "
          f"delta={b-ref:+.5f} {'OK' if match else 'FAIL'}")
# main on 2022
v22 = np.flatnonzero(years == 2022)
b22 = brier_on(st_main, v22, sm_h1[v22])
ref22 = sb["final_val"]["brier_sm"]
print(f"main 2022: recomputed={b22:.5f} ref={ref22:.5f} "
      f"delta={b22-ref22:+.5f} {'OK' if abs(b22-ref22) < 0.012 else 'FAIL'}")
print(f"\nG4.3 pre-check: {ok}/10 folds match; main "
      f"{'OK' if abs(b22 - ref22) < 0.012 else 'FAIL'}")
assert ok >= 8 and abs(b22 - ref22) < 0.012, "G4.3 would FAIL on Kaggle"
print("G4.3 PRE-CHECK PASSED - safe to push")
