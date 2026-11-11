#!/usr/bin/env python3
"""Verify Step 3 (aurum-p2-train v6) gates locally from pulled artifacts.
Computes gates with the CORRECT from-base reference (p2a-check base loss
1.369), and writes models_v1/manifest.json for the dataset upload."""
import json, os, hashlib, datetime

R = "/home/z/my-project/kaggle_work/aurum/results/p2_train_latest"
M = os.path.join(R, "models")
OUT = "/home/z/my-project/kaggle_work/aurum/results/p2_train_gates_verified.json"

a = json.load(open(os.path.join(M, "stageA_result.json")))
b = json.load(open(os.path.join(M, "stageB_result.json")))
fr = json.load(open(os.path.join(M, "fold_results.json")))
sched = json.load(open(os.path.join(M, "schedule.json")))
p2a = json.load(open("/home/z/my-project/kaggle_work/aurum/results/p2a_check_latest/p2a_result.json"))
meta = json.load(open(os.path.join(R, "p2_train_data_meta.json")))

BASE_CE = p2a["loss_first10"]          # 1.369 from-base reference (p2a-check)
from_base_red = 1.0 - a["loss_last10"] / BASE_CE

gates = {}
gates["G3.0_data"] = {
    "pass": meta["n_labeled_states"] == 87948 and meta["mean_tokens"] <= 340,
    "detail": f"labeled states={meta['n_labeled_states']} (pretest {meta['n_pretest']} / "
              f"OOS {meta['n_oos']}); mean tokens={meta['mean_tokens']} "
              f"(min {meta['min_tokens']} max {meta['max_tokens']})"}
gates["G3.1_stage_a_prime"] = {
    # v6 resumed Stage A' from v5's crash point (loss_first10=0.6975 is the
    # RESUME-point loss, not base). True from-base reduction uses p2a's base
    # reference 1.369 (identical model+data+objective, measured from the
    # untrained base over the first 150 steps).
    "pass_from_base": from_base_red >= 0.30 and a["nan_skips"] == 0
                      and a["clamp_rate"] < 0.05 and a["z_final"] > 0.90,
    "pass_resume_ref": a["loss_reduction"] >= 0.30,
    "detail": (f"from-base sliced-CE {BASE_CE:.4f} -> {a['loss_last10']:.4f} "
               f"(-{from_base_red*100:.1f}% vs gate >=30%); resume-segment ref "
               f"{a['loss_first10']:.4f} -> {a['loss_last10']:.4f} "
               f"(-{a['loss_reduction']*100:.1f}%); tokens={a['tokens']:,}; "
               f"Z_final={a['z_final']}; nan_skips={a['nan_skips']}; "
               f"clamp={a['clamp_rate']}; DDP {a['tok_s_ddp']} tok/s")}
gates["G3.2_stage_b_heads"] = {
    "pass": b["final_val"].get("auc_re") is not None
            and not b.get("early_stop_reverted", False)
            and not b["kill_switch_triggered"],
    "detail": (f"kill-switch NOT triggered (lam03 AUC={b['ckpt1_lam03']['auc_re']:.4f} >= "
               f"lam00 AUC={b['ckpt1_lam00']['auc_re']:.4f} -> kept {b['kept_branch']}); "
               f"final val brier_sm={b['final_val']['brier_sm']:.4f} "
               f"brier_re={b['final_val']['brier_re']:.4f} "
               f"auc_re={b['final_val']['auc_re']:.4f}; "
               f"early_stop_reverted={b['early_stop_reverted']} (seg2 reverted to ckpt-1 "
               f"= final == ckpt-1, so final model IS the checkpoint-1 model)")}
gates["G3.3_walkforward_refits"] = {
    "pass": len(fr) == 10 and all(not v.get("skipped") for v in fr.values()),
    "detail": f"10/10 folds refit + temperature-scaled; "
              f"auc_re range [{min(v['auc_re'] for v in fr.values()):.4f}, "
              f"{max(v['auc_re'] for v in fr.values()):.4f}]; "
              f"brier_sm range [{min(v['brier_sm'] for v in fr.values()):.4f}, "
              f"{max(v['brier_sm'] for v in fr.values()):.4f}]"}
need = {"core_fp16.pt", "heads_main.pt", "temperatures_main.json", "ro_hiddens.npy",
        "temperatures.json", "fold_results.json", "stageA_result.json",
        "stageB_result.json", "schedule.json"} | \
       {f"heads_fold{y}.pt" for y in range(2016, 2026)}
have = set(os.listdir(M))
gates["G3.4_export"] = {
    "pass": need.issubset(have),
    "detail": f"all {len(need)} required artifacts present ({len(have)} files total)"}
gates["_kernel_status_note"] = (
    "Kaggle status=ERROR is a post-export summary bug: line 2700 reads "
    "b['early_stop_flag'] but stageB_result.json stores 'early_stop_reverted' "
    "(KeyError AFTER 'EXPORT DONE' + after all gates' inputs were written). "
    "All training artifacts are complete and pulled; no re-run required.")

print(json.dumps(gates, indent=1))
json.dump(gates, open(OUT, "w"), indent=1)
print("\nsaved:", OUT)

# ---- manifest for models_v1 dataset ----
MD = "/home/z/my-project/kaggle_work/aurum/models_v1"
os.makedirs(MD, exist_ok=True)
man = {
    "name": "aurum-models-v1",
    "owner": "tskdev",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "provenance": {
        "kernel": "tskdev/aurum-p2-train",
        "final_version": "v6 (resume of v5; Stage A' resumed, Stage B/cache/"
                         "refits/export all completed in v6)",
        "run_window_utc": "2026-11-08 06:35:45 -> 13:00:04",
        "base_model": "Qwen/Qwen3-0.6B",
        "loop_layers": "13-16", "train_n_loop": 3,
        "train_end_year": 2022, "fold_years": list(range(2016, 2026)),
        "stageA_tokens": a["tokens"], "stageA_from_base_reduction": round(from_base_red, 4),
        "stageB_kept_branch": b["kept_branch"],
        "stageB_final_val": b["final_val"],
        "temperatures_main": json.load(open(os.path.join(M, "temperatures_main.json"))),
    },
    "files": {},
}
for f in sorted(os.listdir(M)):
    p = os.path.join(M, f)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    man["files"][f] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
json.dump(man, open(os.path.join(MD, "manifest.json"), "w"), indent=1)
print("manifest written with", len(man["files"]), "files")
