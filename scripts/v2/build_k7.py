#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import bundle, V2
bundle(
    title="AURUM v2 Live Daily",
    kernel_id="tskdev/aurum-v2-live-daily",
    modules=[os.path.join(V2, "model", m) for m in ["online_engine.py", "v2_common.py"]],
    main_path=os.path.join(V2, "kernels", "k7_live_daily.py"),
    out_dir="/home/z/my-project/kaggle_work/v2/k7",
    dataset_sources=["tskdev/aurum-v2-featstore", "tskdev/aurum-v2-cpu-preds",
                     "tskdev/aurum-v13-artifacts", "tskdev/aurum-v2-backtest-outputs"],
    enable_gpu=False,
)
