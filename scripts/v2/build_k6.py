#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import bundle, V2
bundle(
    title="AURUM v2 Backtest",
    kernel_id="tskdev/aurum-v2-backtest",
    modules=[os.path.join(V2, "model", m) for m in ["online_engine.py", "v2_common.py"]],
    main_path=os.path.join(V2, "kernels", "k6_backtest.py"),
    out_dir="/home/z/my-project/kaggle_work/v2/k6",
    dataset_sources=["tskdev/aurum-v2-featstore", "tskdev/aurum-v2-cpu-preds",
                     "tskdev/aurum-v2-neural-preds", "tskdev/aurum-v13-artifacts",
                     "tskdev/aurum-phase1-data"],
    enable_gpu=False,
)
