#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import bundle, V2
bundle(
    title="AURUM v2 Neural Completion",
    kernel_id="tskdev/aurum-v2-neural-complete",
    modules=[os.path.join(V2, "model", m) for m in
             ["losses.py", "continuum.py", "ae_mlp_dmn.py", "purge.py",
              "train_continuum.py", "online_engine.py", "v2_common.py"]],
    main_path=os.path.join(V2, "kernels", "k3b_completion.py"),
    out_dir="/home/z/my-project/kaggle_work/v2/k3b",
    dataset_sources=["tskdev/aurum-v2-featstore", "tskdev/aurum-phase1-data",
                     "tskdev/aurum-v2-forecasts-neural"],
    enable_gpu=True,
)
