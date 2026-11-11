#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import bundle, V2
bundle(
    title="AURUM v2 CPU Models RG",
    kernel_id="ranigain/aurum-v2-cpu-models-rg",
    modules=[os.path.join(V2, "model", m) for m in
             ["online_engine.py", "rff_ridge.py", "v2_common.py"]],
    main_path=os.path.join(V2, "kernels", "k2_cpu_models.py"),
    out_dir="/home/z/my-project/kaggle_work/v2/k2rg",
    dataset_sources=["ranigain/aurum-v2-featstore"],
    enable_gpu=False,
)
