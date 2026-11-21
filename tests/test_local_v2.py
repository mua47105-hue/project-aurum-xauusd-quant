import os, sys
# Repro gap #1 fix (Spec v2.1): resolve module dirs from the REPO ROOT, not tests/.
# The module packages live at aurum_v2/aurum_features and aurum_v2/model; the
# pre-v2.1 version pointed at tests/aurum_features and tests/model, which do not
# exist, and required an external PYTHONPATH=aurum_v2/aurum_features:aurum_v2/model.
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(R, "aurum_v2", "aurum_features"),
          os.path.join(R, "aurum_v2", "model")):
    if p not in sys.path: sys.path.insert(0, p)
import f1_clock, f2_realized, f3_signatures, f4_toxicity, f5_residual, f6_trend
f1_clock.f1_self_test(); f2_realized.f2_self_test(); f3_signatures.f3_self_test()
f4_toxicity.f4_self_test(); f5_residual.f5_self_test(); f6_trend.f6_self_test()
import losses, continuum, ae_mlp_dmn, purge, online_engine, rff_ridge, v2_common, train_continuum
losses.self_test(); continuum.self_test(); ae_mlp_dmn.self_test(); purge.self_test()
online_engine.self_test(); rff_ridge.self_test(); v2_common.self_test()
train_continuum.self_test()
print("RESTORED SOURCES: all module tests pass")
