import os, sys
H = os.path.dirname(os.path.abspath(__file__))
for p in (H, os.path.join(H, "aurum_features"), os.path.join(H, "model")):
    if p not in sys.path: sys.path.insert(0, p)
import f1_clock, f2_realized, f3_signatures, f4_toxicity, f5_residual, f6_trend
f1_clock.f1_self_test(); f2_realized.f2_self_test(); f3_signatures.f3_self_test()
f4_toxicity.f4_self_test(); f5_residual.f5_self_test(); f6_trend.f6_self_test()
import losses, continuum, ae_mlp_dmn, purge, online_engine, rff_ridge, v2_common, train_continuum
losses.self_test(); continuum.self_test(); ae_mlp_dmn.self_test(); purge.self_test()
online_engine.self_test(); rff_ridge.self_test(); v2_common.self_test()
train_continuum.self_test()
print("RESTORED SOURCES: all module tests pass")
