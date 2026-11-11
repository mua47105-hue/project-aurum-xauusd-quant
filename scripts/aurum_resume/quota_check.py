#!/usr/bin/env python3
"""Live GPU quota check via kagglesdk (working method discovered this session)."""
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest,
)

tok = open("/home/z/.kaggle/access_token").read().strip()
client = KaggleClient(env=KaggleEnv.PROD, verbose=False,
                      username=None, password=None, api_token=tok)
resp = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
    ApiGetAcceleratorQuotaStatisticsRequest())
gq = resp.gpu_quota
used = gq.time_used.total_seconds()
res = gq.time_reserved.total_seconds()
tot = gq.total_time_allowed.total_seconds()
print(f"GPU quota: used {used/3600:.2f}h + reserved {res/3600:.2f}h "
      f"of {tot/3600:.0f}h -> remaining {(tot-used-res)/3600:.2f}h")
print(f"reset: {resp.quota_refresh_time}")
