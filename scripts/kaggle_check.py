import os
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest, ApiListKernelsRequest, ApiGetKernelSessionStatusRequest)

tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
client = KaggleClient(env=KaggleEnv.PROD, api_token=tok)
api = client.kernels.kernels_api_client

print("=== QUOTA ===")
try:
    r = ApiGetAcceleratorQuotaStatisticsRequest()
    resp = api.get_accelerator_quota_statistics(r)
    print(resp.to_dict() if hasattr(resp,'to_dict') else resp)
except Exception as e:
    import traceback; traceback.print_exc()

print("\n=== KERNELS (mine, sort by dateRun) ===")
try:
    req = ApiListKernelsRequest()
    req.user = "tskdev"
    req.page_size = 100
    # try sort_by variants
    try:
        req.sort_by = "dateRun"
    except Exception:
        pass
    resp = api.list_kernels(req)
    d = resp.to_dict() if hasattr(resp, 'to_dict') else {}
    import json
    items = d.get("kernels", d) if isinstance(d, dict) else d
    print(json.dumps(d, indent=1, default=str)[:6000])
except Exception as e:
    import traceback; traceback.print_exc()
