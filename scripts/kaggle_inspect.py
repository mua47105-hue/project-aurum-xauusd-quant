#!/usr/bin/env python3
"""Inspect Kaggle state: kernel versions, dataset file lists, GPU quota."""
import os, sys, json
from kagglesdk import KaggleClient, KaggleEnv
TOK = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
client = KaggleClient(env=KaggleEnv.PROD, api_token=TOK)

def kernel_versions(owner, slug):
    # Use kernel_metadata CLI equivalent: list kernels then filter — not available;
    # instead use kernels_api_client kernel pull of metadata per version is complex.
    # Simplest: scrape from kaggle CLI JSON output.
    import subprocess
    out = subprocess.run(["kaggle", "kernels", "status", f"{owner}/{slug}"], capture_output=True, text=True)
    return out.stdout.strip()

def dataset_files(owner, slug):
    from kagglesdk.datasets.types.datasets_api_service import ApiListDatasetFilesRequest
    req = ApiListDatasetFilesRequest()
    req.owner_slug = owner
    req.dataset_slug = slug
    resp = client.datasets.list_dataset_files(req)
    return [{"name": f.name, "bytes": f.total_bytes} for f in resp.files]

def quota():
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    req = ApiGetAcceleratorQuotaStatisticsRequest()
    resp = client.kernels.get_accelerator_quota_statistics(req)
    pb = resp._pb
    d = {}
    for f in pb.DESCRIPTOR.fields:
        d[f.name] = str(getattr(pb, f.name))
    return d

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "versions":
        print(json.dumps(kernel_versions("tskdev", sys.argv[2]), indent=1))
    elif what == "files":
        print(json.dumps(dataset_files("tskdev", sys.argv[2]), indent=1))
    elif what == "quota":
        print(json.dumps(quota(), indent=1))
