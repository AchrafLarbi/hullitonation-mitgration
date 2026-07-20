import sys, time
sys.stdout.reconfigure(encoding='utf-8')
from huggingface_hub import HfApi
api = HfApi()
sid = 'achraf2203/rare-disease-diagnostic'
for i in range(20):
    info = api.space_info(sid)
    st = info.runtime.stage if info.runtime else 'unknown'
    print(f"[{i*20}s] Status: {st}", flush=True)
    if st == "RUNNING":
        print("Space is RUNNING!")
        break
    if st == "BUILD_ERROR" or st == "RUNTIME_ERROR":
        print(f"ERROR: {st}")
        break
    time.sleep(20)
