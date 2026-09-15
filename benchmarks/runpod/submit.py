"""Submit one training job to the serverless endpoint and follow it to the end.
  RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... python benchmarks/runpod/submit.py reader-v5-gemma4-e4b [--max-length 6912] [--no-liger] [--timeout-hours 5]
Sets the execution timeout to three hours (the endpoint default of 600 s would kill the job)."""

from __future__ import annotations

import argparse
import json
import os
import time

import runpod

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--max-length", type=int, default=8192)
ap.add_argument("--no-liger", action="store_true"); ap.add_argument("--quant", default="Q8_0")
ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--timeout-hours", type=float, default=3.0)
a = ap.parse_args()
runpod.api_key = os.environ["RUNPOD_API_KEY"]
endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])
job = endpoint.run({"input": {"run": a.run, "max_length": a.max_length, "liger": not a.no_liger, "quant": a.quant,
                              "batch_size": a.batch_size, "grad_accum": a.grad_accum},
                    "policy": {"executionTimeout": int(a.timeout_hours * 60 * 60 * 1000), "ttl": 24 * 60 * 60 * 1000}})
print("job", job.job_id)
last = None
started = time.time()
while True:
    # One fetch a poll: _fetch_job carries the status and the in-flight progress string together.
    detail = job._fetch_job() if hasattr(job, "_fetch_job") else {}
    status = detail.get("status") or job.status()
    line = f"{status} {detail.get('output') if isinstance(detail.get('output'), str) else ''}".strip()
    if line != last:
        print(f"[{(time.time() - started) / 60:5.1f} min] {line}"); last = line
    if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    time.sleep(30)
print(json.dumps(job.output(), indent=2) if status == "COMPLETED" else f"job ended {status}")
