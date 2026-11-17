#!/bin/bash
# Run 2 — pinned experts false, chunk 4096, vision offload, cpu_stage_threads 6, threads 7
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export EXL3_QC_STAGING=${EXL3_QC_STAGING:-1}
export EXL3_GDN_REPLAY=${EXL3_GDN_REPLAY:-1}
exec python "$ROOT/tabbyAPI/main.py" --max-seq-len 130816 --cache-size 131072