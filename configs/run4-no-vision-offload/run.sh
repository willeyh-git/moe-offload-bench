#!/bin/bash
# Run 4 — pinned experts false, chunk 4096 (baseline, no vision offload)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export EXL3_QC_STAGING=${EXL3_QC_STAGING:-1}
export EXL3_GDN_REPLAY=${EXL3_GDN_REPLAY:-1}
exec python "$ROOT/tabbyAPI/main.py" --max-seq-len 130816 --cache-size 131072