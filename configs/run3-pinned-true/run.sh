#!/bin/bash
# Run 3 — pinned experts true, chunk 4096, vision offload
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export EXL3_QC_STAGING=${EXL3_QC_STAGING:-1}
export EXL3_GDN_REPLAY=${EXL3_GDN_REPLAY:-1}
exec python "$ROOT/tabbyAPI/main.py" --max-seq-len 130816 --cache-size 131072