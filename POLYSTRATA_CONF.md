# Polystrata Config

{
  "exe": "/PolyStrata/engine-poly/strata-poly",
  "args": [
    "--model",
    "/PolyStrata-data/models/ornith-APEX-I-QUALITY/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf",
    "--max-context",
    "131072",
    "--expert-profile",
    "/PolyStrata-data/packs/ornith-apex-i-quality/expert-profile.txt",
    "--kv-whole"
  ],
  "cwd": "/PolyStrata",
  "tokenizer": "/PolyStrata-data/packs/ornith-apex-i-quality/tokenizer",
  "model_name": "ornith-1.5-35b-a3b-apex-i-quality",
  "display_name": "Ornith 1.5 35B-A3B APEX-I-QUALITY",
  "log": "/PolyStrata/strata-ornith-apex-i-quality.log",
  "lib_dirs": [
    "/opt/rocm/lib"
  ],
  "env": {
    "STRATA_RESIDENT_PIN": "0",
    "STRATA_ROUTE_TAIL_SKIP": "0"
  },
  "host": "0.0.0.0",
  "port": 8081,
  "fit_max_tokens": true,
  "sampling": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.00
  }
}

