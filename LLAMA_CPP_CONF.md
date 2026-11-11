# Llama.cpp config for runs

## Always
```
```
exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias "ornith-llama" \
  --load-mode none \
  --host 0.0.0.0 \
  --port 8096 \
  --ctx-size 131072 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --moe-cache-mib 4096 \
  -fa on \
  --parallel 1 \
  --threads 6 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0

## No MTP
--n-cpu-moe 25

## MTP
--n-cpu-moe 30 \
--spec-type draft-mtp,ngram-mod \
--spec-ngram-mod-n-match 40 \
--spec-ngram-mod-n-min 0 \
--spec-ngram-mod-n-max 16 \
--spec-draft-n-max 3 \
--spec-draft-type-k q4_0 \
--spec-draft-type-v q4_0 \
