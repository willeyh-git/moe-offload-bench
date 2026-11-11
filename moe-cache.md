# LLama.cpp new moe-cache-cpu-share flag

## Results from
`python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768`

- 1/6 2048 MiB, share 0: 35.3 t/s
- 2/6 2048 MiB, share 0.5: 39.4 t/s
- 3/6 2048 MiB, share 0.95: 47.3 t/s
- 4/6 5888 MiB, share 0: 54.2 t/s
- 5/6 5888 MiB, share 0.5: 52.0 t/s
- 6/6 5888 MiB, share 0.95: 54.7 t/s

https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md


Initial after turning on --moe-cache-cpu-share 0.95 | --moe-cache-mib at 4096 | --n-cpue-moe 25
No immediate gain.

✓ warm-up                    
✓ decode greedy (5 runs)     median 62.3 tok/s
✓ 4K prefill + decode (5 runs) prefill 966.4 tok/s, TTFT 4.28s · decode median 61.2 tok/s
✓ 16K prefill + decode (5 runs) prefill 1027.3 tok/s, TTFT 15.86s · decode median 56.4 tok/s

Saw no point in continuing the benchmark
