# Community benchmark on AMD Radeon

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 0 GiB VRAM
- CPU: AMD Ryzen AI 9 HX 470 w/ Radeon 890M (12 cores, 24 threads); installed RAM: 93.4 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit a24a78e0; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --resident-budget-gib 55
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1150-hipblaslt-100401.txt`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 16.3 (13.8-22.2) | 1.40 (1.26-1.55) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 234.1 (234.1-235.4) | not measured | 17.27 (17.17-17.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 237.3 (236.0-237.9) | not measured | 67.78 (67.61-68.17) |
| agent session, follow-up turns | 14493 (14472-14507) at the last turn | see below | up to 64 | 3 | see below | 17.5 (14.3-22.1) | 15.38 (9.21-17.53) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.8); earlier-turn tokens read again per session: 144 (4-144) (earlier turns that the cache did not hold; the 13635 (13635-13636) new tokens of the files are read once); first turn (nothing cached) first token: 5.05 s (5.04-5.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 122.1 | 5.084 | 16.4 |
| 1 | 3705 | 623 | 3082 | 200.8 | 15.46 | 20.9 |
| 2 | 5238 | 3768 | 1470 | 160.9 | 9.24 | 17.4 |
| 3 | 8682 | 5301 | 3381 | 197.0 | 17.274 | 14.3 |
| 4 | 11599 | 8739 | 2860 | 190.0 | 15.161 | 20.4 |
| 5 | 14472 | 11626 | 2846 | 189.5 | 15.138 | 18.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 627 s.
