# Community benchmark on NVIDIA GeForce RTX 2070 SUPER

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2070 SUPER, 8 GiB VRAM
- CPU: AMD Ryzen AI 9 HX 470 w/ Radeon 890M (12 cores, 24 threads); installed RAM: 93.4 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit aad118a4; engine 0.1.43
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 14.2 (11.9-22.0) | 1.42 (1.26-1.78) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 266.1 (265.1-269.8) | not measured | 15.20 (15.00-15.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 273.4 (168.3-287.0) | not measured | 58.86 (56.10-95.56) |
| agent session, follow-up turns | 14491 (14485-14514) at the last turn | see below | up to 64 | 3 | see below | 16.5 (14.1-21.0) | 13.18 (7.96-14.57) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.9); earlier-turn tokens read again per session: 2 (2-4) (earlier turns that the cache did not hold; the 13634 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 5.06 s (5.04-5.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 124.2 | 5.064 | 17.0 |
| 1 | 3716 | 634 | 3082 | 229.9 | 13.568 | 20.9 |
| 2 | 5249 | 3779 | 1470 | 184.2 | 8.112 | 16.1 |
| 3 | 8693 | 5313 | 3380 | 235.0 | 14.52 | 14.1 |
| 4 | 11617 | 8757 | 2860 | 198.1 | 14.569 | 19.7 |
| 5 | 14485 | 11639 | 2846 | 222.3 | 12.926 | 14.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 622 s.
