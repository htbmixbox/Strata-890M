# Community benchmark on NVIDIA GeForce RTX 2070 SUPER

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2070 SUPER, 8 GiB VRAM
- CPU: AMD Ryzen AI 9 HX 470 w/ Radeon 890M (12 cores, 24 threads); installed RAM: 93.4 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 615.71.09; CUDA: not read
- Strata: project 0.1.43, commit deb3098d; engine 0.1.43
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 16.7 (13.7-22.6) | 1.40 (1.27-1.55) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 230.9 (227.0-232.1) | not measured | 17.50 (17.41-17.81) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 237.3 (204.8-240.2) | not measured | 67.81 (66.97-78.53) |
| agent session, follow-up turns | 14481 (14463-14526) at the last turn | see below | up to 64 | 3 | see below | 16.3 (13.8-21.1) | 15.20 (9.23-17.35) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.9); earlier-turn tokens read again per session: 144 (4-145) (earlier turns that the cache did not hold; the 13635 (13633-13636) new tokens of the files are read once); first turn (nothing cached) first token: 5.07 s (5.03-5.08).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 123.8 | 5.078 | 18.2 |
| 1 | 3717 | 634 | 3083 | 200.5 | 15.53 | 21.1 |
| 2 | 5250 | 3780 | 1470 | 158.4 | 9.411 | 15.7 |
| 3 | 8694 | 5313 | 3381 | 198.4 | 17.173 | 15.7 |
| 4 | 11618 | 8757 | 2861 | 189.9 | 15.201 | 16.0 |
| 5 | 14526 | 11682 | 2844 | 190.5 | 15.045 | 16.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 642 s.
