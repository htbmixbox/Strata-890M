#!/usr/bin/env python3
"""Summarize Strata's existing HIP prefill and decode timeline instrumentation.

Reads engine stderr; does not change the engine, or pretend a HIP profiler is
available. Use with STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1
STRATA_VERIFY_PROFILE=1; retain rocprofv3 traces for *individual kernels*.

Example: python tools/hip/profile_890m.py engine.log --json report.json
The collected stage sums are approximate GPU/event timelines, not CUDA/HIP
kernel invocation timings. Do not add wall + GPU time (they overlap).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import statistics

PREFILL = re.compile(r"strata prefill timing: (\d+) tokens, GPU timeline ([\d.]+) ms, wall ([\d.]+) ms, host staging ([\d.]+) ms:(.*)")
STAGE = re.compile(r"([\w/-]+)\s+([\d.]+)\s+\(([\d.]+)%\)")
DECODE = re.compile(r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+)")


def parse_log(text: str) -> dict:
    prefill, decode = [], []
    for line in text.splitlines():
        if m := PREFILL.search(line):
            stage = {k: {"ms": float(ms), "pct": float(pct)} for k, ms, pct in STAGE.findall(m[5])}
            prefill.append({"tokens": int(m[1]), "gpu_ms": float(m[2]), "wall_ms": float(m[3]),
                            "host_staging_ms": float(m[4]), "stages": stage})
        if m := DECODE.search(line):
            decode.append({"windows": int(m[1]), "window_size": float(m[2]),
                           "tokens_per_window": float(m[3]), "ms_per_window": float(m[4]),
                           "verify_ms_per_window": float(m[5])})
    def med(rows: list[dict], key: str):
        return statistics.median(r[key] for r in rows) if rows else None
    all_stages = sorted({key for row in prefill for key in row["stages"]})
    return {"schema": 1, "hardware_scope": "Radeon 890M / gfx1150 (operator must verify)",
            "prefill": prefill, "decode": decode,
            "summary": {"prefill_runs": len(prefill), "decode_reports": len(decode),
                        "prefill_tps_median": (statistics.median(r["tokens"] / (r["wall_ms"] / 1000)
                                                          for r in prefill if r["wall_ms"] > 0)
                                               if any(r["wall_ms"] > 0 for r in prefill) else None),
                        "decode_ms_per_window_median": med(decode, "ms_per_window"),
                        "prefill_stage_ms_medians": {name: statistics.median(r["stages"][name]["ms"]
                                                       for r in prefill if name in r["stages"])
                                                     for name in all_stages}}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path, help="write full machine-readable result")
    args = ap.parse_args()
    out = parse_log(args.log.read_text(encoding="utf-8", errors="replace"))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    return 0 if out["prefill"] or out["decode"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
