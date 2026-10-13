#!/usr/bin/env python3
"""Controlled 890M experiment runner (always fresh engine process per arm).

Pass the actual Strata executable, model and prompt flags after '--'; this
runner adds --890m-prefill-fusion only in the experimental arm. Each invocation
starts a new process; adjust run count for noise and watch GPU clocks/temperature.
Example:
  python tools/hip/sweep_890m_prefill.py --runs 5 --out sweep.json -- \
      build/strata generate --pack pack/iq4 --tokens-file prompt.txt --prefill auto:8192 --max-new 24
This is NOT a before/after benchmark until the token dump and model are fixed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

SPEED = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)')


def parse(output):
    m = SPEED.search(output)
    return ({'prompt': int(m[1]), 'reused': int(m[2]), 'read': int(m[3]),
             'prompt_ms': int(m[4]), 'prompt_tps': float(m[5])} if m else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not cmd or args.runs < 1:
        ap.error('provide -- <engine> generate ... and at least one run')
    records = []
    for run in range(args.runs):
        for mode in ('baseline', 'fused'):
            env = os.environ.copy()
            env['STRATA_PREFILL_TIMING'] = '1'
            # The opt-in flag gates the fast=2 path against the actual HIP device.
            env.pop('STRATA_GFX1150_FAST', None)
            arm = [*cmd, '--890m-prefill-fusion'] if mode == 'fused' else cmd
            t = time.perf_counter()
            proc = subprocess.run(arm, env=env, capture_output=True, text=True)
            result = parse(proc.stdout + '\n' + proc.stderr)
            records.append({'run': run, 'arm': mode, 'exit_code': proc.returncode,
                            'elapsed_s': round(time.perf_counter()-t, 3), 'timings': result,
                            'stderr_tail': proc.stderr[-700:]})
            if proc.returncode or result is None:
                args.out.write_text(json.dumps(records, indent=2) + '\n')
                raise SystemExit(f'{mode} failed or produced no timing; check {args.out}')
    summary = {arm: statistics.median(r['timings']['prompt_tps'] for r in records if r['arm'] == arm)
               for arm in ('baseline', 'fused')}
    args.out.write_text(json.dumps({'records': records, 'median_prompt_tps': summary}, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
