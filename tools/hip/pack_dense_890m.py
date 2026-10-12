#!/usr/bin/env python3
"""Lossless 890M dense-tensor *layout experiment*, NOT a runnable Strata pack.

Reads GGUF headers, selects dense quantized tensors, and optionally writes an
alignment-preserving, packed sidecar plus sha256 manifest. Tensor contents are
byte-identical. Strata does NOT load this sidecar yet; this is a controlled
basis for a future 890M reader/GEMV co-design, not a throughput improvement.
No precision reduction or changes to the RTX overflow cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile

ALIGN = 256
COPY_CHUNK = 1 << 20


def is_dense(name: str) -> bool:
    lower = name.lower()
    return (name.endswith('.weight') and
            'ffn_' not in lower and 'per_layer_token_embd' not in lower and
            'token_embd' not in lower and 'output.weight' not in lower)


def make_plan(g: GGUFFile, max_mib: int = 0) -> list[dict]:
    entries = []
    occupied = 0
    budget = max_mib * (1 << 20) if max_mib else None
    for t in g.tensors:
        n = t.expected_bytes()
        if not is_dense(t.name) or n is None or n <= 0:
            continue
        if budget is not None and occupied + n > budget:
            continue
        # The source tensor bytes are already quantized in GGUF; never decode/requantize.
        entries.append({'name': t.name, 'dtype': t.type_name, 'shape': list(t.shape),
                        'source_offset': g.data_start + t.offset, 'bytes': n,
                        'packed_offset': (occupied + ALIGN - 1) // ALIGN * ALIGN})
        occupied = entries[-1]['packed_offset'] + n
    return entries


def write_sidecar(source: Path, sidecar: Path, entries: list[dict]) -> None:
    with source.open('rb') as src, sidecar.open('wb') as dst:
        for entry in entries:
            pad = entry['packed_offset'] - dst.tell()
            if pad < 0:
                raise ValueError('overlapping packed tensors')
            dst.write(b'\0' * pad)
            src.seek(entry['source_offset'])
            h = hashlib.sha256()
            remaining = entry['bytes']
            while remaining:
                piece = src.read(min(remaining, COPY_CHUNK))
                if not piece:
                    raise ValueError('truncated source tensor: ' + entry['name'])
                dst.write(piece)
                h.update(piece)
                remaining -= len(piece)
            entry['sha256'] = h.hexdigest()


def verify_sidecar(source: Path, sidecar: Path, entries: list[dict]) -> bool:
    with source.open('rb') as src, sidecar.open('rb') as dst:
        for entry in entries:
            src.seek(entry['source_offset'])
            dst.seek(entry['packed_offset'])
            remaining = entry['bytes']
            while remaining:
                n = min(remaining, COPY_CHUNK)
                a, b = src.read(n), dst.read(n)
                if len(a) != n or a != b:
                    return False
                remaining -= n
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', required=True, type=Path, help='one GGUF shard')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--sidecar', type=Path, help='opt-in actual byte-for-byte repack')
    ap.add_argument('--max-mib', type=int, default=512, help='subset cap; 0 selects all eligible tensors')
    args = ap.parse_args()
    if args.max_mib < 0:
        ap.error('--max-mib must be nonnegative')
    g = GGUFFile(args.gguf)
    entries = make_plan(g, args.max_mib)
    if args.sidecar:
        if args.sidecar.resolve() == args.gguf.resolve():
            ap.error('sidecar must differ from input')
        write_sidecar(args.gguf, args.sidecar, entries)
        if not verify_sidecar(args.gguf, args.sidecar, entries):
            raise SystemExit('repack verification failed')
    payload = {'schema': 1, 'device_target': 'Radeon 890M / gfx1150',
               'runtime_supported': False, 'lossless': True, 'alignment': ALIGN,
               'source': str(args.gguf), 'sidecar': str(args.sidecar) if args.sidecar else None,
               'tensor_count': len(entries), 'tensor_bytes': sum(e['bytes'] for e in entries),
               'entries': entries}
    args.manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f"{len(entries)} dense tensors, {payload['tensor_bytes'] / (1<<20):.1f} MiB; this is an OFFLINE experiment")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
