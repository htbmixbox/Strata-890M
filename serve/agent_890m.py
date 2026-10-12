"""Opt-in Radeon 890M admission policy for resumed agent sessions.

The engine already maintains prompt/session checkpoints. This favors cached
continuations based on their *unread* prefix, rather than rewriting checkpoints.
No GPU operations are performed here. Aging prevents persistent starvation.
"""
from __future__ import annotations

AGE_LIMIT_S = 3.0


def cached_prefix(prompt: list[int], held: list[list[int]]) -> int:
    """Longest whole stored session prefix that can be reused by this request."""
    return max((len(h) for h in held if h and len(h) < len(prompt) and prompt[:len(h)] == h), default=0)


def next_waiter(entries: list[list], now: float) -> list | None:
    """Entries: [total_tokens, admitted_at, ticket, fresh_tokens].

    If anybody waited >= 3 seconds, oldest wins. Otherwise admit the request
    that needs the least new prefill, FIFO for ties. No token data is stored in
    entries and this does not promise a checkpoint is still resident.
    """
    if not entries:
        return None
    old = [e for e in entries if now - e[1] >= AGE_LIMIT_S]
    if old:
        return min(old, key=lambda e: (e[1], e[2]))
    return min(entries, key=lambda e: (e[3], e[2]))
