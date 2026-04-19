from __future__ import annotations

DEFAULT_SAMPLE_RATE = 1_000_000.0
DEFAULT_CHUNK_SIZE = 100_000
MAX_WORKERS = 4
EPS = 1e-12

STAGES = [
    "signal_nature",
    "channel",
    "modulation",
    "protocol",
    "application",
]
