from __future__ import annotations

import time

import numpy as np

from feature_extraction.core import extract_features
from preprocessing.filters import preprocess_iq


def main() -> int:
    fs = 1_000_000.0
    n = 100_000

    rng = np.random.default_rng(7)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    iq = preprocess_iq(iq)

    t0 = time.perf_counter()
    extract_features(iq, fs, cache_key=("bench", 0, n), use_parallel=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Feature extraction latency for 100k samples: {elapsed_ms:.2f} ms")
    if elapsed_ms <= 200.0:
        print("PASS: latency target met (< 200 ms)")
        return 0

    print("WARN: latency target not met on this machine/profile")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
