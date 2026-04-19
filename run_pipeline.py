from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import SigIntPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SIGINT IQ classification pipeline")
    parser.add_argument("file", type=Path, help="Path to raw IQ file")
    parser.add_argument("--dtype", choices=["int8", "int16", "float32"], default="int16")
    parser.add_argument("--fs", type=float, default=1_000_000.0)
    parser.add_argument("--chunk", type=int, default=100_000)
    args = parser.parse_args()

    pipe = SigIntPipeline(sample_rate=args.fs, chunk_size=args.chunk)
    result, _, _ = pipe.process_file(args.file, args.dtype)

    print("Signal Type:", result.signal_type)
    print("Channel Type:", result.channel_type)
    print("Modulation:", result.modulation)
    print("Protocol:", result.protocol)
    print("Application:", result.application)
    print("Confidence (%):", f"{result.confidence * 100:.2f}")
    print("Top 3 candidates:")
    for c in result.top_candidates:
        print(
            f"- {c.signal_type}/{c.channel_type}/{c.modulation}/{c.protocol}/{c.application}: {c.confidence * 100:.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
