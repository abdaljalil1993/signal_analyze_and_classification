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
    print("Best Channel Offset (Hz):", f"{result.best_channel_frequency_offset_hz:.1f}")
    print("Top 3 candidates:")
    for c in result.top_candidates:
        print(
            f"- {c.signal_type}/{c.channel_type}/{c.modulation}/{c.protocol}/{c.application}: {c.confidence * 100:.2f}%"
        )

    print("Per-channel classification scores:")
    for ch in result.per_channel_classification_scores:
        print(
            "- idx={idx:.0f}, f_off={fo:.1f} Hz, bw={bw:.1f} Hz, SNR={snr:.2f} dB, sel={sel:.3f}, conf={conf:.3f}, mod={mod}, proto={proto}".format(
                idx=float(ch.get("channel_index", 0.0)),
                fo=float(ch.get("frequency_offset_hz", 0.0)),
                bw=float(ch.get("bandwidth_hz", 0.0)),
                snr=float(ch.get("snr_db", 0.0)),
                sel=float(ch.get("selection_score", 0.0)),
                conf=float(ch.get("confidence", 0.0)),
                mod=str(ch.get("modulation", "")),
                proto=str(ch.get("protocol", "")),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
