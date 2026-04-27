from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks, firwin, lfilter, peak_widths

from utils.config import EPS


@dataclass
class ChannelCandidate:
    frequency_offset_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    snr_db: float
    iq: np.ndarray


def _estimate_snr_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -30.0
    p = np.abs(x) ** 2
    signal = float(np.mean(p))
    noise = float(np.percentile(p, 20))
    return float(10.0 * np.log10((signal + EPS) / (noise + EPS)))


def _extract_channel(iq: np.ndarray, fs: float, center_hz: float, bw_hz: float) -> ChannelCandidate | None:
    n = iq.size
    if n < 1024:
        return None

    t = np.arange(n, dtype=np.float64) / fs
    mixed = iq * np.exp(-1j * 2.0 * np.pi * center_hz * t)

    decim = max(1, int(fs / max(2.5 * bw_hz, 10_000.0)))
    fs_out = fs / decim

    cutoff_hz = max(2_500.0, min(0.45 * fs_out, 0.65 * bw_hz))
    taps = firwin(numtaps=129, cutoff=cutoff_hz, fs=fs)
    filt = lfilter(taps, [1.0], mixed)

    valid = filt[128:]
    if valid.size < 512:
        return None
    dec = valid[::decim]
    if dec.size < 1024:
        return None

    snr_db = _estimate_snr_db(dec)
    return ChannelCandidate(
        frequency_offset_hz=float(center_hz),
        bandwidth_hz=float(min(max(5_000.0, bw_hz), 50_000.0)),
        sample_rate_hz=float(fs_out),
        snr_db=snr_db,
        iq=dec.astype(np.complex64),
    )


def detect_and_extract_channels(
    iq: np.ndarray,
    fs: float,
    min_bw_hz: float = 5_000.0,
    max_bw_hz: float = 50_000.0,
    max_channels: int = 8,
) -> List[ChannelCandidate]:
    if iq.size < 2048:
        return [
            ChannelCandidate(
                frequency_offset_hz=0.0,
                bandwidth_hz=float(min(max_bw_hz, fs)),
                sample_rate_hz=float(fs),
                snr_db=_estimate_snr_db(iq),
                iq=iq.astype(np.complex64),
            )
        ]

    nfft = min(131_072, 1 << (iq.size - 1).bit_length())
    segment = iq[:nfft]
    if segment.size < nfft:
        pad = np.zeros(nfft, dtype=np.complex64)
        pad[: segment.size] = segment
        segment = pad

    window = np.hanning(nfft)
    spec = np.fft.fftshift(np.fft.fft(segment * window))
    pwr = np.abs(spec) ** 2
    pwr_db = 10.0 * np.log10(pwr + EPS)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    bin_hz = float(abs(freqs[1] - freqs[0])) if freqs.size > 1 else fs

    smooth = np.convolve(pwr_db.astype(np.float64), np.ones(9) / 9.0, mode="same")
    distance_bins = max(2, int(min_bw_hz / max(bin_hz, EPS)))
    prominence_db = 6.0
    peaks, props = find_peaks(smooth, prominence=prominence_db, distance=distance_bins)

    channels: List[ChannelCandidate] = []
    taken: List[tuple[float, float]] = []

    if peaks.size:
        widths, _, _, _ = peak_widths(smooth, peaks, rel_height=0.5)
        order = np.argsort(props.get("prominences", np.zeros_like(peaks)))[::-1]

        for i in order:
            peak_bin = int(peaks[i])
            est_bw = float(widths[i] * bin_hz * 1.6)
            bw_hz = float(min(max_bw_hz, max(min_bw_hz, est_bw)))
            center = float(freqs[peak_bin])

            overlap = False
            for c0, bw0 in taken:
                if abs(center - c0) < 0.5 * (bw_hz + bw0):
                    overlap = True
                    break
            if overlap:
                continue

            ch = _extract_channel(iq, fs, center, bw_hz)
            if ch is None:
                continue
            channels.append(ch)
            taken.append((center, bw_hz))
            if len(channels) >= max_channels:
                break

    if not channels:
        channels.append(
            ChannelCandidate(
                frequency_offset_hz=0.0,
                bandwidth_hz=float(min(max_bw_hz, fs)),
                sample_rate_hz=float(fs),
                snr_db=_estimate_snr_db(iq),
                iq=iq.astype(np.complex64),
            )
        )

    return channels
