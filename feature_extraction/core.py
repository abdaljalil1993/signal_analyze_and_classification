from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable

import numpy as np

from utils.cache import LRUFeatureCache
from utils.config import EPS, MAX_WORKERS

_FEATURE_CACHE = LRUFeatureCache(capacity=128)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if x.size < 3 or sigma <= 0.0:
        return x
    radius = max(1, int(3.0 * sigma))
    idx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(idx**2) / (2.0 * sigma * sigma))
    kernel = kernel / (np.sum(kernel) + EPS)
    return np.convolve(x, kernel, mode="same")


def _detect_hist_peaks(hist_smooth: np.ndarray, rel_threshold: float = 0.35) -> np.ndarray:
    if hist_smooth.size < 3:
        return np.zeros((0,), dtype=np.int64)
    thr = rel_threshold * float(np.max(hist_smooth) + EPS)
    mid = hist_smooth[1:-1]
    left = hist_smooth[:-2]
    right = hist_smooth[2:]
    peaks = np.where((mid > left) & (mid > right) & (mid >= thr))[0] + 1
    if peaks.size <= 1:
        return peaks.astype(np.int64)

    # Keep strongest separated peaks to avoid overcounting noisy shoulders.
    min_sep = max(2, hist_smooth.size // 40)
    order = peaks[np.argsort(hist_smooth[peaks])[::-1]]
    kept = []
    for p in order:
        if all(abs(int(p) - int(k)) >= min_sep for k in kept):
            kept.append(int(p))
    kept = np.array(sorted(kept), dtype=np.int64)
    return kept


def _spectral_features(iq: np.ndarray, fs: float) -> Dict[str, float]:
    n = iq.size
    window = np.hanning(n).astype(np.float32) if n > 1 else np.ones(n, dtype=np.float32)
    spec = np.fft.fftshift(np.fft.fft(iq * window))
    psd = np.abs(spec) ** 2
    psd = psd / (np.sum(psd) + EPS)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))

    entropy = float(-np.sum(psd * np.log2(psd + EPS)))
    flatness = float(np.exp(np.mean(np.log(psd + EPS))) / (np.mean(psd) + EPS))
    cumsum = np.cumsum(psd)
    low_idx = int(np.searchsorted(cumsum, 0.01))
    high_idx = int(np.searchsorted(cumsum, 0.99))
    bw_percentile = float(abs(freqs[min(high_idx, n - 1)] - freqs[max(low_idx, 0)]))

    psd_db = 10.0 * np.log10(psd + EPS)
    noise_db = float(np.median(psd_db))
    occ = psd_db > (noise_db + 6.0)
    if np.any(occ):
        occ_idx = np.where(occ)[0]
        bw_occ = float(abs(freqs[occ_idx[-1]] - freqs[occ_idx[0]]))
        bandwidth = bw_occ
    else:
        bandwidth = bw_percentile

    bandwidth = float(min(max(0.0, bandwidth), fs))

    occ_mask = psd > (3.0 * np.mean(psd))
    subcarrier_like = float(np.mean(occ_mask))
    peak_to_avg = float(np.max(psd) / (np.mean(psd) + EPS))

    return {
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "bandwidth_hz": bandwidth,
        "peak_to_avg": peak_to_avg,
        "subcarrier_structure": subcarrier_like,
    }


def _instantaneous_features(iq: np.ndarray, fs: float) -> Dict[str, float]:
    phase = np.unwrap(np.angle(iq))
    dphi = np.diff(phase, prepend=phase[0])
    inst_freq = dphi * fs / (2.0 * np.pi)
    denoise_win = max(3, min(31, (iq.size // 4000) | 1))
    inst_freq_dn = _moving_average(inst_freq.astype(np.float32), denoise_win)

    if_std = float(np.std(inst_freq_dn))
    if_skew = float(np.mean(((inst_freq_dn - np.mean(inst_freq_dn)) / (np.std(inst_freq_dn) + EPS)) ** 3))

    env = np.abs(iq)
    energy_thr = float(np.percentile(env, 35))
    valid_mask = env >= energy_thr
    if np.count_nonzero(valid_mask) >= 64:
        inst_hist_data = inst_freq_dn[valid_mask]
    else:
        inst_hist_data = inst_freq_dn

    hist, edges = np.histogram(inst_hist_data, bins=96)
    hist_smooth = _gaussian_smooth(hist.astype(np.float32), sigma=1.2)
    peaks = _detect_hist_peaks(hist_smooth, rel_threshold=0.35)
    peak_count = int(peaks.size)

    chirp_corr = 0.0
    if inst_freq_dn.size > 16:
        t = np.linspace(-1.0, 1.0, inst_freq_dn.size, dtype=np.float32)
        f0 = inst_freq_dn - np.mean(inst_freq_dn)
        clip = np.percentile(np.abs(f0), 95)
        if clip > 0:
            f0 = np.clip(f0, -clip, clip)
        fi = f0 / (np.std(f0) + EPS)
        chirp_corr = float(abs(np.corrcoef(fi, t)[0, 1])) if fi.size > 3 else 0.0

    multi_fsk_score = float(min(1.0, max(0.0, (peak_count - 2) / 6.0)))

    return {
        "if_std": if_std,
        "if_skew": if_skew,
        "if_peak_count": float(peak_count),
        "inst_freq_mean": float(np.mean(inst_freq_dn)),
        "inst_freq_denoised_std": float(np.std(inst_freq_dn)),
        "chirp_linearity": chirp_corr,
        "multi_fsk_score": multi_fsk_score,
        "if_hist_max": float(np.max(hist_smooth) if hist_smooth.size else 0.0),
        "if_hist_bin_span_hz": float(edges[-1] - edges[0]) if edges.size else 0.0,
    }


def _cyclostationary_features(iq: np.ndarray) -> Dict[str, float]:
    mag = np.abs(iq)
    mag2 = mag ** 2
    fft_mag2 = np.abs(np.fft.fft(mag2))
    fft_mag2 = fft_mag2 / (np.max(fft_mag2) + EPS)
    cyclic_strength = float(np.mean(np.sort(fft_mag2)[-8:]))

    x = iq / (np.sqrt(np.mean(np.abs(iq) ** 2)) + EPS)
    ac = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x)))
    ac = np.real(ac)
    max_lag = min(512, ac.size - 1)
    if max_lag > 0:
        lag_norm = np.arange(ac.size, dtype=np.float64)
        lag_norm = np.maximum(1.0, x.size - lag_norm)
        ac = ac / lag_norm
    ac = ac / (abs(ac[0]) + EPS)
    ac_side = ac[1 : max_lag + 1]
    ac_peak = float(np.max(np.abs(ac_side)) if ac_side.size else 0.0)

    lag = min(128, iq.size // 4)
    if lag > 0:
        cyc = iq[:-lag] * np.conj(iq[lag:])
        cyc_spec = np.abs(np.fft.fft(cyc))
        cyclic_ac_peak = float(np.max(cyc_spec) / (np.mean(cyc_spec) + EPS))
    else:
        cyclic_ac_peak = 0.0

    symbol_lag = int(np.argmax(np.abs(ac_side)) + 1) if ac_side.size else 0

    return {
        "cyclic_strength": cyclic_strength,
        "autocorr_peak": ac_peak,
        "cyclic_autocorr_peak": cyclic_ac_peak,
        "symbol_lag_samples": float(symbol_lag),
    }


def _kmeans_constellation(points: np.ndarray, k: int = 4, iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    if n == 0:
        return np.zeros((k, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    idx = np.linspace(0, n - 1, k, dtype=np.int64)
    centers = points[idx].copy()

    for _ in range(iters):
        d2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = np.mean(points[mask], axis=0)

    return centers, labels


def _constellation_and_burst_features(iq: np.ndarray, fs: float) -> Dict[str, float]:
    points = np.column_stack([np.real(iq), np.imag(iq)]).astype(np.float32)
    max_pts = min(4000, points.shape[0])
    if points.shape[0] > max_pts:
        sel = np.linspace(0, points.shape[0] - 1, max_pts, dtype=np.int64)
        points = points[sel]

    _, labels = _kmeans_constellation(points, k=4)
    if labels.size:
        cluster_var = 0.0
        for c in range(4):
            p = points[labels == c]
            if p.size:
                cluster_var += float(np.var(p[:, 0]) + np.var(p[:, 1]))
        cluster_var /= 4.0
    else:
        cluster_var = 0.0

    env = np.abs(iq)
    thr = np.mean(env) + 0.8 * np.std(env)
    active = env > thr
    edges = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    if starts.size and stops.size and stops[0] < starts[0]:
        stops = stops[1:]
    m = min(starts.size, stops.size)
    starts = starts[:m]
    stops = stops[:m]

    min_burst_samples = max(8, int(0.0002 * fs))
    valid_bursts = (stops - starts) >= min_burst_samples
    burst_starts = int(np.sum(valid_bursts))
    duty_cycle = float(np.mean(active))
    duration_s = float(iq.size / max(fs, EPS))
    packet_rate = float(burst_starts / max(duration_s, EPS))
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_floor = float(np.percentile(np.abs(iq) ** 2, 20))
    snr_db = float(10.0 * np.log10((signal_power + EPS) / (noise_floor + EPS)))
    amp_stability = float(1.0 / (np.std(env) + EPS))

    ook_score = float(min(1.0, max(0.0, (0.5 - duty_cycle) * 2.0)) * min(1.0, max(0.0, snr_db / 20.0)))
    burstiness = float(np.std(active.astype(np.float32)))

    return {
        "constellation_stability": float(1.0 / (cluster_var + EPS)),
        "duty_cycle": duty_cycle,
        "packet_rate_hz": packet_rate,
        "snr_db": snr_db,
        "amplitude_stability": amp_stability,
        "ook_score": ook_score,
        "burstiness": burstiness,
    }


def _validate_and_cross_check(features: Dict[str, float], fs: float) -> Dict[str, float]:
    out = dict(features)
    invalid = 0.0
    core_invalid = 0.0

    out["bandwidth_valid"] = 1.0
    out["symbol_rate_valid"] = 1.0
    out["packet_rate_valid"] = 1.0

    bw = out.get("bandwidth_hz", 0.0)
    if bw < 0.0 or bw > fs:
        out["bandwidth_hz"] = float(min(max(0.0, bw), fs))
        invalid += 1.0
        core_invalid += 1.0
        out["bandwidth_valid"] = 0.0

    if out.get("if_peak_count", 0.0) < 0.0:
        out["if_peak_count"] = 0.0
        invalid += 1.0

    packet_rate = out.get("packet_rate_hz", 0.0)
    if packet_rate < 0.0:
        out["packet_rate_hz"] = 0.0
        invalid += 1.0
        core_invalid += 1.0
        out["packet_rate_valid"] = 0.0

    symbol_lag = int(out.get("symbol_lag_samples", 0.0))
    if symbol_lag > 0:
        symbol_rate = fs / float(symbol_lag)
    else:
        symbol_rate = 0.0
    if symbol_rate <= 0.0 or symbol_rate > fs / 2.0:
        symbol_rate = 0.0
        invalid += 1.0
        core_invalid += 1.0
        out["symbol_rate_valid"] = 0.0
    out["symbol_rate_est_hz"] = float(symbol_rate)

    packet_max = min(5_000.0, symbol_rate / 2.0) if symbol_rate > 0 else 5_000.0
    if out.get("packet_rate_hz", 0.0) > packet_max:
        out["packet_rate_hz"] = float(packet_max)
        invalid += 1.0
        core_invalid += 1.0
        out["packet_rate_valid"] = 0.0

    # Cross-check physically plausible relation: occupied BW should be within Nyquist span.
    out["bandwidth_ratio_to_fs"] = float(out.get("bandwidth_hz", 0.0) / max(fs, EPS))
    if out["bandwidth_ratio_to_fs"] > 1.0:
        out["bandwidth_ratio_to_fs"] = 1.0
        invalid += 1.0

    # Heuristic realism check for low-rate telemetry class.
    low_rate = 1.0 if (symbol_rate > 0 and symbol_rate < 15_000.0 and out.get("packet_rate_hz", 0.0) < 200.0) else 0.0
    out["low_rate_telemetry_score"] = low_rate
    dsss_confirmed = 1.0 if (
        out.get("bandwidth_ratio_to_fs", 0.0) > 0.25
        and out.get("spectral_flatness", 0.0) > 0.5
        and out.get("cyclic_autocorr_peak", 0.0) > 2.0
    ) else 0.0
    out["dsss_confirmed"] = dsss_confirmed
    out["feature_invalid_count"] = invalid
    out["feature_validity_score"] = float(max(0.0, 1.0 - invalid / 6.0))
    out["core_feature_invalid_count"] = core_invalid
    out["core_feature_validity_score"] = float(max(0.0, 1.0 - core_invalid / 3.0))
    return out


def _extract_group(args: tuple[str, np.ndarray, float]) -> Dict[str, float]:
    name, iq, fs = args
    if name == "spectral":
        return _spectral_features(iq, fs)
    if name == "instant":
        return _instantaneous_features(iq, fs)
    if name == "cyclo":
        return _cyclostationary_features(iq)
    if name == "const_burst":
        return _constellation_and_burst_features(iq, fs)
    raise ValueError(f"Unknown feature group: {name}")


def extract_features(
    iq: np.ndarray,
    fs: float,
    cache_key: tuple | None = None,
    use_parallel: bool = True,
) -> Dict[str, float]:
    if cache_key is not None:
        cached = _FEATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    groups: Iterable[str] = ("spectral", "instant", "cyclo", "const_burst")
    features: Dict[str, float] = {}

    # On Windows, process spawn overhead dominates at smaller sizes.
    if use_parallel and iq.size >= 400_000:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = ex.map(_extract_group, ((g, iq, fs) for g in groups))
            for f in results:
                features.update(f)
    else:
        for g in groups:
            features.update(_extract_group((g, iq, fs)))

    features = _validate_and_cross_check(features, fs)

    if cache_key is not None:
        _FEATURE_CACHE.put(cache_key, features)
    return features
