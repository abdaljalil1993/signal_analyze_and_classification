from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde, kurtosis, skew

from utils.cache import LRUFeatureCache
from utils.config import EPS, MAX_WORKERS

_FEATURE_CACHE = LRUFeatureCache(capacity=128)


def _bimodality_coefficient(x: np.ndarray) -> float:
    if x.size < 32:
        return 0.0
    s = float(skew(x, bias=False))
    k = float(kurtosis(x, fisher=False, bias=False))
    if k <= 0.0:
        return 0.0
    return float((s * s + 1.0) / (k + EPS))


def _kmeans_states_1d(x: np.ndarray, k: int, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.zeros((k,), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    q = np.linspace(0.0, 100.0, k)
    centers = np.percentile(x, q).astype(np.float64)
    labels = np.zeros((x.size,), dtype=np.int32)
    for _ in range(iters):
        d2 = (x[:, None] - centers[None, :]) ** 2
        labels = np.argmin(d2, axis=1).astype(np.int32)
        moved = 0.0
        for i in range(k):
            m = labels == i
            if np.any(m):
                new_c = float(np.mean(x[m]))
                moved += abs(new_c - centers[i])
                centers[i] = new_c
        if moved < 1e-3:
            break
    return np.sort(centers), labels


def _gmm_states_1d(x: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Lightweight EM for 1D Gaussian mixtures to avoid external ML dependencies.
    n = x.size
    if n == 0:
        return np.zeros((k,), dtype=np.float64), np.zeros((k,), dtype=np.float64), np.zeros((k,), dtype=np.float64)
    means = np.percentile(x, np.linspace(0.0, 100.0, k)).astype(np.float64)
    vars_ = np.full((k,), max(float(np.var(x)), 1e-3), dtype=np.float64)
    weights = np.full((k,), 1.0 / k, dtype=np.float64)

    for _ in range(iters):
        probs = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            var_j = max(vars_[j], 1e-6)
            norm = 1.0 / np.sqrt(2.0 * np.pi * var_j)
            probs[:, j] = weights[j] * norm * np.exp(-0.5 * ((x - means[j]) ** 2) / var_j)
        denom = np.sum(probs, axis=1, keepdims=True) + EPS
        resp = probs / denom
        nk = np.sum(resp, axis=0) + EPS
        weights = nk / np.sum(nk)
        means = np.sum(resp * x[:, None], axis=0) / nk
        for j in range(k):
            vars_[j] = np.sum(resp[:, j] * ((x - means[j]) ** 2)) / nk[j]
        vars_ = np.maximum(vars_, 1e-6)

    order = np.argsort(means)
    return means[order], np.sqrt(vars_[order]), weights[order]


def _fallback_if_state_estimate(inst_freq_dn: np.ndarray, if_std: float) -> tuple[int, float, float, float]:
    # Returns: state_count, spacing_mean, spacing_cv, reliability
    if inst_freq_dn.size < 64 or if_std < 1e-6:
        return 0, 0.0, 1.0, 0.0

    x = inst_freq_dn.astype(np.float64)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return 0, 0.0, 1.0, 0.0

    bc = _bimodality_coefficient(x)
    best_states = 0
    best_reliability = 0.0
    best_spacing = 0.0
    best_cv = 1.0

    for k in range(2, 6):
        centers, labels = _kmeans_states_1d(x, k)
        counts = np.array([np.count_nonzero(labels == i) for i in range(k)], dtype=np.float64)
        active = counts > (0.06 * x.size)
        c_active = centers[active]
        if c_active.size < 2:
            continue
        spacings = np.diff(np.sort(c_active))
        if spacings.size == 0:
            continue
        spacing_mean = float(np.mean(spacings))
        spacing_cv = float(np.std(spacings) / (spacing_mean + EPS))
        sep = float(np.min(spacings) / (if_std + EPS))
        rel = float(min(1.0, 0.45 * min(1.0, sep) + 0.35 * min(1.0, 1.0 - spacing_cv) + 0.2 * min(1.0, bc)))
        if rel > best_reliability:
            best_reliability = rel
            best_states = int(c_active.size)
            best_spacing = spacing_mean
            best_cv = spacing_cv

    # GMM-based refinement for state count reliability.
    gmm_best = 0
    gmm_rel = 0.0
    for k in range(2, 6):
        means, sigmas, weights = _gmm_states_1d(x, k)
        active = weights > 0.06
        m = means[active]
        s = sigmas[active]
        if m.size < 2:
            continue
        spacings = np.diff(np.sort(m))
        if spacings.size == 0:
            continue
        sep = float(np.min(spacings) / (np.mean(s) + EPS))
        rel = float(min(1.0, 0.6 * min(1.0, sep / 2.0) + 0.4 * min(1.0, bc)))
        if rel > gmm_rel:
            gmm_rel = rel
            gmm_best = int(m.size)

    if gmm_rel > best_reliability + 0.05 and gmm_best >= 2:
        best_states = gmm_best
        best_reliability = gmm_rel

    if bc < 0.45 and best_reliability < 0.55:
        return 0, 0.0, 1.0, best_reliability
    return best_states, best_spacing, best_cv, best_reliability


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
    denoise_win = max(5, min(41, (iq.size // 3500) | 1))
    inst_freq_dn = _moving_average(inst_freq.astype(np.float32), denoise_win)
    if inst_freq_dn.size >= 11:
        sg_win = max(11, min(51, (inst_freq_dn.size // 80) | 1))
        if sg_win % 2 == 0:
            sg_win += 1
        sg_win = min(sg_win, inst_freq_dn.size - (1 - inst_freq_dn.size % 2))
        if sg_win >= 11 and sg_win < inst_freq_dn.size:
            inst_freq_dn = savgol_filter(inst_freq_dn, window_length=sg_win, polyorder=3, mode="interp").astype(np.float32)

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

    x_kde = np.linspace(float(np.min(inst_hist_data)), float(np.max(inst_hist_data)), 256, dtype=np.float64)
    if x_kde.size > 4 and float(np.std(inst_hist_data)) > 1e-6:
        try:
            kde = gaussian_kde(inst_hist_data.astype(np.float64), bw_method="scott")
            y_kde = kde(x_kde)
        except Exception:  # noqa: BLE001
            y_kde = np.zeros_like(x_kde)
    else:
        y_kde = np.zeros_like(x_kde)

    kde_peak = float(np.max(y_kde) if y_kde.size else 0.0)
    kde_mean = float(np.mean(y_kde) if y_kde.size else 0.0)
    kde_p2m = kde_peak / (kde_mean + EPS)
    is_flat_or_noisy = kde_peak <= 0.0 or kde_p2m < 1.35

    prom = 0.1 * float(kde_peak + EPS)
    min_width = max(1, x_kde.size // 140)
    peak_idx, props = find_peaks(
        y_kde,
        prominence=prom,
        width=min_width,
        distance=max(1, x_kde.size // 24),
    )
    if peak_idx.size and props.get("prominences") is not None:
        p_keep = props["prominences"] >= (0.1 * kde_peak)
        peak_idx = peak_idx[p_keep]

    if is_flat_or_noisy:
        peak_idx = np.zeros((0,), dtype=np.int64)
    peak_count = int(peak_idx.size)
    if peak_count >= 2:
        peak_freqs = x_kde[peak_idx]
        spacings = np.diff(np.sort(peak_freqs))
        peak_spacing_mean = float(np.mean(spacings))
        peak_spacing_cv = float(np.std(spacings) / (np.mean(spacings) + EPS))
    else:
        peak_spacing_mean = 0.0
        peak_spacing_cv = 1.0

    # Peak stability: compare top KDE peaks over overlapping slices of IF.
    stability_samples = []
    seg_len = max(512, inst_freq_dn.size // 4)
    if inst_freq_dn.size >= seg_len * 2 and peak_count > 0:
        for start in range(0, inst_freq_dn.size - seg_len + 1, max(1, seg_len // 2)):
            seg = inst_freq_dn[start : start + seg_len]
            if float(np.std(seg)) < 1e-6:
                continue
            x_seg = np.linspace(float(np.min(seg)), float(np.max(seg)), 128, dtype=np.float64)
            try:
                kde_seg = gaussian_kde(seg.astype(np.float64), bw_method="scott")
                y_seg = kde_seg(x_seg)
            except Exception:  # noqa: BLE001
                continue
            p_seg, _ = find_peaks(
                y_seg,
                prominence=0.08 * float(np.max(y_seg) + EPS),
                width=max(1, x_seg.size // 120),
                distance=max(1, x_seg.size // 24),
            )
            if p_seg.size > 0:
                stability_samples.append(float(x_seg[p_seg[np.argmax(y_seg[p_seg])]]))

    if len(stability_samples) >= 2 and peak_spacing_mean > 0.0:
        peak_stability = float(1.0 / (1.0 + np.std(stability_samples) / (peak_spacing_mean + EPS)))
    elif len(stability_samples) >= 2:
        peak_stability = float(1.0 / (1.0 + np.std(stability_samples) / (if_std + EPS)))
    else:
        peak_stability = 0.0

    chirp_corr = 0.0
    if inst_freq_dn.size > 16:
        t = np.linspace(-1.0, 1.0, inst_freq_dn.size, dtype=np.float32)
        f0 = inst_freq_dn - np.mean(inst_freq_dn)
        clip = np.percentile(np.abs(f0), 95)
        if clip > 0:
            f0 = np.clip(f0, -clip, clip)
        fi = f0 / (np.std(f0) + EPS)
        chirp_corr = float(abs(np.corrcoef(fi, t)[0, 1])) if fi.size > 3 else 0.0
    if chirp_corr < 0.35:
        chirp_corr = 0.0

    seg_len_hop = max(128, inst_freq_dn.size // 20)
    hop_events = 0
    if inst_freq_dn.size >= seg_len_hop * 3:
        seg_centers = []
        for start in range(0, inst_freq_dn.size - seg_len_hop + 1, seg_len_hop):
            seg = inst_freq_dn[start : start + seg_len_hop]
            seg_centers.append(float(np.median(seg)))
        if len(seg_centers) >= 3:
            diffs = np.abs(np.diff(np.array(seg_centers, dtype=np.float64)))
            hop_thr = max(peak_spacing_mean * 0.45, if_std * 0.6, 500.0)
            hop_events = int(np.count_nonzero(diffs >= hop_thr))
    fhss_detected = 1.0 if (hop_events >= 2 and chirp_corr == 0.0 and if_std > 1_000.0) else 0.0

    valid_peak_count = peak_count if (peak_count >= 2 and not is_flat_or_noisy) else 0
    fallback_state_count, fallback_spacing, fallback_cv, fallback_reliability = _fallback_if_state_estimate(inst_freq_dn, if_std)
    if valid_peak_count == 0 and fallback_state_count >= 2 and fallback_reliability >= 0.55:
        valid_peak_count = int(fallback_state_count)
        if peak_spacing_mean <= 0.0:
            peak_spacing_mean = float(fallback_spacing)
            peak_spacing_cv = float(fallback_cv)

    if_reliable = 1.0 if (valid_peak_count >= 2 and not is_flat_or_noisy) else 0.0
    multi_fsk_score = float(min(1.0, max(0.0, (valid_peak_count - 2) / 4.0)) * (0.4 + 0.6 * peak_stability))

    return {
        "if_std": if_std,
        "if_skew": if_skew,
        "if_peak_count": float(valid_peak_count),
        "if_peak_spacing_hz": peak_spacing_mean,
        "if_peak_spacing_cv": peak_spacing_cv,
        "if_peak_stability": peak_stability,
        "if_hist_valid": 0.0 if is_flat_or_noisy else 1.0,
        "if_reliability": if_reliable,
        "if_fallback_state_count": float(fallback_state_count),
        "if_fallback_reliability": float(fallback_reliability),
        "inst_freq_mean": float(np.mean(inst_freq_dn)),
        "inst_freq_denoised_std": float(np.std(inst_freq_dn)),
        "chirp_linearity": chirp_corr,
        "fhss_detected": fhss_detected,
        "multi_fsk_score": multi_fsk_score,
        "if_kde_max": kde_peak,
        "if_kde_peak_to_mean": kde_p2m,
        "if_hist_max": float(np.max(hist_smooth) if hist_smooth.size else 0.0),
        "if_hist_bin_span_hz": float(edges[-1] - edges[0]) if edges.size else 0.0,
    }


def _cyclostationary_features(iq: np.ndarray, fs: float) -> Dict[str, float]:
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

    # Lightweight SCF proxy: dominant non-DC cyclic line in squared-envelope spectrum.
    mag2c = mag2 - np.mean(mag2)
    scf = np.abs(np.fft.rfft(mag2c.astype(np.float64)))
    scf_freqs = np.fft.rfftfreq(mag2c.size, d=1.0 / fs)
    if scf.size > 3:
        scf[0] = 0.0
        peaks, _ = find_peaks(
            scf,
            prominence=0.12 * float(np.max(scf) + EPS),
            distance=max(1, scf.size // 512),
            width=max(1, scf.size // 2048),
        )
        valid = peaks[scf_freqs[peaks] >= 100.0] if peaks.size else np.zeros((0,), dtype=np.int64)
        idx = int(valid[np.argmax(scf[valid])]) if valid.size else int(np.argmax(scf))
        scf_bin = float(idx)
        scf_peak_ratio = float(scf[idx] / (np.mean(scf) + EPS))
        scf_symbol_rate_hz = float(scf_freqs[idx])
    else:
        scf_bin = 0.0
        scf_peak_ratio = 0.0
        scf_symbol_rate_hz = 0.0

    symbol_lag = int(np.argmax(np.abs(ac_side)) + 1) if ac_side.size else 0

    return {
        "cyclic_strength": cyclic_strength,
        "autocorr_peak": ac_peak,
        "cyclic_autocorr_peak": cyclic_ac_peak,
        "symbol_lag_samples": float(symbol_lag),
        "scf_symbol_bin": scf_bin,
        "scf_peak_ratio": scf_peak_ratio,
        "scf_fft_len": float(iq.size),
        "scf_symbol_rate_hz": scf_symbol_rate_hz,
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
    # Normalize IQ cloud before clustering to keep constellation stability scale consistent across captures.
    iq_norm = iq / (np.sqrt(np.mean(np.abs(iq) ** 2)) + EPS)
    points = np.column_stack([np.real(iq_norm), np.imag(iq_norm)]).astype(np.float32)
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
    mu = float(np.mean(env))
    sd = float(np.std(env))
    thr_hi = mu + 1.0 * sd
    thr_lo = mu + 0.4 * sd
    active = np.zeros(env.size, dtype=np.bool_)
    on = False
    for i, v in enumerate(env):
        if not on and v >= thr_hi:
            on = True
        elif on and v <= thr_lo:
            on = False
        active[i] = on

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
    burst_times = starts[valid_bursts].astype(np.float64) / max(fs, EPS) if starts.size else np.array([], dtype=np.float64)
    if burst_times.size >= 3:
        intervals = np.diff(burst_times)
        repetition = float(1.0 / (1.0 + np.std(intervals) / (np.mean(intervals) + EPS)))
    else:
        repetition = 0.0
    power = np.abs(iq) ** 2
    signal_power = float(np.mean(power))
    noise_floor = float(np.percentile(power, 20))
    snr_db = float(10.0 * np.log10((signal_power + EPS) / (noise_floor + EPS)))
    amp_stability = float(1.0 / (np.std(env) + EPS))

    env_centers, env_labels = _kmeans_states_1d(env.astype(np.float64), 2)
    env_sep = float(abs(env_centers[1] - env_centers[0]) / (np.std(env) + EPS)) if env_centers.size == 2 else 0.0
    env_bc = _bimodality_coefficient(env.astype(np.float64))
    p_hist, p_edges = np.histogram(power.astype(np.float64), bins=64)
    p_s = _gaussian_smooth(p_hist.astype(np.float32), sigma=1.0)
    p_peaks, p_props = find_peaks(p_s, prominence=0.1 * float(np.max(p_s) + EPS), width=max(1, p_s.size // 40))
    power_bimodal = p_peaks.size >= 2

    ook_gate = (env_bc >= 0.55) and (env_sep >= 1.0) and power_bimodal
    ook_score = float(min(1.0, max(0.0, (0.5 - duty_cycle) * 2.0)) * min(1.0, max(0.0, snr_db / 20.0))) if ook_gate else 0.0
    burstiness = float(np.std(active.astype(np.float32)))

    const_stability = float(1.0 / (cluster_var + EPS))
    const_stability = float(min(10.0, max(0.0, const_stability)))

    return {
        "constellation_stability": const_stability,
        "duty_cycle": duty_cycle,
        "packet_rate_hz": packet_rate,
        "burst_repetition_score": repetition,
        "snr_db": snr_db,
        "amplitude_stability": amp_stability,
        "ook_score": ook_score,
        "envelope_bimodality": float(env_bc),
        "power_hist_bimodal": 1.0 if power_bimodal else 0.0,
        "envelope_state_separation": env_sep,
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

    if out.get("if_peak_count", 0.0) == 0.0:
        out["if_peak_valid"] = 0.0
        invalid += 1.0
    else:
        out["if_peak_valid"] = 1.0

    packet_rate = out.get("packet_rate_hz", 0.0)
    if packet_rate < 0.0:
        out["packet_rate_hz"] = 0.0
        invalid += 1.0
        core_invalid += 1.0
        out["packet_rate_valid"] = 0.0

    symbol_rate = float(out.get("scf_symbol_rate_hz", 0.0)) if out.get("scf_peak_ratio", 0.0) > 1.2 else 0.0
    if symbol_rate <= 0.0 or symbol_rate > fs / 2.0:
        symbol_rate = 0.0
        invalid += 1.0
        core_invalid += 1.0
        out["symbol_rate_valid"] = 0.0
    if 0.0 < symbol_rate < 100.0:
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
        return _cyclostationary_features(iq, fs)
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
