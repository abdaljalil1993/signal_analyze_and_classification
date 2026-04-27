from __future__ import annotations

from typing import Callable, Dict


DetectorFn = Callable[[Dict[str, float], str, str, str], float]


class ProtocolDetectorRegistry:
    def __init__(self) -> None:
        self._detectors: Dict[str, DetectorFn] = {}

    def register(self, name: str, fn: DetectorFn) -> None:
        self._detectors[name] = fn

    def run(self, features: Dict[str, float], channel: str, modulation: str, nature: str) -> Dict[str, float]:
        return {
            name: max(0.0, min(1.0, fn(features, channel, modulation, nature)))
            for name, fn in self._detectors.items()
        }


def _range_match(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if low <= value <= high:
        mid = 0.5 * (low + high)
        span = max(high - low, 1.0)
        return max(0.6, 1.0 - abs(value - mid) / span)
    margin = max(0.25 * (high - low), 1.0)
    if value < low:
        return max(0.0, 1.0 - (low - value) / margin) * 0.35
    return max(0.0, 1.0 - (value - high) / margin) * 0.35


def _target_match(value: float, target: float, tol: float) -> float:
    if tol <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tol)


def _burst_signature(features: Dict[str, float]) -> float:
    burstiness = min(1.0, max(0.0, features.get("burstiness", 0.0)))
    repetition = min(1.0, max(0.0, features.get("burst_repetition_score", 0.0)))
    duty = min(1.0, max(0.0, features.get("duty_cycle", 0.0)))
    return min(1.0, 0.45 * burstiness + 0.35 * repetition + 0.2 * (1.0 - duty))


def _wifi_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    bw = features.get("bandwidth_hz", 0.0)
    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    if channel != "Wideband" or nature != "Digital":
        return 0.0
    mod_factor = 1.0 if modulation == "OFDM" else 0.45
    bw_score = _range_match(bw, 1_000_000.0, 25_000_000.0)
    sym_score = _range_match(symbol_rate, 100_000.0, 20_000_000.0) if symbol_rate > 0 else 0.3
    cyclic_score = _range_match(features.get("cyclic_strength", 0.0), 0.15, 1.0)
    burst_score = 1.0 - _burst_signature(features) * 0.5
    return float(min(1.0, (0.35 * bw_score + 0.25 * sym_score + 0.2 * cyclic_score + 0.2 * burst_score) * mod_factor))


def _lora_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    bw = features.get("bandwidth_hz", 0.0)
    chirp_strength = max(abs(features.get("if_skew", 0.0)), features.get("chirp_linearity", 0.0))
    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    if nature != "Digital":
        return 0.0
    mod_factor = 1.0 if modulation == "Chirp" else 0.6
    # Accept both full-band LoRa BW and narrowed channelized windows.
    bw_match = max(
        _target_match(bw, 125_000.0, 80_000.0),
        _target_match(bw, 250_000.0, 120_000.0),
        _target_match(bw, 500_000.0, 200_000.0),
        0.7 * _range_match(bw, 8_000.0, 60_000.0),
    )
    sym_score = _range_match(symbol_rate, 100.0, 50_000.0) if symbol_rate > 0 else 0.3
    burst_score = _burst_signature(features)
    chirp_score = min(1.0, chirp_strength)
    if chirp_score < 0.45:
        return 0.0
    return float(min(1.0, (0.3 * bw_match + 0.45 * chirp_score + 0.1 * sym_score + 0.15 * burst_score) * mod_factor))


def _dmr_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if nature != "Digital":
        return 0.0
    mod_factor = 1.0 if modulation in {"FSK", "MFSK", "OOK"} else 0.55

    if channel != "Narrowband":
        # Allow degraded scoring when channel stage is imperfect but narrowband DMR evidence exists.
        channel_factor = 0.7
    else:
        channel_factor = 1.0

    bw = features.get("bandwidth_hz", 0.0)
    if not (6_000.0 <= bw <= 20_000.0):
        return 0.0

    fm_analog_signature = (
        features.get("if_std", 0.0) > 8_000.0
        and features.get("if_peak_count", 0.0) <= 2.5
        and features.get("burstiness", 0.0) < 0.2
        and features.get("packet_rate_hz", 0.0) < 8.0
        and features.get("if_peak_stability", 0.0) < 0.55
    )
    if fm_analog_signature:
        return 0.0

    if features.get("duty_cycle", 0.0) > 0.75 and features.get("burstiness", 0.0) < 0.12:
        return 0.0

    if_peaks = features.get("if_peak_count", 0.0)
    if if_peaks < 2.0 or if_peaks > 5.5:
        return 0.0

    if features.get("if_reliability", 0.0) < 0.5:
        return 0.0

    spacing_cv = features.get("if_peak_spacing_cv", 1.0)
    if spacing_cv > 0.65:
        return 0.0

    cyc_ac = features.get("cyclic_autocorr_peak", 0.0)
    if cyc_ac < 2.5:
        return 0.0

    if if_peaks < 3.0 and cyc_ac < 6.0:
        return 0.0

    if features.get("packet_rate_hz", 0.0) < 0.5:
        return 0.0

    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    # Keep DMR candidate alive under noisy symbol-rate estimation, but with lower score.
    sym_score = _target_match(symbol_rate, 4_800.0, 3_800.0) if symbol_rate > 0 else 0.0

    chirp = max(features.get("chirp_linearity", 0.0), abs(features.get("if_skew", 0.0)))
    if chirp > 0.3:
        return 0.0

    peak_shape = max(0.0, 1.0 - abs(if_peaks - 4.0) / 2.0)
    spacing_score = max(0.0, 1.0 - spacing_cv / 0.65)
    cyclic_score = min(1.0, features.get("cyclic_strength", 0.0) / 0.8)
    tdma_score = min(1.0, cyc_ac / 6.0)
    burst_score = _burst_signature(features)
    dmr_region_boost = 0.0
    if 10_000.0 <= bw <= 15_000.0 and if_peaks >= 2.0 and tdma_score > 0.65:
        dmr_region_boost = 0.12
    if cyc_ac > 8.0 and 9_000.0 <= bw <= 18_000.0:
        dmr_region_boost += 0.1
    return float(min(1.0, (0.24 * peak_shape + 0.16 * spacing_score + 0.18 * sym_score + 0.2 * cyclic_score + 0.12 * tdma_score + 0.1 * burst_score + dmr_region_boost) * mod_factor * channel_factor))


def _rc_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if nature != "Digital":
        return 0.0
    channel_factor = 1.0 if channel == "Narrowband" else 0.75
    bw = features.get("bandwidth_hz", 0.0)
    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    bursty = features.get("burstiness", 0.0)
    repetition = features.get("burst_repetition_score", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)

    if bw >= 20_000.0:
        return 0.0
    if symbol_rate <= 0.0 or symbol_rate >= 5_000.0:
        return 0.0
    if bursty < 0.15:
        return 0.0
    if cyclic > 0.45:
        return 0.0
    if features.get("if_peak_count", 0.0) >= 3.5 and features.get("if_peak_count", 0.0) <= 4.5 and abs(symbol_rate - 4_800.0) <= 1_200.0:
        return 0.0

    bw_score = max(0.0, 1.0 - bw / 20_000.0)
    sr_score = max(0.0, 1.0 - symbol_rate / 5_000.0)
    burst_score = min(1.0, 0.55 * bursty + 0.45 * repetition)
    structure_low = max(0.0, 1.0 - min(1.0, cyclic / 0.45))
    return float(min(1.0, (0.3 * bw_score + 0.25 * sr_score + 0.25 * burst_score + 0.2 * structure_low) * channel_factor))


def _drone_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    duty = features.get("duty_cycle", 0.0)
    rate = features.get("packet_rate_hz", 0.0)
    bw = features.get("bandwidth_hz", 0.0)
    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)
    if nature == "Digital" and duty > 0.35:
        bw_score = _range_match(bw, 50_000.0, 5_000_000.0)
        rate_score = _range_match(rate, 10.0, 400.0)
        sym_score = _range_match(symbol_rate, 5_000.0, 2_000_000.0) if symbol_rate > 0 else 0.35
        cyclic_score = _range_match(cyclic, 0.15, 0.9)
        return float(min(1.0, 0.25 * bw_score + 0.25 * rate_score + 0.2 * sym_score + 0.15 * cyclic_score + 0.15 * duty))
    return 0.0


def _iot_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if nature != "Digital":
        return 0.0
    bw = features.get("bandwidth_hz", 0.0)
    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    if_peaks = features.get("if_peak_count", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)
    burst_score = _burst_signature(features)
    chirp = features.get("chirp_linearity", 0.0)
    if chirp > 0.45:
        return 0.0
    # IoT should not dominate very low-rate or strong-cyclic narrowband voice/control links.
    if symbol_rate <= 0.0 or symbol_rate < 600.0:
        return 0.0
    if cyclic > 0.62:
        return 0.0
    if 3.5 <= if_peaks <= 4.5 and abs(symbol_rate - 4_800.0) <= 1_800.0:
        return 0.0
    if 10_000.0 <= bw <= 20_000.0 and if_peaks >= 2.0 and abs(symbol_rate - 4_800.0) <= 2_200.0:
        return 0.0
    if features.get("cyclic_autocorr_peak", 0.0) >= 2.8 and 8_000.0 <= bw <= 20_000.0:
        return 0.0

    bw_score = _range_match(bw, 8_000.0, 220_000.0)
    sym_score = _range_match(symbol_rate, 700.0, 40_000.0)
    peak_score = _range_match(if_peaks, 1.0, 4.0)
    cyclic_score = _range_match(cyclic, 0.08, 0.55)
    return float(min(1.0, 0.22 * bw_score + 0.22 * sym_score + 0.16 * peak_score + 0.2 * cyclic_score + 0.2 * burst_score))


def _unknown_digital_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    duty = features.get("duty_cycle", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)
    const = features.get("constellation_stability", 0.0)
    if nature == "Digital":
        return min(1.0, 0.25 + 0.35 * min(1.0, cyclic) + 0.25 * min(1.0, const / 3.0) + 0.15 * duty)
    return 0.0


def _unknown_narrowband_digital_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if channel != "Narrowband":
        return 0.0
    low_rate = features.get("low_rate_telemetry_score", 0.0)
    bursty = features.get("burstiness", 0.0)
    digitality = min(1.0, features.get("cyclic_strength", 0.0) + features.get("cyclic_autocorr_peak", 0.0) / 8.0)
    return min(1.0, 0.18 + 0.35 * low_rate + 0.22 * bursty + 0.25 * digitality)


def _analog_broadcast_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if nature != "Analog":
        return 0.0
    if modulation in {"AM", "FM"}:
        return min(1.0, 0.4 + 0.6 * min(1.0, features.get("amplitude_stability", 0.0) / 8.0))
    return 0.0


def _analog_fm_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if nature != "Analog" or modulation != "FM":
        return 0.0
    bw = features.get("bandwidth_hz", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)
    burst = _burst_signature(features)
    bw_score = _range_match(bw, 50_000.0, 250_000.0)
    cyclic_score = max(0.0, 1.0 - min(1.0, cyclic / 0.25))
    return float(min(1.0, 0.5 * bw_score + 0.3 * cyclic_score + 0.2 * (1.0 - burst)))


def _unknown_signal_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    core_valid = features.get("core_feature_validity_score", 1.0)
    validity = features.get("feature_validity_score", 1.0)
    return min(1.0, 0.15 + 0.7 * (1.0 - core_valid) + 0.15 * (1.0 - validity))


def build_default_registry() -> ProtocolDetectorRegistry:
    reg = ProtocolDetectorRegistry()
    reg.register("WiFi-like", _wifi_detector)
    reg.register("LoRa-like", _lora_detector)
    reg.register("DMR-like", _dmr_detector)
    reg.register("RC-like", _rc_detector)
    reg.register("Drone-link-like", _drone_detector)
    reg.register("IoT-like", _iot_detector)
    reg.register("Analog FM", _analog_fm_detector)
    reg.register("Unknown Digital Signal", _unknown_digital_detector)
    reg.register("Unknown Narrowband Digital Signal", _unknown_narrowband_digital_detector)
    reg.register("Analog Broadcast", _analog_broadcast_detector)
    reg.register("Unknown Signal", _unknown_signal_detector)
    return reg


_DEFAULT_REGISTRY = build_default_registry()


def detect_protocols(features: Dict[str, float], channel: str, modulation: str, nature: str) -> Dict[str, float]:
    return _DEFAULT_REGISTRY.run(features, channel, modulation, nature)
