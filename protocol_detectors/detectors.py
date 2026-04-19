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


def _wifi_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    bw = features.get("bandwidth_hz", 0.0)
    if channel == "Wideband" and modulation == "OFDM":
        return 0.6 + 0.4 * min(1.0, bw / 20_000_000.0)
    return 0.0


def _lora_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    bw = features.get("bandwidth_hz", 0.0)
    chirp_strength = max(abs(features.get("if_skew", 0.0)), features.get("chirp_linearity", 0.0))
    if modulation != "Chirp":
        return 0.0
    bw_match = 1.0 - min(1.0, abs(bw - 125_000.0) / 300_000.0)
    return (0.4 + 0.6 * max(0.0, bw_match)) * min(1.0, chirp_strength / 1.0)


def _dmr_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    if_peaks = features.get("if_peak_count", 0.0)
    if modulation not in {"FSK", "MFSK", "OOK"}:
        return 0.0
    peak_score = min(1.0, if_peaks / 4.0)
    partial_score = 0.0
    if channel == "Narrowband" and nature == "Digital":
        partial_score = 0.35 + 0.35 * min(1.0, features.get("cyclic_strength", 0.0) / 0.6)
    return min(1.0, 0.65 * peak_score + 0.35 * partial_score)


def _rc_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    duty = features.get("duty_cycle", 0.0)
    rate = features.get("packet_rate_hz", 0.0)
    if channel == "Narrowband" and duty < 0.4:
        return min(1.0, 0.4 + 0.6 * min(1.0, rate / 50.0))
    return 0.0


def _drone_detector(features: Dict[str, float], channel: str, modulation: str, nature: str) -> float:
    duty = features.get("duty_cycle", 0.0)
    rate = features.get("packet_rate_hz", 0.0)
    if nature == "Digital" and duty > 0.5:
        return min(1.0, 0.4 + 0.6 * min(1.0, rate / 100.0))
    return 0.0


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
    reg.register("Unknown Digital Signal", _unknown_digital_detector)
    reg.register("Unknown Narrowband Digital Signal", _unknown_narrowband_digital_detector)
    reg.register("Analog Broadcast", _analog_broadcast_detector)
    reg.register("Unknown Signal", _unknown_signal_detector)
    return reg


_DEFAULT_REGISTRY = build_default_registry()


def detect_protocols(features: Dict[str, float], channel: str, modulation: str, nature: str) -> Dict[str, float]:
    return _DEFAULT_REGISTRY.run(features, channel, modulation, nature)
