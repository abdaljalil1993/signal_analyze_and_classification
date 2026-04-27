from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ConstraintReport:
    penalties: Dict[str, float]
    rejections: List[str]
    reasoning: List[str]


def apply_physics_constraints(
    channel: str,
    modulation_scores: Dict[str, float],
    protocol_scores: Dict[str, float],
    features: Dict[str, float],
) -> ConstraintReport:
    penalties: Dict[str, float] = {}
    rejections: List[str] = []
    reasoning: List[str] = []

    if channel == "Narrowband" and modulation_scores.get("OFDM", 0.0) > 0.0:
        penalties["OFDM"] = penalties.get("OFDM", 0.0) + 0.95
        reasoning.append("Narrowband channel rejects OFDM.")

    if features.get("bandwidth_hz", 0.0) < 50_000.0 and modulation_scores.get("OFDM", 0.0) > 0.0:
        penalties["OFDM"] = max(penalties.get("OFDM", 0.0), 0.98)
        reasoning.append("Occupied bandwidth under 50 kHz strictly rejects OFDM.")

    if channel == "Narrowband" and protocol_scores.get("WiFi-like", 0.0) > 0.0:
        penalties["WiFi-like"] = penalties.get("WiFi-like", 0.0) + 0.95
        reasoning.append("Narrowband channel rejects WiFi-like protocol.")

    if features.get("bandwidth_hz", 0.0) < 50_000.0 and protocol_scores.get("WiFi-like", 0.0) > 0.0:
        penalties["WiFi-like"] = max(penalties.get("WiFi-like", 0.0), 0.98)
        reasoning.append("Occupied bandwidth under 50 kHz strictly rejects WiFi-like protocol.")

    if features.get("chirp_linearity", 0.0) <= 0.35 and features.get("fhss_detected", 0.0) <= 0.5:
        penalties["Spread"] = max(penalties.get("Spread", 0.0), 0.98)
        reasoning.append("Spread-spectrum channel requires chirp or FHSS evidence.")

    chirp_present = features.get("chirp_linearity", 0.0) > 0.35
    if not chirp_present and protocol_scores.get("LoRa-like", 0.0) > 0.0:
        penalties["LoRa-like"] = penalties.get("LoRa-like", 0.0) + 0.95
        reasoning.append("LoRa rejected due to missing chirp signature.")

    if features.get("if_peak_count", 0.0) < 2.0 and modulation_scores.get("FSK", 0.0) > 0.0:
        penalties["FSK"] = max(0.98, penalties.get("FSK", 0.0) + 0.85)
        reasoning.append("FSK penalized due to absent discrete IF peaks.")

    low_constellation = features.get("constellation_stability", 0.0) < 0.5
    if low_constellation:
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.75
        penalties["PSK"] = penalties.get("PSK", 0.0) + 0.75
        reasoning.append("Low constellation stability penalizes QAM and PSK.")

    if features.get("if_peak_count", 0.0) < 2.0 and modulation_scores.get("MFSK", 0.0) > 0.0:
        penalties["MFSK"] = penalties.get("MFSK", 0.0) + 0.8
        reasoning.append("MFSK penalized due to weak multi-peak IF evidence.")

    analog_continuous = (
        features.get("duty_cycle", 0.0) > 0.75
        and features.get("burstiness", 0.0) < 0.12
        and features.get("packet_rate_hz", 0.0) < 10.0
    )
    if analog_continuous:
        penalties["DMR-like"] = penalties.get("DMR-like", 0.0) + 0.8
        penalties["RC-like"] = penalties.get("RC-like", 0.0) + 0.65
        penalties["IoT-like"] = penalties.get("IoT-like", 0.0) + 0.65
        reasoning.append("Continuous analog-like envelope penalizes narrowband digital protocol hypotheses.")

    fm_analog_signature = (
        features.get("if_std", 0.0) > 8_000.0
        and features.get("if_peak_count", 0.0) <= 2.5
        and features.get("burstiness", 0.0) < 0.2
        and features.get("packet_rate_hz", 0.0) < 8.0
        and features.get("if_peak_stability", 0.0) < 0.55
    )
    if fm_analog_signature:
        penalties["DMR-like"] = penalties.get("DMR-like", 0.0) + 0.9
        penalties["RC-like"] = penalties.get("RC-like", 0.0) + 0.55
        penalties["IoT-like"] = penalties.get("IoT-like", 0.0) + 0.55
        reasoning.append("FM-like continuous signature rejects DMR/RC/IoT digital protocols.")

    noise_signature = (
        features.get("spectral_flatness", 0.0) > 0.72
        and features.get("spectral_entropy", 0.0) > 9.0
        and features.get("cyclic_autocorr_peak", 0.0) < 1.8
        and features.get("packet_rate_hz", 0.0) < 5.0
    )
    if noise_signature:
        penalties["DMR-like"] = penalties.get("DMR-like", 0.0) + 0.95
        penalties["RC-like"] = penalties.get("RC-like", 0.0) + 0.75
        penalties["IoT-like"] = penalties.get("IoT-like", 0.0) + 0.75
        reasoning.append("Noise-like spectral signature rejects narrowband digital protocol claims.")

    if features.get("if_peak_valid", 1.0) < 0.5 or features.get("if_reliability", 0.0) < 0.5:
        penalties["FSK"] = penalties.get("FSK", 0.0) + 0.45
        penalties["MFSK"] = penalties.get("MFSK", 0.0) + 0.45
        penalties["PSK"] = penalties.get("PSK", 0.0) + 0.35
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.35
        penalties["OFDM"] = penalties.get("OFDM", 0.0) + 0.35
        penalties["OOK"] = penalties.get("OOK", 0.0) + 0.35
        reasoning.append("Missing/unreliable IF evidence lowers modulation confidence across candidates.")

    if features.get("symbol_rate_est_hz", 0.0) < 100.0:
        penalties["FSK"] = penalties.get("FSK", 0.0) + 0.35
        penalties["MFSK"] = penalties.get("MFSK", 0.0) + 0.35
        penalties["PSK"] = penalties.get("PSK", 0.0) + 0.35
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.35
        penalties["OFDM"] = penalties.get("OFDM", 0.0) + 0.35
        reasoning.append("Invalid/low symbol rate reduces digital modulation confidence.")

    snr_db = features.get("snr_db", 0.0)
    if snr_db < 6.0 and modulation_scores.get("QAM", 0.0) > 0.0:
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.65
        reasoning.append("Low SNR penalizes QAM confidence.")

    symbol_rate = features.get("symbol_rate_est_hz", 0.0)
    if channel == "Narrowband" and 0.0 < symbol_rate < 20_000.0 and modulation_scores.get("QAM", 0.0) > 0.0:
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.7
        reasoning.append("Narrowband low-symbol-rate regime penalizes QAM.")

    if features.get("if_peak_count", 0.0) >= 2.0 and features.get("constellation_stability", 0.0) < 1.4 and modulation_scores.get("QAM", 0.0) > 0.0:
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.75
        reasoning.append("IF peak structure with unstable constellation favors FSK-like over QAM.")

    bw_hz = features.get("bandwidth_hz", 0.0)
    cyclic = features.get("cyclic_strength", 0.0)
    if channel == "Narrowband" and 10_000.0 <= bw_hz <= 20_000.0 and 0.2 <= cyclic <= 0.75:
        if protocol_scores.get("DMR-like", 0.0) > 0.0:
            penalties["DMR-like"] = max(0.0, penalties.get("DMR-like", 0.0) - 0.15)
        if protocol_scores.get("IoT-like", 0.0) > 0.0 and features.get("cyclic_autocorr_peak", 0.0) > 2.5:
            penalties["IoT-like"] = penalties.get("IoT-like", 0.0) + 0.35
        reasoning.append("Narrowband moderate-cyclic signature is consistent with DMR-like FSK behavior.")

    if features.get("cyclic_strength", 0.0) > 0.25:
        penalties["Noise"] = penalties.get("Noise", 0.0) + 0.98
        reasoning.append("Digital cyclic structure rejects Noise label.")

    if features.get("feature_validity_score", 1.0) < 0.6:
        reasoning.append("Some extracted features violated physical limits and were clamped.")

    if features.get("core_feature_validity_score", 1.0) < 0.7:
        penalties["Analog Broadcast"] = penalties.get("Analog Broadcast", 0.0) + 0.7
        penalties["Analog FM"] = penalties.get("Analog FM", 0.0) + 0.7
        penalties["DMR-like"] = penalties.get("DMR-like", 0.0) + 0.45
        penalties["IoT-like"] = penalties.get("IoT-like", 0.0) + 0.35
        penalties["WiFi-like"] = penalties.get("WiFi-like", 0.0) + 0.45
        reasoning.append("Core physical feature validity is low; specific protocol confidence reduced.")

    if features.get("cyclic_strength", 0.0) > 0.35:
        reasoning.append("Strong cyclic behavior indicates digital nature.")

    for label, p in penalties.items():
        if p >= 0.9:
            rejections.append(label)

    return ConstraintReport(penalties=penalties, rejections=rejections, reasoning=reasoning)


def enforce_consistency(
    nature: str,
    channel: str,
    modulation: str,
    protocol: str,
) -> Tuple[bool, str]:
    if nature == "Analog" and modulation in {"OFDM", "QAM", "PSK"}:
        return False, "Analog nature inconsistent with high-order digital modulation."
    if channel == "Narrowband" and protocol == "WiFi-like":
        return False, "WiFi requires wideband operation."
    if channel == "Spread" and protocol not in {"LoRa-like", "IoT-like", "Unknown Digital Signal", "Unknown Signal"}:
        return False, "Spread channel must map to chirp/FHSS-capable protocols."
    if modulation == "Chirp" and protocol not in {"LoRa-like", "Unknown Digital Signal"}:
        return False, "Chirp modulation without LoRa/IoT is inconsistent."
    if nature == "Digital" and protocol in {"Analog Broadcast", "Analog FM"}:
        return False, "Digital nature conflicts with analog broadcast protocol."
    if modulation == "OFDM" and channel == "Narrowband":
        return False, "OFDM cannot be narrowband in this physical model."
    if channel == "Narrowband" and nature == "Digital" and protocol in {"Analog Broadcast", "Analog FM"}:
        return False, "Narrowband digital channel cannot be analog broadcast."
    if nature == "Digital" and modulation in {"AM", "FM"}:
        return False, "Digital nature cannot map to AM/FM modulation."
    return True, "Combination is physically consistent."
