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

    if channel == "Narrowband" and protocol_scores.get("WiFi-like", 0.0) > 0.0:
        penalties["WiFi-like"] = penalties.get("WiFi-like", 0.0) + 0.95
        reasoning.append("Narrowband channel rejects WiFi-like protocol.")

    chirp_present = abs(features.get("if_skew", 0.0)) > 0.5
    if not chirp_present and protocol_scores.get("LoRa-like", 0.0) > 0.0:
        penalties["LoRa-like"] = penalties.get("LoRa-like", 0.0) + 0.95
        reasoning.append("LoRa rejected due to missing chirp signature.")

    if features.get("if_peak_count", 0.0) < 2.0 and modulation_scores.get("FSK", 0.0) > 0.0:
        penalties["FSK"] = penalties.get("FSK", 0.0) + 0.85
        reasoning.append("FSK penalized due to absent discrete IF peaks.")

    low_constellation = features.get("constellation_stability", 0.0) < 0.5
    if low_constellation:
        penalties["QAM"] = penalties.get("QAM", 0.0) + 0.75
        penalties["PSK"] = penalties.get("PSK", 0.0) + 0.75
        reasoning.append("Low constellation stability penalizes QAM and PSK.")

    if features.get("if_peak_count", 0.0) < 2.0 and modulation_scores.get("MFSK", 0.0) > 0.0:
        penalties["MFSK"] = penalties.get("MFSK", 0.0) + 0.8
        reasoning.append("MFSK penalized due to weak multi-peak IF evidence.")

    if features.get("cyclic_strength", 0.0) > 0.25:
        penalties["Noise"] = penalties.get("Noise", 0.0) + 0.98
        reasoning.append("Digital cyclic structure rejects Noise label.")

    if features.get("feature_validity_score", 1.0) < 0.6:
        reasoning.append("Some extracted features violated physical limits and were clamped.")

    if features.get("core_feature_validity_score", 1.0) < 0.7:
        penalties["Analog Broadcast"] = penalties.get("Analog Broadcast", 0.0) + 0.7
        penalties["DMR-like"] = penalties.get("DMR-like", 0.0) + 0.45
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
    if modulation == "Chirp" and protocol not in {"LoRa-like", "Unknown Digital Signal"}:
        return False, "Chirp modulation without LoRa/IoT is inconsistent."
    if nature == "Digital" and protocol == "Analog Broadcast":
        return False, "Digital nature conflicts with analog broadcast protocol."
    if modulation == "OFDM" and channel == "Narrowband":
        return False, "OFDM cannot be narrowband in this physical model."
    if channel == "Narrowband" and nature == "Digital" and protocol == "Analog Broadcast":
        return False, "Narrowband digital channel cannot be analog broadcast."
    if nature == "Digital" and modulation in {"AM", "FM"}:
        return False, "Digital nature cannot map to AM/FM modulation."
    return True, "Combination is physically consistent."
