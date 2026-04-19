from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.config import EPS
from utils.types import StageDecision


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores.keys())
    vals = np.array([max(0.0, scores[k]) for k in keys], dtype=np.float64)
    s = float(np.sum(vals) + EPS)
    vals = vals / s
    return {k: float(v) for k, v in zip(keys, vals)}


class HierarchicalClassifier:
    def classify(self, features: Dict[str, float], protocol_scores: Dict[str, float]) -> List[StageDecision]:
        decisions: List[StageDecision] = []

        stage1 = self._stage_signal_nature(features)
        decisions.append(stage1)

        stage2 = self._stage_channel(features, stage1.selected)
        decisions.append(stage2)

        stage3 = self._stage_modulation(features, stage1.selected, stage2.selected)
        decisions.append(stage3)

        stage4 = self._stage_protocol(features, stage2.selected, stage3.selected, protocol_scores)
        decisions.append(stage4)

        stage5 = self._stage_application(features, stage1.selected, stage4.selected)
        decisions.append(stage5)

        return decisions

    def _stage_signal_nature(self, f: Dict[str, float]) -> StageDecision:
        cyclic = min(1.0, f.get("cyclic_strength", 0.0))
        packet_struct = min(1.0, f.get("burstiness", 0.0) + min(1.0, f.get("packet_rate_hz", 0.0) / 150.0))
        digital_structure = min(1.0, cyclic + min(1.0, f.get("cyclic_autocorr_peak", 0.0) / 6.0) + packet_struct)
        flatness = min(1.0, f.get("spectral_flatness", 0.5))
        entropy = min(1.0, f.get("spectral_entropy", 0.0) / 12.0)

        digital = 0.15 + 0.5 * cyclic + 0.35 * packet_struct + 0.3 * min(1.0, f.get("cyclic_autocorr_peak", 0.0) / 6.0)
        analog = 0.2 + 0.5 * (1.0 - flatness) + 0.3 * min(1.0, f.get("amplitude_stability", 0.0) / 10.0)
        noise = 0.15 + 0.65 * entropy + 0.25 * flatness

        # Hard anti-noise guards when digital structure is present.
        if cyclic > 0.2:
            noise *= 0.15
        if packet_struct > 0.2:
            noise *= 0.2
        if digital_structure > 0.35:
            analog *= 0.25
        if f.get("core_feature_validity_score", 1.0) < 0.5:
            analog *= 0.5

        scores = _normalize_scores({"Noise": noise, "Analog": analog, "Digital": digital})
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Signal Nature", scores=scores, selected=selected)

    def _stage_channel(self, f: Dict[str, float], nature: str) -> StageDecision:
        bw = f.get("bandwidth_hz", 0.0)
        narrow = 1.0 if bw < 100_000.0 else max(0.0, 1.0 - (bw - 100_000.0) / 200_000.0)
        wide = 1.0 if bw > 1_000_000.0 else max(0.0, (bw - 300_000.0) / 700_000.0)

        chirp_confirmed = f.get("chirp_linearity", 0.0) > 0.35
        dsss_confirmed = f.get("dsss_confirmed", 0.0) > 0.5
        spread_confirmed = chirp_confirmed or dsss_confirmed
        spread = max(0.0, 0.2 + 0.8 * max(f.get("chirp_linearity", 0.0), f.get("dsss_confirmed", 0.0))) if spread_confirmed else 0.0

        if nature == "Noise":
            spread *= 0.6

        scores = _normalize_scores({"Narrowband": narrow, "Wideband": wide, "Spread": spread})
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Channel", scores=scores, selected=selected)

    def _stage_modulation(self, f: Dict[str, float], nature: str, channel: str) -> StageDecision:
        if_std = f.get("if_std", 0.0)
        if_peaks = f.get("if_peak_count", 0.0)
        cyclic = f.get("cyclic_strength", 0.0)
        const_stab = f.get("constellation_stability", 0.0)
        subcarrier = f.get("subcarrier_structure", 0.0)
        chirp_lin = f.get("chirp_linearity", 0.0)
        ook_score = f.get("ook_score", 0.0)
        m_fsk = f.get("multi_fsk_score", 0.0)

        am = max(0.0, 1.0 - min(1.0, if_std / 18_000.0)) * (0.7 + 0.3 * min(1.0, f.get("amplitude_stability", 0.0) / 8.0))
        fm = max(0.0, min(1.0, if_std / 20_000.0)) * max(0.2, 1.0 - min(1.0, f.get("ook_score", 0.0)))
        fsk = max(0.0, min(1.0, if_peaks / 5.0))
        psk = max(0.0, min(1.0, const_stab / 3.0)) * max(0.0, min(1.0, cyclic))
        qam = max(0.0, min(1.0, const_stab / 4.0))
        ofdm = max(0.0, min(1.0, f.get("bandwidth_ratio_to_fs", 0.0) * 2.0)) * max(0.0, cyclic) * (0.6 + 0.4 * subcarrier)
        chirp = max(0.0, min(1.0, 0.5 * abs(f.get("if_skew", 0.0)) + 0.8 * chirp_lin))
        ook = max(0.0, min(1.0, ook_score))
        mfsk = max(0.0, min(1.0, m_fsk))

        if nature == "Analog":
            psk *= 0.3
            qam *= 0.2
            ofdm *= 0.2
        if nature == "Digital":
            am = 0.0
            fm = 0.0
        if chirp_lin > 0.45:
            fsk *= 0.35
            mfsk *= 0.4
            chirp = min(1.0, chirp + 0.25)
        if channel == "Narrowband":
            ofdm *= 0.1

        scores = _normalize_scores(
            {
                "AM": am,
                "FM": fm,
                "FSK": fsk,
                "PSK": psk,
                "QAM": qam,
                "OFDM": ofdm,
                "Chirp": chirp,
                "OOK": ook,
                "MFSK": mfsk,
            }
        )
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Modulation", scores=scores, selected=selected)

    def _stage_protocol(
        self,
        f: Dict[str, float],
        channel: str,
        modulation: str,
        protocol_scores: Dict[str, float],
    ) -> StageDecision:
        base = {
            "WiFi-like": 0.1,
            "LoRa-like": 0.1,
            "DMR-like": 0.1,
            "RC-like": 0.1,
            "Drone-link-like": 0.1,
            "Analog Broadcast": 0.1,
            "Unknown Digital Signal": 0.12,
            "Unknown Narrowband Digital Signal": 0.14,
            "Unknown Signal": 0.15,
        }
        for k in base:
            base[k] += protocol_scores.get(k, 0.0)

        if channel == "Wideband" and modulation == "OFDM":
            base["WiFi-like"] += 0.35
        if modulation == "Chirp":
            base["LoRa-like"] += 0.35
        if modulation in {"FSK", "MFSK"}:
            base["DMR-like"] += 0.22
            base["Unknown Narrowband Digital Signal"] += 0.25
        if channel == "Narrowband" and modulation in {"OOK", "FSK", "MFSK"}:
            base["RC-like"] += 0.18
        if modulation in {"PSK", "QAM", "OFDM"}:
            base["Unknown Digital Signal"] += 0.2
        if channel == "Narrowband" and modulation in {"FSK", "MFSK", "OOK", "PSK"}:
            base["Unknown Narrowband Digital Signal"] += 0.2
        if modulation in {"AM", "FM"} and f.get("cyclic_strength", 0.0) < 0.2:
            base["Analog Broadcast"] += 0.3
        if f.get("core_feature_validity_score", 1.0) < 0.7:
            base["Unknown Signal"] += 0.35
            base["Analog Broadcast"] *= 0.5

        scores = _normalize_scores(base)
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Protocol", scores=scores, selected=selected)

    def _stage_application(self, f: Dict[str, float], nature: str, protocol: str) -> StageDecision:
        voice = 0.2
        data = 0.2 + 0.4 * min(1.0, f.get("packet_rate_hz", 0.0) / 80.0)
        control = 0.2 + 0.5 * max(0.0, 0.6 - f.get("duty_cycle", 0.0))

        if protocol in {"DMR-like", "RC-like", "Drone-link-like", "Unknown Narrowband Digital Signal"}:
            control += 0.2
        if protocol in {"WiFi-like", "Unknown Digital Signal"}:
            data += 0.25
        if protocol == "Analog Broadcast":
            voice += 0.25
        if nature == "Analog":
            voice += 0.4

        scores = _normalize_scores({"Voice": voice, "Data": data, "Control": control})
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Application", scores=scores, selected=selected)
