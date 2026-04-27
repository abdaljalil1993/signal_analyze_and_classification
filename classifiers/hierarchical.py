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
        duty = min(1.0, max(0.0, f.get("duty_cycle", 0.0)))
        burstiness = min(1.0, max(0.0, f.get("burstiness", 0.0)))
        packet_rate = max(0.0, f.get("packet_rate_hz", 0.0))
        if_peaks = max(0.0, f.get("if_peak_count", 0.0))
        if_std = max(0.0, f.get("if_std", 0.0))
        if_peak_stability = min(1.0, max(0.0, f.get("if_peak_stability", 0.0)))

        continuous_analog = min(1.0, 0.55 * duty + 0.3 * (1.0 - burstiness) + 0.15 * max(0.0, 1.0 - min(1.0, packet_rate / 20.0)))
        fm_like = min(1.0, if_std / 25_000.0) * max(0.0, 1.0 - min(1.0, if_peaks / 3.0))
        am_like = max(0.0, 1.0 - min(1.0, if_std / 5_000.0)) * max(0.0, 1.0 - min(1.0, if_peaks / 2.0))
        fm_analog_signature = (
            if_std > 8_000.0
            and if_peaks <= 2.5
            and burstiness < 0.2
            and packet_rate < 8.0
            and if_peak_stability < 0.55
        )
        noise_signature = (
            flatness > 0.72
            and entropy > 0.75
            and f.get("cyclic_autocorr_peak", 0.0) < 1.8
            and packet_rate < 5.0
            and burstiness < 0.22
        )

        digital = 0.15 + 0.5 * cyclic + 0.35 * packet_struct + 0.3 * min(1.0, f.get("cyclic_autocorr_peak", 0.0) / 6.0)
        analog = 0.2 + 0.5 * (1.0 - flatness) + 0.3 * min(1.0, f.get("amplitude_stability", 0.0) / 10.0) + 0.35 * continuous_analog + 0.2 * max(fm_like, am_like)
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

        # Strong continuous carrier evidence should not be interpreted as packetized digital.
        if continuous_analog > 0.7 and packet_rate < 10.0 and if_peaks <= 1.5:
            digital *= 0.45
            analog = min(1.5, analog * 1.4)
        if fm_analog_signature:
            digital *= 0.38
            analog = min(1.6, analog * 1.5)
        if noise_signature:
            noise = min(1.8, noise * 1.8)
            digital *= 0.35
            analog *= 0.7

        scores = _normalize_scores({"Noise": noise, "Analog": analog, "Digital": digital})
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Signal Nature", scores=scores, selected=selected)

    def _stage_channel(self, f: Dict[str, float], nature: str) -> StageDecision:
        bw = f.get("bandwidth_hz", 0.0)
        narrow = 1.0 if bw < 100_000.0 else max(0.0, 1.0 - (bw - 100_000.0) / 200_000.0)
        wide = 1.0 if bw > 1_000_000.0 else max(0.0, (bw - 300_000.0) / 700_000.0)

        chirp_confirmed = f.get("chirp_linearity", 0.0) > 0.35
        fhss_confirmed = f.get("fhss_detected", 0.0) > 0.5
        spread_confirmed = chirp_confirmed or fhss_confirmed
        spread = max(0.0, 0.2 + 0.8 * max(f.get("chirp_linearity", 0.0), f.get("fhss_detected", 0.0))) if spread_confirmed else 0.0

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
        snr_db = f.get("snr_db", 0.0)
        bw_hz = f.get("bandwidth_hz", 0.0)
        symbol_rate_hz = f.get("symbol_rate_est_hz", 0.0)
        subcarrier = f.get("subcarrier_structure", 0.0)
        chirp_lin = f.get("chirp_linearity", 0.0)
        ook_score = f.get("ook_score", 0.0)
        m_fsk = f.get("multi_fsk_score", 0.0)
        if_reliability = f.get("if_reliability", 0.0)
        if_peak_valid = f.get("if_peak_valid", 1.0)
        if_variance_low = if_std < 1_500.0
        analog_continuous = f.get("duty_cycle", 0.0) > 0.75 and f.get("burstiness", 0.0) < 0.12 and f.get("packet_rate_hz", 0.0) < 10.0
        analog_fm_hint = (
            if_std > 8_000.0
            and if_peaks <= 2.5
            and f.get("burstiness", 0.0) < 0.2
            and f.get("packet_rate_hz", 0.0) < 8.0
            and f.get("if_peak_stability", 0.0) < 0.55
        )

        if_peak_norm = max(0.0, min(1.0, if_peaks / 4.0))
        if_var_norm = max(0.0, min(1.0, if_std / 12_000.0))
        if_cluster = max(0.0, min(1.0, 0.55 * if_peak_norm + 0.45 * m_fsk))
        snr_gate = max(0.0, min(1.0, (snr_db - 2.0) / 16.0))

        am = max(0.0, 1.0 - min(1.0, if_std / 18_000.0)) * (0.7 + 0.3 * min(1.0, f.get("amplitude_stability", 0.0) / 8.0))
        fm = max(0.0, min(1.0, if_std / 20_000.0)) * max(0.2, 1.0 - min(1.0, f.get("ook_score", 0.0)))
        fsk = max(0.0, min(1.0, 0.5 * if_peak_norm + 0.3 * if_var_norm + 0.2 * if_cluster))
        psk = max(0.0, min(1.0, const_stab / 3.0)) * max(0.0, min(1.0, cyclic))
        qam = max(0.0, min(1.0, const_stab / 4.0)) * (0.2 + 0.8 * snr_gate) * (1.0 - 0.45 * if_cluster)
        ofdm = max(0.0, min(1.0, f.get("bandwidth_ratio_to_fs", 0.0) * 2.0)) * max(0.0, cyclic) * (0.6 + 0.4 * subcarrier)
        chirp = max(0.0, min(1.0, 0.5 * abs(f.get("if_skew", 0.0)) + 0.8 * chirp_lin))
        if chirp_lin < 0.35:
            chirp = 0.0
        ook = max(0.0, min(1.0, ook_score))
        if not (
            f.get("envelope_bimodality", 0.0) >= 0.55
            and f.get("power_hist_bimodal", 0.0) > 0.5
            and if_variance_low
        ):
            ook = 0.0
        mfsk = max(0.0, min(1.0, 0.65 * m_fsk + 0.35 * if_peak_norm))

        if nature == "Analog":
            fsk *= 0.15
            mfsk *= 0.12
            psk *= 0.3
            qam *= 0.2
            ofdm *= 0.2
            chirp *= 0.35
            am = min(1.0, am * 1.25)
            fm = min(1.0, fm * 1.25)
        if nature == "Digital":
            am = 0.0
            fm = 0.0
        if chirp_lin > 0.45:
            fsk *= 0.35
            mfsk *= 0.4
            chirp = min(1.0, chirp + 0.25)
        if channel == "Narrowband":
            ofdm *= 0.1

        if analog_continuous:
            am = min(1.0, am * 1.3)
            fm = min(1.0, fm * 1.35)
            fsk *= 0.35
            mfsk *= 0.3
            psk *= 0.4
            qam *= 0.3

        if analog_fm_hint:
            fm = min(1.0, fm * 1.55)
            am = min(1.0, am * 1.2)
            fsk *= 0.22
            mfsk *= 0.2
            psk *= 0.4
            qam *= 0.35

        # Constellation stability alone should not dominate under low SNR.
        if snr_db < 6.0:
            qam *= 0.25

        # Multi-peak IF evidence should favor FSK-like modulation families.
        if if_peaks >= 2.0:
            fsk = min(1.0, fsk * 1.35)
            mfsk = min(1.0, mfsk * 1.3)
            qam *= 0.6
        else:
            fsk *= 0.25
            mfsk *= 0.25

        if f.get("if_hist_valid", 1.0) < 0.5:
            fsk *= 0.35
            mfsk *= 0.35

        if if_peaks >= 3.5 and if_peaks <= 4.5:
            fsk = min(1.0, fsk * 1.2)
            mfsk = min(1.0, mfsk * 1.3)

        # Conflict rule: IF peaks with unstable constellation prefer FSK over QAM.
        if if_peaks >= 2.0 and const_stab < 1.4:
            fsk = min(1.0, fsk * 1.45)
            mfsk = min(1.0, mfsk * 1.35)
            qam *= 0.2

        if channel == "Narrowband" and 0.0 < symbol_rate_hz < 20_000.0:
            qam *= 0.2

        if symbol_rate_hz < 100.0:
            psk *= 0.55
            qam *= 0.55
            ofdm *= 0.5
            fsk *= 0.6
            mfsk *= 0.6
            chirp *= 0.5
            ook *= 0.7

        if if_peak_valid < 0.5 or if_reliability < 0.5:
            am *= 0.65
            fm *= 0.65
            fsk *= 0.45
            psk *= 0.6
            qam *= 0.6
            ofdm *= 0.6
            chirp *= 0.5
            ook *= 0.5
            mfsk *= 0.45

            # Missing/unreliable IF evidence flattens modulation evidence to avoid hard fallback labels.
            vals = np.array([am, fm, fsk, psk, qam, ofdm, chirp, ook, mfsk], dtype=np.float64)
            mean_v = float(np.mean(vals))
            vals = 0.55 * vals + 0.45 * mean_v
            am, fm, fsk, psk, qam, ofdm, chirp, ook, mfsk = [float(v) for v in vals]

        dmr_like_hint = (
            channel == "Narrowband"
            and nature == "Digital"
            and 10_000.0 <= bw_hz <= 20_000.0
            and 0.2 <= cyclic <= 0.75
        )
        if dmr_like_hint:
            fsk = min(1.0, fsk + 0.18)
            mfsk = min(1.0, mfsk + 0.14)
            qam *= 0.35
            psk *= 0.8

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
            "IoT-like": 0.1,
            "Analog FM": 0.1,
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
        bw_hz = f.get("bandwidth_hz", 0.0)
        cyclic = f.get("cyclic_strength", 0.0)
        symbol_rate_hz = f.get("symbol_rate_est_hz", 0.0)
        if_peaks = f.get("if_peak_count", 0.0)
        cyc_ac = f.get("cyclic_autocorr_peak", 0.0)
        dmr_proto_hint = (
            6_000.0 <= bw_hz <= 20_000.0
            and if_peaks >= 2.0
            and cyc_ac >= 2.3
            and (symbol_rate_hz <= 0.0 or 2_000.0 <= symbol_rate_hz <= 8_000.0)
        )

        if modulation in {"FSK", "MFSK"}:
            if dmr_proto_hint:
                base["DMR-like"] += 0.24
            else:
                base["DMR-like"] += 0.03
            base["Unknown Narrowband Digital Signal"] += 0.2
        if channel == "Narrowband" and modulation in {"OOK", "FSK", "MFSK"}:
            base["RC-like"] += 0.18
        if modulation in {"PSK", "QAM", "OFDM"}:
            base["Unknown Digital Signal"] += 0.2
        if channel == "Narrowband" and modulation in {"FSK", "MFSK", "OOK", "PSK"}:
            base["Unknown Narrowband Digital Signal"] += 0.2
        dmr_like_hint = channel == "Narrowband" and 10_000.0 <= bw_hz <= 20_000.0 and 0.2 <= cyclic <= 0.75
        if dmr_like_hint:
            base["DMR-like"] += 0.12
            base["Unknown Narrowband Digital Signal"] += 0.2
        if dmr_proto_hint and cyc_ac >= 6.0:
            base["DMR-like"] += 0.22
            base["Unknown Narrowband Digital Signal"] *= 0.7
        if channel == "Narrowband" and modulation == "QAM" and 0.0 < symbol_rate_hz < 20_000.0:
            base["Unknown Narrowband Digital Signal"] += 0.2
            base["DMR-like"] += 0.1
        if modulation in {"AM", "FM"} and f.get("cyclic_strength", 0.0) < 0.2:
            base["Analog Broadcast"] += 0.3
            base["Analog FM"] += 0.35 if modulation == "FM" else 0.0
        if f.get("core_feature_validity_score", 1.0) < 0.7:
            base["Unknown Signal"] += 0.35
            base["Analog Broadcast"] *= 0.5

        known_protocols = [
            "WiFi-like",
            "LoRa-like",
            "DMR-like",
            "RC-like",
            "Drone-link-like",
            "IoT-like",
            "Analog FM",
            "Analog Broadcast",
        ]
        strongest_known = max((protocol_scores.get(k, 0.0) for k in known_protocols), default=0.0)
        if strongest_known >= 0.6:
            base["Unknown Digital Signal"] *= 0.15
            base["Unknown Narrowband Digital Signal"] *= 0.15
            base["Unknown Signal"] *= 0.35
            for k in known_protocols:
                if protocol_scores.get(k, 0.0) >= 0.6:
                    base[k] += 0.15 * protocol_scores.get(k, 0.0)
        elif strongest_known < 0.25:
            base["Unknown Digital Signal"] += 0.18
            base["Unknown Narrowband Digital Signal"] += 0.18

        scores = _normalize_scores(base)
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Protocol", scores=scores, selected=selected)

    def _stage_application(self, f: Dict[str, float], nature: str, protocol: str) -> StageDecision:
        voice = 0.2
        data = 0.2 + 0.4 * min(1.0, f.get("packet_rate_hz", 0.0) / 80.0)
        control = 0.2 + 0.5 * max(0.0, 0.6 - f.get("duty_cycle", 0.0))

        if protocol in {"DMR-like", "RC-like", "Drone-link-like", "IoT-like", "Unknown Narrowband Digital Signal"}:
            control += 0.2
        if protocol in {"WiFi-like", "IoT-like", "Unknown Digital Signal"}:
            data += 0.25
        if protocol in {"Analog Broadcast", "Analog FM"}:
            voice += 0.25
        if nature == "Analog":
            voice += 0.4

        scores = _normalize_scores({"Voice": voice, "Data": data, "Control": control})
        selected = max(scores, key=scores.get)
        return StageDecision(stage="Application", scores=scores, selected=selected)
