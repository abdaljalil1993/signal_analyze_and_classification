from __future__ import annotations

from itertools import product
from typing import Dict, List

from constraint_engine.physics_rules import ConstraintReport, enforce_consistency
from utils.config import EPS
from utils.types import Candidate, PipelineResult, StageDecision


class DecisionEngine:
    def build_result(
        self,
        stage_trace: List[StageDecision],
        features: Dict[str, float],
        constraints: ConstraintReport,
    ) -> PipelineResult:
        by_stage = {s.stage: s for s in stage_trace}
        nature_scores = by_stage["Signal Nature"].scores
        channel_scores = by_stage["Channel"].scores
        modulation_scores = by_stage["Modulation"].scores
        protocol_scores = by_stage["Protocol"].scores
        app_scores = by_stage["Application"].scores

        nature_scores = self._apply_penalties(nature_scores, constraints.penalties)
        channel_scores = self._apply_penalties(channel_scores, constraints.penalties)
        modulation_scores = self._apply_penalties(modulation_scores, constraints.penalties)
        protocol_scores = self._apply_penalties(protocol_scores, constraints.penalties)
        app_scores = self._apply_penalties(app_scores, constraints.penalties)

        candidates: List[Candidate] = []
        for nature, channel, mod, proto, app in product(
            nature_scores.keys(),
            channel_scores.keys(),
            modulation_scores.keys(),
            protocol_scores.keys(),
            app_scores.keys(),
        ):
            ok, reason = enforce_consistency(nature, channel, mod, proto)
            if not ok:
                continue

            score = (
                nature_scores[nature]
                * channel_scores[channel]
                * modulation_scores[mod]
                * protocol_scores[proto]
                * app_scores[app]
            )
            if score <= 0.0:
                continue

            quality = self._quality_factor(features)
            agreement = self._feature_agreement(features, mod, proto)
            stage_consistency = self._stage_consistency(stage_trace)
            satisfaction = self._constraint_satisfaction(constraints)
            alignment = self._stage_alignment(by_stage, nature, channel, mod, proto, app)
            confidence = min(
                1.0,
                (score * 8.0) * 0.28
                + agreement * 0.2
                + stage_consistency * 0.16
                + satisfaction * 0.15
                + quality * 0.1
                + alignment * 0.11,
            )
            strong_signature = proto not in {"Unknown Digital Signal", "Unknown Narrowband Digital Signal", "Unknown Signal"} and protocol_scores.get(proto, 0.0) >= 0.6
            if strong_signature:
                confidence = min(1.0, confidence + 0.18 * protocol_scores.get(proto, 0.0))
            core_validity = max(0.0, min(1.0, features.get("core_feature_validity_score", 0.0)))
            confidence *= (0.1 + 0.9 * core_validity)

            reasoning = [
                f"Stage-consistent path: {nature} -> {channel} -> {mod} -> {proto} -> {app}.",
                reason,
            ]
            reasoning.extend(self._dominant_feature_reasons(features, mod, proto))
            reasoning.extend(constraints.reasoning)

            candidates.append(
                Candidate(
                    signal_type=nature,
                    channel_type=channel,
                    modulation=mod,
                    protocol=proto,
                    application=app,
                    score=score,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )

        if not candidates:
            candidates = [
                Candidate(
                    signal_type=by_stage["Signal Nature"].selected,
                    channel_type=by_stage["Channel"].selected,
                    modulation=by_stage["Modulation"].selected,
                    protocol=by_stage["Protocol"].selected,
                    application=by_stage["Application"].selected,
                    score=0.0,
                    confidence=0.0,
                    reasoning=["All candidates rejected by physical constraints."],
                )
            ]

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        best = candidates[0]
        strict_match = [
            c
            for c in candidates
            if c.signal_type == by_stage["Signal Nature"].selected
            and c.channel_type == by_stage["Channel"].selected
            and c.modulation == by_stage["Modulation"].selected
            and c.protocol == by_stage["Protocol"].selected
            and c.application == by_stage["Application"].selected
        ]
        if strict_match:
            strict_best = strict_match[0]
            # Keep strict stage path only when it is competitive; otherwise preserve highest-confidence candidate.
            if strict_best.confidence >= best.confidence * 0.95:
                best = strict_best
        reasoning = best.reasoning[:]
        if constraints.rejections:
            reasoning.append(f"Rejected labels: {', '.join(constraints.rejections)}")

        summary = {
            "bandwidth_hz": features.get("bandwidth_hz", 0.0),
            "cyclic_strength": features.get("cyclic_strength", 0.0),
            "cyclic_autocorr_peak": features.get("cyclic_autocorr_peak", 0.0),
            "if_peak_count": features.get("if_peak_count", 0.0),
            "constellation_stability": features.get("constellation_stability", 0.0),
            "snr_db": features.get("snr_db", 0.0),
            "duty_cycle": features.get("duty_cycle", 0.0),
            "packet_rate_hz": features.get("packet_rate_hz", 0.0),
            "symbol_rate_est_hz": features.get("symbol_rate_est_hz", 0.0),
            "feature_validity_score": features.get("feature_validity_score", 0.0),
            "core_feature_validity_score": features.get("core_feature_validity_score", 0.0),
            "core_feature_invalid_count": features.get("core_feature_invalid_count", 0.0),
        }

        return PipelineResult(
            signal_type=best.signal_type,
            channel_type=best.channel_type,
            modulation=best.modulation,
            protocol=best.protocol,
            application=best.application,
            confidence=best.confidence,
            top_candidates=candidates[:3],
            stage_trace=stage_trace,
            feature_summary=summary,
            reasoning=reasoning,
        )

    @staticmethod
    def _apply_penalties(scores: Dict[str, float], penalties: Dict[str, float]) -> Dict[str, float]:
        adjusted = {}
        for k, v in scores.items():
            p = penalties.get(k, 0.0)
            adjusted[k] = max(0.0, v * (1.0 - p))

        s = sum(adjusted.values()) + EPS
        return {k: v / s for k, v in adjusted.items()}

    @staticmethod
    def _constraint_satisfaction(constraints: ConstraintReport) -> float:
        total_penalty = sum(constraints.penalties.values())
        return max(0.0, 1.0 - 0.4 * total_penalty)

    @staticmethod
    def _quality_factor(features: Dict[str, float]) -> float:
        cyclic = min(1.0, max(0.0, features.get("cyclic_strength", 0.0)))
        flatness = min(1.0, max(0.0, features.get("spectral_flatness", 0.0)))
        stability = min(1.0, features.get("constellation_stability", 0.0) / 3.0)
        snr = min(1.0, max(0.0, (features.get("snr_db", 0.0) + 5.0) / 30.0))
        amp_stability = min(1.0, features.get("amplitude_stability", 0.0) / 8.0)
        validity = min(1.0, max(0.0, features.get("feature_validity_score", 0.0)))
        core_validity = min(1.0, max(0.0, features.get("core_feature_validity_score", 0.0)))
        physics_penalty = 1.0
        if features.get("bandwidth_ratio_to_fs", 0.0) > 1.0:
            physics_penalty *= 0.6
        if features.get("symbol_rate_est_hz", 0.0) <= 0.0:
            physics_penalty *= 0.8
        if core_validity < 0.7:
            physics_penalty *= 0.5
        return max(
            0.0,
            min(
                1.0,
                (0.26 * cyclic + 0.18 * stability + 0.1 * (1.0 - flatness) + 0.16 * snr + 0.08 * amp_stability + 0.1 * validity + 0.12 * core_validity)
                * physics_penalty,
            ),
        )

    @staticmethod
    def _feature_agreement(features: Dict[str, float], modulation: str, protocol: str) -> float:
        agreement = 0.2
        snr_db = features.get("snr_db", 0.0)
        snr_gate = min(1.0, max(0.0, (snr_db - 2.0) / 16.0))
        if_peak_count = features.get("if_peak_count", 0.0)
        if_peak_norm = min(1.0, max(0.0, if_peak_count / 4.0))
        if_var_norm = min(1.0, max(0.0, features.get("if_std", 0.0) / 12_000.0))
        if_cluster = min(1.0, max(0.0, 0.55 * if_peak_norm + 0.45 * features.get("multi_fsk_score", 0.0)))
        if modulation == "OFDM":
            agreement += min(0.4, features.get("bandwidth_hz", 0.0) / 2_000_000.0)
        if modulation in {"FSK", "MFSK"}:
            agreement += min(0.45, 0.28 * if_peak_norm + 0.17 * if_var_norm)
        if modulation in {"PSK", "QAM"}:
            const_term = min(0.4, features.get("constellation_stability", 0.0) / 4.0)
            conflict_penalty = 0.5 if if_peak_count >= 2.0 and features.get("constellation_stability", 0.0) < 1.4 else 0.0
            agreement += max(0.0, const_term * (0.25 + 0.75 * snr_gate) * (1.0 - conflict_penalty) * (1.0 - 0.35 * if_cluster))
        if protocol == "LoRa-like":
            agreement += min(0.3, abs(features.get("if_skew", 0.0)) / 2.0)
        if protocol == "Unknown Narrowband Digital Signal":
            agreement += 0.2 * min(1.0, features.get("cyclic_strength", 0.0))
        if protocol in {"DMR-like", "LoRa-like", "WiFi-like", "RC-like", "Drone-link-like", "IoT-like", "Analog FM", "Analog Broadcast"}:
            agreement += 0.12
        agreement += 0.15 * min(1.0, max(0.0, features.get("feature_validity_score", 0.0)))
        return max(0.0, min(1.0, agreement))

    @staticmethod
    def _stage_consistency(stage_trace: List[StageDecision]) -> float:
        max_scores = [max(s.scores.values()) for s in stage_trace if s.scores]
        if not max_scores:
            return 0.0
        return max(0.0, min(1.0, float(sum(max_scores) / len(max_scores))))

    @staticmethod
    def _stage_alignment(
        by_stage: Dict[str, StageDecision],
        nature: str,
        channel: str,
        mod: str,
        proto: str,
        app: str,
    ) -> float:
        matches = 0
        matches += 1 if nature == by_stage["Signal Nature"].selected else 0
        matches += 1 if channel == by_stage["Channel"].selected else 0
        matches += 1 if mod == by_stage["Modulation"].selected else 0
        matches += 1 if proto == by_stage["Protocol"].selected else 0
        matches += 1 if app == by_stage["Application"].selected else 0
        return matches / 5.0

    @staticmethod
    def _dominant_feature_reasons(features: Dict[str, float], modulation: str, protocol: str) -> List[str]:
        reasons: List[str] = []
        reasons.append(
            f"Bandwidth={features.get('bandwidth_hz', 0.0):.1f} Hz, IF peaks={features.get('if_peak_count', 0.0):.1f}, cyclic={features.get('cyclic_strength', 0.0):.3f}."
        )
        reasons.append(
            f"SNR={features.get('snr_db', 0.0):.2f} dB, constellation_stability={features.get('constellation_stability', 0.0):.3f}, packet_rate={features.get('packet_rate_hz', 0.0):.2f} Hz."
        )
        if modulation == "OFDM":
            reasons.append(f"OFDM evidence from subcarrier structure={features.get('subcarrier_structure', 0.0):.3f}.")
        if modulation in {"FSK", "MFSK"}:
            reasons.append("FSK evidence based on denoised IF histogram peak structure.")
        if modulation == "Chirp":
            reasons.append(f"Chirp linearity metric={features.get('chirp_linearity', 0.0):.3f}.")
        if protocol in {"Unknown Digital Signal", "Unknown Narrowband Digital Signal"}:
            reasons.append("No strong named-protocol fingerprint; selected best generic physically-consistent class.")
        return reasons
