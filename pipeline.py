from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from classifiers.hierarchical import HierarchicalClassifier
from constraint_engine.physics_rules import apply_physics_constraints
from decision_engine.engine import DecisionEngine
from feature_extraction.core import extract_features
from iq_loader.loader import DTypeName, load_iq_file, stream_iq_file
from preprocessing.filters import preprocess_iq
from protocol_detectors.detectors import detect_protocols
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_RATE
from utils.types import PipelineResult


class SigIntPipeline:
    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_analysis_samples: int = 300_000,
        max_plot_samples: int = 60_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_analysis_samples = max_analysis_samples
        self.max_plot_samples = max_plot_samples
        self.classifier = HierarchicalClassifier()
        self.decision_engine = DecisionEngine()

    def process_file(self, path: str | Path, dtype_name: DTypeName) -> tuple[PipelineResult, np.ndarray, dict]:
        iq = self._load_analysis_window(path, dtype_name)
        pre = preprocess_iq(iq)
        features = extract_features(pre, self.sample_rate, cache_key=(str(path), "full", pre.size), use_parallel=True)

        stage1 = self.classifier._stage_signal_nature(features)
        stage2 = self.classifier._stage_channel(features, stage1.selected)
        stage3 = self.classifier._stage_modulation(features, stage1.selected, stage2.selected)

        protocol_scores = detect_protocols(features, stage2.selected, stage3.selected, stage1.selected)
        stage_trace = self.classifier.classify(features, protocol_scores)

        constraints = apply_physics_constraints(
            stage_trace[1].selected,
            stage_trace[2].scores,
            stage_trace[3].scores,
            features,
        )
        result = self.decision_engine.build_result(stage_trace, features, constraints)

        plot_data = self._build_plot_data(pre)
        return result, pre, plot_data

    def _load_analysis_window(self, path: str | Path, dtype_name: DTypeName) -> np.ndarray:
        # Read only a bounded window for interactive latency.
        chunks = []
        collected = 0
        for chunk in stream_iq_file(path, dtype_name, self.chunk_size):
            chunks.append(chunk)
            collected += chunk.size
            if collected >= self.max_analysis_samples:
                break
        if chunks:
            iq = np.concatenate(chunks)
            return iq[: self.max_analysis_samples]
        return load_iq_file(path, dtype_name)

    def process_streaming(self, path: str | Path, dtype_name: DTypeName) -> Iterable[PipelineResult]:
        for idx, chunk in enumerate(stream_iq_file(path, dtype_name, self.chunk_size)):
            pre = preprocess_iq(chunk)
            features = extract_features(
                pre,
                self.sample_rate,
                cache_key=(str(path), idx, pre.size),
                use_parallel=True,
            )

            stage1 = self.classifier._stage_signal_nature(features)
            stage2 = self.classifier._stage_channel(features, stage1.selected)
            stage3 = self.classifier._stage_modulation(features, stage1.selected, stage2.selected)
            protocol_scores = detect_protocols(features, stage2.selected, stage3.selected, stage1.selected)
            stage_trace = self.classifier.classify(features, protocol_scores)
            constraints = apply_physics_constraints(
                stage_trace[1].selected,
                stage_trace[2].scores,
                stage_trace[3].scores,
                features,
            )
            yield self.decision_engine.build_result(stage_trace, features, constraints)

    def _build_plot_data(self, iq: np.ndarray) -> dict:
        if iq.size > self.max_plot_samples:
            sel = np.linspace(0, iq.size - 1, self.max_plot_samples, dtype=np.int64)
            iq_plot = iq[sel]
        else:
            iq_plot = iq

        n = iq_plot.size
        time = np.arange(n) / self.sample_rate

        nfft = 16384 if n >= 16384 else max(1024, 1 << (n - 1).bit_length())
        segment = iq_plot[:nfft]
        if segment.size < nfft:
            pad = np.zeros(nfft, dtype=np.complex64)
            pad[: segment.size] = segment
            segment = pad

        window = np.hanning(nfft)
        spec = np.fft.fftshift(np.fft.fft(segment * window))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / self.sample_rate))
        psd_db = 20.0 * np.log10(np.abs(spec) + 1e-12)

        phase = np.unwrap(np.angle(iq_plot))
        inst_f = np.diff(phase, prepend=phase[0]) * self.sample_rate / (2.0 * np.pi)

        if_win = max(3, min(31, (inst_f.size // 4000) | 1))
        if if_win > 1 and inst_f.size >= if_win:
            kernel = np.ones(if_win, dtype=np.float32) / float(if_win)
            inst_f_dn = np.convolve(inst_f.astype(np.float32), kernel, mode="same")
        else:
            inst_f_dn = inst_f.astype(np.float32)

        if_hist, if_edges = np.histogram(inst_f_dn, bins=96)
        if_hist_s = np.convolve(if_hist.astype(np.float32), np.ones(5, dtype=np.float32) / 5.0, mode="same")
        mid = if_hist_s[1:-1]
        left = if_hist_s[:-2]
        right = if_hist_s[2:]
        thr = 0.35 * float(np.max(if_hist_s) + 1e-12)
        if_peaks = np.where((mid > left) & (mid > right) & (mid >= thr))[0] + 1
        if_centers = 0.5 * (if_edges[:-1] + if_edges[1:])

        wf_window = max(256, min(1024, n // 30 if n // 30 > 0 else 256))
        step = max(64, wf_window // 2)
        strips = []
        for i in range(0, max(0, n - wf_window), step):
            seg = iq_plot[i : i + wf_window]
            s = np.fft.fftshift(np.fft.fft(seg * np.hanning(wf_window)))
            strips.append(20.0 * np.log10(np.abs(s) + 1e-12))
            if len(strips) >= 200:
                break
        waterfall = np.array(strips, dtype=np.float32) if strips else np.zeros((1, wf_window), dtype=np.float32)

        env = np.abs(iq_plot)
        env_thr = np.mean(env) + 0.8 * np.std(env)
        burst_active = (env > env_thr).astype(np.float32)

        return {
            "time": time,
            "iq_real": np.real(iq_plot),
            "iq_imag": np.imag(iq_plot),
            "freq": freq,
            "psd_db": psd_db,
            "const_i": np.real(iq_plot),
            "const_q": np.imag(iq_plot),
            "inst_freq": inst_f_dn,
            "if_hist_x": if_centers,
            "if_hist_y": if_hist_s,
            "if_hist_peak_x": if_centers[if_peaks] if if_peaks.size else np.array([], dtype=np.float32),
            "if_hist_peak_y": if_hist_s[if_peaks] if if_peaks.size else np.array([], dtype=np.float32),
            "waterfall": waterfall,
            "burst_time": time,
            "burst_active": burst_active,
        }
