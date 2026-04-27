from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from pipeline import SigIntPipeline


class ProcessingWorker(QThread):
    done = Signal(object, object)
    failed = Signal(str)

    def __init__(self, pipeline: SigIntPipeline, path: str, dtype_name: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.path = path
        self.dtype_name = dtype_name

    def run(self) -> None:
        try:
            result, _, plots = self.pipeline.process_file(self.path, self.dtype_name)
            self.done.emit(result, plots)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class DropZone(QLabel):
    file_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__("Drop IQ file here or click Browse")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel {border: 2px dashed #3b6ea5; border-radius: 8px; padding: 20px; background:#ecf2f9; color:#123;}"
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IQ SIGINT Analyzer")
        self.resize(1400, 900)

        self.pipeline = SigIntPipeline()
        self.worker: ProcessingWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._start_processing)
        self.btn_browse = QPushButton("Browse IQ File")
        self.btn_browse.clicked.connect(self._browse)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["int8", "int16", "float32"])
        self.status_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        top.addWidget(self.drop_zone, 3)
        side = QVBoxLayout()
        side.addWidget(QLabel("Input dtype"))
        side.addWidget(self.dtype_combo)
        side.addWidget(self.btn_browse)
        side.addWidget(self.status_label)
        side.addWidget(self.progress)
        side.addStretch(1)
        top.addLayout(side, 1)
        root.addLayout(top)

        grid = QGridLayout()
        root.addLayout(grid, 1)

        pg.setConfigOptions(antialias=True)

        self.time_plot = pg.PlotWidget(title="Time Domain")
        self.spec_plot = pg.PlotWidget(title="Spectrum (FFT)")
        self.const_plot = pg.PlotWidget(title="Constellation")
        self.if_plot = pg.PlotWidget(title="Instantaneous Frequency")
        self.if_hist_plot = pg.PlotWidget(title="IF Histogram + Peaks")
        self.burst_plot = pg.PlotWidget(title="Packet/Burst Detection")
        self.waterfall_plot = pg.ImageView(view=pg.PlotItem(title="Waterfall"))

        self.const_plot.setAspectLocked(True)

        grid.addWidget(self.time_plot, 0, 0)
        grid.addWidget(self.spec_plot, 0, 1)
        grid.addWidget(self.const_plot, 1, 0)
        grid.addWidget(self.if_plot, 1, 1)
        grid.addWidget(self.if_hist_plot, 2, 0)
        grid.addWidget(self.burst_plot, 2, 1)
        grid.addWidget(self.waterfall_plot, 0, 2, 3, 1)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(200)
        root.addWidget(self.output_box)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select IQ File",
            str(Path.cwd()),
            "All files (*);;Raw IQ files (*.iq *.bin *.dat)",
        )
        if path:
            self._start_processing(path)

    def _start_processing(self, path: str) -> None:
        self.status_label.setText(f"Processing: {Path(path).name}")
        self.progress.setVisible(True)
        self.btn_browse.setEnabled(False)

        self.worker = ProcessingWorker(self.pipeline, path, self.dtype_combo.currentText())
        self.worker.done.connect(self._on_result)
        self.worker.failed.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, result, plots) -> None:
        self._render_plots(plots)
        self._render_output(result)

    def _on_error(self, msg: str) -> None:
        self.output_box.setPlainText(f"Error: {msg}")

    def _on_finished(self) -> None:
        self.progress.setVisible(False)
        self.btn_browse.setEnabled(True)
        self.status_label.setText("Idle")

    def _render_plots(self, plots: dict) -> None:
        self.time_plot.clear()
        self.time_plot.plot(plots["time"], plots["iq_real"], pen=pg.mkPen("#0f4c5c", width=1.0), name="I")
        self.time_plot.plot(plots["time"], plots["iq_imag"], pen=pg.mkPen("#e36414", width=1.0), name="Q")

        self.spec_plot.clear()
        self.spec_plot.plot(plots["freq"], plots["psd_db"], pen=pg.mkPen("#1d3557", width=1.2))

        self.const_plot.clear()
        stride = max(1, len(plots["const_i"]) // 5000)
        self.const_plot.plot(
            plots["const_i"][::stride],
            plots["const_q"][::stride],
            pen=None,
            symbol="o",
            symbolSize=3,
            symbolBrush=(29, 78, 137, 120),
        )

        self.if_plot.clear()
        self.if_plot.plot(plots["inst_freq"], pen=pg.mkPen("#6a994e", width=1.0))

        self.if_hist_plot.clear()
        self.if_hist_plot.plot(plots["if_hist_x"], plots["if_hist_y"], pen=pg.mkPen("#264653", width=1.3))
        if len(plots["if_hist_peak_x"]) > 0:
            self.if_hist_plot.plot(
                plots["if_hist_peak_x"],
                plots["if_hist_peak_y"],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=(231, 111, 81, 220),
            )

        self.burst_plot.clear()
        self.burst_plot.plot(plots["burst_time"], plots["burst_active"], pen=pg.mkPen("#8338ec", width=1.0))
        self.burst_plot.setYRange(-0.05, 1.05)

        wf = np.array(plots["waterfall"], dtype=np.float32)
        self.waterfall_plot.setImage(wf, autoLevels=True)

    def _render_output(self, result) -> None:
        lines = [
            f"Signal Type: {result.signal_type}",
            f"Channel Type: {result.channel_type}",
            f"Modulation: {result.modulation}",
            f"Protocol: {result.protocol}",
            f"Application: {result.application}",
            f"Confidence: {result.confidence * 100:.2f}%",
            f"Best Channel Offset (Hz): {result.best_channel_frequency_offset_hz:.1f}",
            "",
            "Top 3 Candidates:",
        ]

        for i, c in enumerate(result.top_candidates, start=1):
            lines.append(
                f"{i}. {c.signal_type}/{c.channel_type}/{c.modulation}/{c.protocol}/{c.application} - {c.confidence * 100:.2f}%"
            )

        lines.append("")
        lines.append("Per-channel Classification Scores:")
        for ch in result.per_channel_classification_scores:
            lines.append(
                "- idx={idx:.0f}, f_off={fo:.1f} Hz, bw={bw:.1f} Hz, SNR={snr:.2f} dB, sel={sel:.3f}, conf={conf:.3f}, mod={mod}, proto={proto}".format(
                    idx=float(ch.get("channel_index", 0.0)),
                    fo=float(ch.get("frequency_offset_hz", 0.0)),
                    bw=float(ch.get("bandwidth_hz", 0.0)),
                    snr=float(ch.get("snr_db", 0.0)),
                    sel=float(ch.get("selection_score", 0.0)),
                    conf=float(ch.get("confidence", 0.0)),
                    mod=str(ch.get("modulation", "")),
                    proto=str(ch.get("protocol", "")),
                )
            )

        lines.append("")
        lines.append("Signal Specifications:")
        for k, v in result.feature_summary.items():
            lines.append(f"- {k}: {v:.4f}")

        lines.append("")
        lines.append("Hierarchical Stage Decisions:")
        for st in result.stage_trace:
            top_score = st.scores.get(st.selected, 0.0) * 100.0
            lines.append(f"- {st.stage}: {st.selected} ({top_score:.2f}%)")

        lines.append("")
        lines.append("Detailed Reasoning:")
        lines.extend(result.reasoning)

        self.output_box.setPlainText("\n".join(lines))
