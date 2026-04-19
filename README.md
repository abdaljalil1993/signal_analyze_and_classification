# RF IQ SIGINT System

Production-style modular RF signal intelligence pipeline and UI for raw IQ files.

## Modules

- `iq_loader/` : raw IQ loaders with int8/int16/float32 support + streaming
- `preprocessing/` : DC removal and normalization
- `feature_extraction/` : vectorized, parallel feature extraction + caching
- `classifiers/` : strict hierarchical classifier (stages 1-5)
- `protocol_detectors/` : protocol-aware scoring (WiFi, LoRa, DMR, RC, Drone, IoT)
- `constraint_engine/` : physics-based hard constraints and conflict checks
- `decision_engine/` : candidate generation, confidence, top-3 output
- `ui/` : PySide6 + pyqtgraph interactive UI
- `utils/` : shared config, cache, and data classes

## Run

```powershell
pip install -r requirements.txt
python main.py
```

## CLI Processing

```powershell
python run_pipeline.py <path_to_iq_file> --dtype int16 --fs 1000000 --chunk 100000
```

## Latency Check

```powershell
python benchmark_latency.py
```

## Pipeline

1. IQ load (`int8`, `int16`, `float32`)
2. Preprocess (DC removal, normalization)
3. Parallel feature extraction (instantaneous frequency, IF peaks, cyclostationary, autocorr, constellation clustering, spectral, burst)
4. Hierarchical classification
5. Physics constraints + conflict resolution
6. Final physically valid decision + confidence
7. Interactive visualization and reasoning output
