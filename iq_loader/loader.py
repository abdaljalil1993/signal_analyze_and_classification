from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal

import numpy as np

DTypeName = Literal["int8", "int16", "float32"]


def _resolve_dtype(dtype_name: DTypeName) -> np.dtype:
    if dtype_name == "int8":
        return np.dtype(np.int8)
    if dtype_name == "int16":
        return np.dtype(np.int16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _to_complex_interleaved(raw: np.ndarray, dtype_name: DTypeName) -> np.ndarray:
    if raw.size % 2 != 0:
        raw = raw[:-1]
    iq = raw.reshape(-1, 2)
    if dtype_name == "float32":
        scale = 1.0
    elif dtype_name == "int16":
        scale = 32768.0
    else:
        scale = 128.0
    i = iq[:, 0].astype(np.float32) / scale
    q = iq[:, 1].astype(np.float32) / scale
    return i + 1j * q


def load_iq_file(path: str | Path, dtype_name: DTypeName) -> np.ndarray:
    dtype = _resolve_dtype(dtype_name)
    raw = np.fromfile(path, dtype=dtype)
    return _to_complex_interleaved(raw, dtype_name)


def stream_iq_file(
    path: str | Path,
    dtype_name: DTypeName,
    chunk_samples: int,
) -> Generator[np.ndarray, None, None]:
    dtype = _resolve_dtype(dtype_name)
    chunk_values = chunk_samples * 2
    with open(path, "rb") as f:
        while True:
            raw = np.fromfile(f, dtype=dtype, count=chunk_values)
            if raw.size == 0:
                break
            yield _to_complex_interleaved(raw, dtype_name)
