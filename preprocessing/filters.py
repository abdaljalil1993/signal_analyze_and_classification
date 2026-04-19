from __future__ import annotations

import numpy as np

from utils.config import EPS


def remove_dc_offset(iq: np.ndarray) -> np.ndarray:
    return iq - np.mean(iq)


def normalize(iq: np.ndarray) -> np.ndarray:
    power = np.sqrt(np.mean(np.abs(iq) ** 2) + EPS)
    return iq / power


def preprocess_iq(iq: np.ndarray) -> np.ndarray:
    return normalize(remove_dc_offset(iq))
