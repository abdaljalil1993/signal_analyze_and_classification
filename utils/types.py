from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageDecision:
    stage: str
    scores: Dict[str, float]
    selected: str


@dataclass
class Candidate:
    signal_type: str
    channel_type: str
    modulation: str
    protocol: str
    application: str
    score: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    signal_type: str
    channel_type: str
    modulation: str
    protocol: str
    application: str
    confidence: float
    top_candidates: List[Candidate]
    stage_trace: List[StageDecision]
    feature_summary: Dict[str, float]
    reasoning: List[str]
