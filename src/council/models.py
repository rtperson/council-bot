# src/council/models.py
"""Structured outputs for the tasks that feed code, not just other agents."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

AdvisorName = Literal["red_team", "first_principles", "expansionist", "outsider", "executor"]


# Step 1
class TriageResult(BaseModel):
    proceed: bool
    rejection_category: Optional[Literal["factual", "obvious", "not_a_decision", "too_vague"]] = None
    rationale: str
    direct_answer: Optional[str] = None
    clarifying_questions: list[str] = []


# Step 4
class ModeDecision(BaseModel):
    mode: Literal["quick", "standard", "deep"]
    advisors: list[AdvisorName]
    rationale: str


# Step 5 (scoring roles)
class ScoredOption(BaseModel):
    option: str
    upside: int = Field(ge=1, le=10)
    effort: int = Field(ge=1, le=10)
    ratio: float
    rationale: str
    newly_surfaced: bool


class ExpansionistReport(BaseModel):
    options: list[ScoredOption]


class RiceItem(BaseModel):
    action: str
    reach: float
    impact: float
    confidence: float = Field(ge=0, le=1)
    effort: float
    rice_score: float


class ExecutorReport(BaseModel):
    ooda_stage: Literal["Observe", "Orient", "Decide", "Act"]
    ooda_rationale: str
    rice: list[RiceItem]
    data_known: list[str]
    data_assumed: list[str]
    data_missing: list[str]


# Step 6
class Criterion(BaseModel):
    name: str
    weight: float = Field(ge=0, le=1)
    reason: str


class OptionScore(BaseModel):
    option: str
    scores: dict[str, int]          # criterion name -> 1..10
    weighted_total: float


class DecisionMatrix(BaseModel):
    criteria: list[Criterion]
    options: list[OptionScore]
    dominated_options: list[str]
    flip_points: list[str]
    conflicts_with_advisors: list[str]


# Step 7
class ResponseCritique(BaseModel):
    label: str                      # "Response A" ...
    strongest_point: str
    weakest_point: str
    errors: list[str] = []


class PeerReview(BaseModel):
    critiques: list[ResponseCritique]
    ranking: list[str]              # labels, best first
    ranking_rationale: str
    key_disagreement: str


# Step 10: only what the LLM should decide. flow.py adds run_id, timestamp,
# question, mode, advisors and the outcome fields before writing the record.
class JournalEntry(BaseModel):
    recommendation: str
    confidence: int = Field(ge=0, le=100)
    dissent: str
    revisit_triggers: list[str]
