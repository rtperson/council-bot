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

class ScoreCell(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=10)


class OptionScore(BaseModel):
    option: str
    scores: list[ScoreCell]
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


def render_for_transcript(model: BaseModel) -> str:
    """Markdown for the transcript. Field text is copied, not rewritten."""
    renderer = _TRANSCRIPT_RENDERERS.get(type(model))
    if renderer is None:
        return model.model_dump_json(indent=2)
    return renderer(model)


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _cell(value: object) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(_cell(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "None."


def _triage(model: TriageResult) -> str:
    lines = [f"**Proceed:** {_yes_no(model.proceed)}", ""]
    if model.rejection_category:
        lines += [f"**Rejection:** {_label(model.rejection_category)}", ""]
    lines.append(model.rationale)
    if model.direct_answer:
        lines += ["", f"**Direct answer:** {model.direct_answer}"]
    if model.clarifying_questions:
        lines += ["", "**Clarifying questions:**", "", _bullets(model.clarifying_questions)]
    return "\n".join(lines)


def _mode(model: ModeDecision) -> str:
    advisors = ", ".join(_label(name) for name in model.advisors)
    return f"**Mode:** {model.mode}\n\n**Advisors:** {advisors}\n\n{model.rationale}"


def _expansionist(model: ExpansionistReport) -> str:
    blocks = []
    for option in model.options:
        title = " ".join(option.option.split())
        blocks.append(
            f"### {title}\n\n"
            f"**Upside:** {option.upside} \u00b7 **Effort:** {option.effort} \u00b7 **Ratio:** {option.ratio:g}\n\n"
            f"**Newly surfaced:** {_yes_no(option.newly_surfaced)}\n\n"
            f"{option.rationale}"
        )
    return "\n\n".join(blocks) if blocks else "No options."


def _executor(model: ExecutorReport) -> str:
    rows = [
        [item.action, item.reach, item.impact, item.confidence, item.effort, item.rice_score]
        for item in model.rice
    ]
    return "\n".join([
        f"**Stage:** {model.ooda_stage}",
        "",
        model.ooda_rationale,
        "",
        "### RICE",
        "",
        _table(["Action", "Reach", "Impact", "Confidence", "Effort", "RICE"], rows),
        "",
        "### Known",
        "",
        _bullets(model.data_known),
        "",
        "### Assumed",
        "",
        _bullets(model.data_assumed),
        "",
        "### Missing",
        "",
        _bullets(model.data_missing),
    ])


def _matrix(model: DecisionMatrix) -> str:
    columns = [criterion.name for criterion in model.criteria]
    for option in model.options:
        for cell in option.scores:
            if cell.criterion not in columns:
                columns.append(cell.criterion)
    score_rows = []
    for option in model.options:
        by_name = {cell.criterion: cell.score for cell in option.scores}
        score_rows.append([
            option.option,
            *[by_name.get(name, "") for name in columns],
            option.weighted_total,
        ])
    return "\n".join([
        "### Criteria",
        "",
        _table(["Criterion", "Weight", "Reason"], [[c.name, c.weight, c.reason] for c in model.criteria]),
        "",
        "### Scores",
        "",
        _table(["Option", *columns, "Weighted total"], score_rows),
        "",
        "### Dominated options",
        "",
        _bullets(model.dominated_options),
        "",
        "### What would flip the result",
        "",
        _bullets(model.flip_points),
        "",
        "### Conflicts with the advisors",
        "",
        _bullets(model.conflicts_with_advisors),
    ])


def _peer_review(model: PeerReview) -> str:
    blocks = []
    for critique in model.critiques:
        errors = _bullets(critique.errors) if critique.errors else "None."
        blocks.append(
            f"### {critique.label}\n\n"
            f"**Strongest point:** {critique.strongest_point}\n\n"
            f"**Weakest point:** {critique.weakest_point}\n\n"
            f"**Errors:**\n\n{errors}"
        )
    ranking = "\n".join(f"{i}. {label}" for i, label in enumerate(model.ranking, start=1))
    blocks.append(
        f"### Ranking\n\n{ranking or 'None.'}\n\n"
        f"{model.ranking_rationale}\n\n"
        f"**Key disagreement:** {model.key_disagreement}"
    )
    return "\n\n".join(blocks)


def _journal(model: JournalEntry) -> str:
    return "\n".join([
        f"**Recommendation:** {model.recommendation}",
        "",
        f"**Confidence:** {model.confidence}",
        "",
        f"**Dissent:** {model.dissent}",
        "",
        "**Revisit when:**",
        "",
        _bullets(model.revisit_triggers),
    ])


_TRANSCRIPT_RENDERERS = {
    TriageResult: _triage,
    ModeDecision: _mode,
    ExpansionistReport: _expansionist,
    ExecutorReport: _executor,
    DecisionMatrix: _matrix,
    PeerReview: _peer_review,
    JournalEntry: _journal,
}
