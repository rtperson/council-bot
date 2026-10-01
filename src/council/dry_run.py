# src/council/dry_run.py
"""
Run the whole flow without calling any model.

Every crew still gets built and still interpolates its inputs, so a missing
{placeholder}, a broken context link, or a crash in the flow's own code shows
up exactly as it would in a real run. Only the LLM calls are replaced: each
task gets a canned output that matches its output_pydantic model.

    with fake_llm_run(mode="standard") as seen:
        CouncilFlow().kickoff(inputs={"question": "..."})
    seen["red_team_analysis"]   # the task's prompt, exactly as the model would get it

Used by `council --dry-run` and by the tests.
"""
import types
import typing
from contextlib import contextmanager
from typing import Any, Literal, Union, get_args, get_origin

from crewai import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel

from .models import ModeDecision, TriageResult

FAKE_ADVISORS = {
    "quick": ["red_team", "outsider", "executor"],
    "standard": ["red_team", "first_principles", "expansionist", "executor"],
    "deep": ["red_team", "first_principles", "expansionist", "outsider", "executor"],
}

FAKE_HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Dry run</title></head>"
    "<body style='font-family:system-ui;max-width:40rem;margin:2rem auto'>"
    "<h1>Dry-run report</h1><p>No model was called. This file shows the flow "
    "reached step 9 and wrote its outputs.</p></body></html>"
)


def _fake_value(tp: Any, name: str, meta: list) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        return None if type(None) in args else _fake_value(args[0], name, meta)
    if origin is Literal:
        return args[0]
    if origin in (list, typing.List):
        return [_fake_value(args[0], name, [])] if args else []
    if origin in (dict, typing.Dict):
        return {"example": _fake_value(args[1], name, [])} if args else {}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return fake_model(tp)
    if tp is bool:
        return True
    if tp in (int, float):
        lo = next((m.ge for m in meta if hasattr(m, "ge")), None)
        hi = next((m.le for m in meta if hasattr(m, "le")), None)
        v = 5 if lo is None or hi is None else (lo + hi) // 2
        return tp(v)
    return f"[dry-run {name}]"


def fake_model(model: type[BaseModel]) -> BaseModel:
    values = {n: _fake_value(f.annotation, n, f.metadata) for n, f in model.model_fields.items()}
    return model.model_validate(values)


@contextmanager
def fake_llm_run(mode: str = "standard", triage_proceeds: bool = True):
    """
    Patch Crew.kickoff so every crew returns canned, schema-valid outputs.
    Yields a dict of task name -> the task's fully interpolated prompt.
    """
    original = Crew.kickoff
    seen: dict[str, str] = {}

    def fake_kickoff(self: Crew, inputs: dict | None = None, **_: Any) -> CrewOutput:
        # Same input handling as a real kickoff: callbacks, then interpolation.
        for cb in self.before_kickoff_callbacks:
            inputs = cb(inputs or {})
        inputs = inputs or {}
        for agent in self.agents:
            agent.interpolate_inputs(inputs)
        outputs = []
        for task in self.tasks:
            task.interpolate_inputs_and_add_conversation_history(inputs)
            seen[task.name] = task.description
            model = task.output_pydantic
            if model is TriageResult:
                obj = TriageResult(
                    proceed=triage_proceeds,
                    rejection_category=None if triage_proceeds else "factual",
                    rationale="[dry-run triage]",
                    direct_answer=None if triage_proceeds else "[dry-run direct answer]",
                )
            elif model is ModeDecision:
                obj = ModeDecision(mode=mode, advisors=FAKE_ADVISORS[mode], rationale="[dry-run mode choice]")
            elif model is not None:
                obj = fake_model(model)
            else:
                obj = None
            if obj is not None:
                raw = obj.model_dump_json(indent=2)
            elif task.name == "html_report":
                raw = FAKE_HTML
            else:
                raw = f"[dry-run output of {task.name}]"
            outputs.append(TaskOutput(
                description=task.description,
                name=task.name,
                expected_output=task.expected_output,
                raw=raw,
                pydantic=obj,
                agent=task.agent.role.strip() if task.agent else "unknown",
                output_format=OutputFormat.PYDANTIC if obj is not None else OutputFormat.RAW,
            ))
        last = outputs[-1]
        return CrewOutput(raw=last.raw, pydantic=last.pydantic, tasks_output=outputs)

    Crew.kickoff = fake_kickoff
    try:
        yield seen
    finally:
        Crew.kickoff = original
