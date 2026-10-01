# src/council/crew.py
"""
Four crews built from one shared config/agents.yaml and config/tasks.yaml:

  TriageCrew   step 1        -> TriageResult
  IntakeCrew   steps 2-4     -> council brief + ModeDecision
  AdvisorCrew  steps 5-6     -> advisor analyses (parallel) + DecisionMatrix in DEEP mode
  ReviewCrew   steps 7-10    -> peer reviews, debate, chairs, HTML report, JournalEntry

Why a shared base class: CrewBase resolves every entry in tasks.yaml (its agent
and context names) for every crew that loads the file. So each crew must be able
to resolve all 9 agents and 20 tasks, even though it only runs a few. The base
class defines them all (they're built lazily, and nothing runs until kickoff),
and each crew's @crew method picks the tasks for its step.

Why four crews: CrewAI fixes a crew's task list when the crew is built, and
steps 5-10 depend on step 4's decision. flow.py runs the crews in order.
"""
import os
import warnings
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool

from .journal import JournalSearchTool
from .sources import Document, ReadDocumentTool, prompt_inputs, workspace_dir
from .models import (
    DecisionMatrix,
    ExecutorReport,
    ExpansionistReport,
    JournalEntry,
    ModeDecision,
    PeerReview,
    TriageResult,
)

# The crews' input callback is a bound method, which CrewAI can't serialize for
# checkpoint/resume. The flow doesn't use checkpointing, so silence the notice.
warnings.filterwarnings("ignore", message="method callbacks cannot be serialized")

ADVISORS = ["red_team", "first_principles", "expansionist", "outsider", "executor"]

# Models: each agent uses the `llm:` in agents.yaml. Two optional overrides,
# handy for testing:
#   COUNCIL_MODEL   use this model for every agent, e.g. ollama/qwen3:14b
#   DECISION_MODEL  use this model for the decision scientist only. Step 6 is
#                   meant as a cross-model check, so pick a different family
#                   from the advisors when you run it for real.
#
# CrewAI finds Ollama through OLLAMA_HOST (default localhost:11434). Accept the
# OLLAMA_BASE_URL name too, since that's a common way to write it in .env.
if os.getenv("OLLAMA_BASE_URL") and not os.getenv("OLLAMA_HOST"):
    os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_BASE_URL"]

# Loop limits. Each round an agent takes resends its whole conversation so far,
# so an agent that keeps calling tools gets expensive fast. CrewAI's default
# is 25 rounds per agent; these are what each job actually needs, plus slack.
MAX_ROUNDS_NO_TOOLS = 3     # advisors, chairs, report: one answer, plus format retries
MAX_ROUNDS_TOOLS = 6        # moderator at intake: a journal search, a few document reads
MAX_ROUNDS_DEBATE = 10      # debate moderator: a few questions to each side
MAX_RETRIES = 1             # CrewAI re-runs a failed task this many times (default 2)


def _check_advisors(advisors: list[str] | None) -> list[str]:
    advisors = advisors or ADVISORS
    bad = set(advisors) - set(ADVISORS)
    if bad:
        raise ValueError(f"Unknown advisors: {bad}")
    return [a for a in ADVISORS if a in advisors]  # stable order


class CouncilBase:
    """All agents and tasks. Subclasses decide which ones run."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Per-crew switches for the moderator (tools at intake, delegation at review).
    moderator_tools: bool = False

    def __init__(
        self,
        mode: str = "standard",
        advisors: list[str] | None = None,
        brief: str = "",
        responses: str = "",
        documents: list[Document] | None = None,
    ):
        """
        mode/advisors: the step 4 decision (defaults let you test a crew alone).
        brief:         the council brief from intake; defaults to the question.
        responses:     anonymized advisor outputs, used by ReviewCrew.
        documents:     files attached to this question (see sources.py).
        """
        self.mode = mode
        self.advisors = _check_advisors(advisors)
        self.brief = brief
        self.responses = responses
        self.documents = documents or []

    def _fill_inputs(self, inputs: dict | None) -> dict:
        """Runs before interpolation, so callers only ever pass `question` (and optionally `context`)."""
        inputs = dict(inputs or {})
        if not inputs.get("question"):
            raise ValueError("A 'question' input is required.")
        inputs["context"] = inputs.get("context") or "(none given)"
        for key, value in prompt_inputs(self.documents).items():
            inputs.setdefault(key, value)
        inputs["intake_brief"] = self.brief or (
            f"Question: {inputs['question']}\n\nBackground from the user:\n{inputs['context']}"
        )
        inputs["anonymized_responses"] = self.responses or "(no advisor responses supplied)"
        return inputs

    def _agent(self, name: str, override_env: str = "COUNCIL_MODEL", config: dict | None = None,
               **kwargs) -> Agent:
        model = os.getenv(override_env) or os.getenv("COUNCIL_MODEL")
        if model:
            kwargs["llm"] = model
        kwargs.setdefault("max_iter", MAX_ROUNDS_TOOLS if kwargs.get("tools") else MAX_ROUNDS_NO_TOOLS)
        kwargs.setdefault("max_retry_limit", MAX_RETRIES)
        return Agent(config=config or self.agents_config[name], **kwargs)

    def _debate_moderator(self) -> Agent:
        """
        The only agent allowed to delegate. CrewAI lets a delegating agent ask
        any other agent in its crew, and delegation is per agent rather than per
        task, so it's a separate agent used for the debate task alone.
        """
        config = {**self.agents_config["moderator"], "role": "Debate Moderator"}
        return self._agent("moderator", config=config, allow_delegation=True, max_iter=MAX_ROUNDS_DEBATE)

    def _crew(self, agent_names: list[str | Agent], tasks: list[Task]) -> Crew:
        return Crew(
            agents=[n if isinstance(n, Agent) else getattr(self, n)() for n in agent_names],
            tasks=tasks,
            process=Process.sequential,
            before_kickoff_callbacks=[self._fill_inputs],
            verbose=os.getenv("COUNCIL_VERBOSE", "1") != "0",
        )

    # ================================================================ agents
    @agent
    def red_team(self) -> Agent:
        return self._agent("red_team", allow_delegation=False)

    @agent
    def first_principles(self) -> Agent:
        return self._agent(
            "first_principles",
            allow_delegation=False,
        )

    @agent
    def expansionist(self) -> Agent:
        return self._agent("expansionist", allow_delegation=False)

    @agent
    def outsider(self) -> Agent:
        return self._agent("outsider", allow_delegation=False)

    @agent
    def executor(self) -> Agent:
        return self._agent("executor", allow_delegation=False)

    @agent
    def moderator(self) -> Agent:
        # Tools are given per task (see IntakeCrew), so each step only has the
        # ones it needs. The round limit still has to allow for tool use.
        rounds = MAX_ROUNDS_TOOLS if self.moderator_tools else MAX_ROUNDS_NO_TOOLS
        return self._agent("moderator", allow_delegation=False, max_iter=rounds)

    def _document_tools(self) -> list:
        if self.documents:
            # Documents were attached to this question: read those, and only those.
            return [ReadDocumentTool.for_documents(self.documents)]
        # Nothing attached: browse the shared workspace folder instead.
        folder = workspace_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return [DirectoryReadTool(directory=str(folder)), ReadDocumentTool.for_folder(folder)]

    @agent
    def decision_scientist(self) -> Agent:
        return self._agent("decision_scientist", override_env="DECISION_MODEL", allow_delegation=False)

    @agent
    def consensus_chair(self) -> Agent:
        return self._agent("consensus_chair", allow_delegation=False)

    @agent
    def dissent_chair(self) -> Agent:
        return self._agent("dissent_chair", allow_delegation=False)

    # ================================================================= tasks
    # Step 1
    @task
    def triage(self) -> Task:
        return Task(config=self.tasks_config["triage"], output_pydantic=TriageResult)

    # Steps 2-4
    @task
    def journal_lookup(self) -> Task:
        return Task(config=self.tasks_config["journal_lookup"])

    @task
    def frame_and_bias_audit(self) -> Task:
        return Task(config=self.tasks_config["frame_and_bias_audit"])

    @task
    def mode_selection(self) -> Task:
        return Task(config=self.tasks_config["mode_selection"], output_pydantic=ModeDecision)

    # Step 5
    @task
    def red_team_analysis(self) -> Task:
        return Task(config=self.tasks_config["red_team_analysis"])

    @task
    def first_principles_analysis(self) -> Task:
        return Task(config=self.tasks_config["first_principles_analysis"])

    @task
    def expansionist_analysis(self) -> Task:
        return Task(config=self.tasks_config["expansionist_analysis"], output_pydantic=ExpansionistReport)

    @task
    def outsider_analysis(self) -> Task:
        return Task(config=self.tasks_config["outsider_analysis"])

    @task
    def executor_analysis(self) -> Task:
        return Task(config=self.tasks_config["executor_analysis"], output_pydantic=ExecutorReport)

    # Step 6
    @task
    def decision_science(self) -> Task:
        return Task(config=self.tasks_config["decision_science"], output_pydantic=DecisionMatrix)

    # Step 7
    @task
    def peer_review_red_team(self) -> Task:
        return Task(config=self.tasks_config["peer_review_red_team"], output_pydantic=PeerReview)

    @task
    def peer_review_first_principles(self) -> Task:
        return Task(config=self.tasks_config["peer_review_first_principles"], output_pydantic=PeerReview)

    @task
    def peer_review_expansionist(self) -> Task:
        return Task(config=self.tasks_config["peer_review_expansionist"], output_pydantic=PeerReview)

    @task
    def peer_review_outsider(self) -> Task:
        return Task(config=self.tasks_config["peer_review_outsider"], output_pydantic=PeerReview)

    @task
    def peer_review_executor(self) -> Task:
        return Task(config=self.tasks_config["peer_review_executor"], output_pydantic=PeerReview)

    @task
    def forced_debate(self) -> Task:
        return Task(config=self.tasks_config["forced_debate"])

    # Step 8
    @task
    def consensus_synthesis(self) -> Task:
        return Task(config=self.tasks_config["consensus_synthesis"])

    @task
    def dissent_synthesis(self) -> Task:
        return Task(config=self.tasks_config["dissent_synthesis"])

    # Step 9
    @task
    def html_report(self) -> Task:
        return Task(config=self.tasks_config["html_report"])

    # Step 10
    @task
    def journal_record(self) -> Task:
        return Task(config=self.tasks_config["journal_record"], output_pydantic=JournalEntry)


# ==================================================================== crews

@CrewBase
class TriageCrew(CouncilBase):
    @crew
    def crew(self) -> Crew:
        return self._crew(["moderator"], [self.triage()])


@CrewBase
class IntakeCrew(CouncilBase):
    moderator_tools = True

    @crew
    def crew(self) -> Crew:
        lookup, frame, mode = self.journal_lookup(), self.frame_and_bias_audit(), self.mode_selection()
        lookup.tools = [JournalSearchTool()]
        frame.tools = self._document_tools()
        # mode_selection gets no tools: it only reads the brief.
        return self._crew(["moderator"], [lookup, frame, mode])


@CrewBase
class AdvisorCrew(CouncilBase):
    @crew
    def crew(self) -> Crew:
        deep = self.mode == "deep"
        fan_out = [getattr(self, f"{a}_analysis")() for a in self.advisors]

        # Parallel fan-out. CrewAI rejects a crew that ends in more than one
        # async task, so without step 6 the last advisor runs synchronously
        # (the crew still waits for the others before returning).
        for i, t in enumerate(fan_out):
            t.async_execution = deep or i < len(fan_out) - 1

        tasks, agent_names = list(fan_out), list(self.advisors)
        if deep:
            ds = self.decision_science()
            ds.context = fan_out                  # waits for every advisor
            tasks.append(ds)
            agent_names.append("decision_scientist")
        return self._crew(agent_names, tasks)


@CrewBase
class ReviewCrew(CouncilBase):
    @crew
    def crew(self) -> Crew:
        reviews = [getattr(self, f"peer_review_{a}")() for a in self.advisors]
        for r in reviews:
            r.async_execution = True              # parallel; next task waits via context

        tasks, extra_agents = list(reviews), []
        if self.mode == "quick":                  # QUICK skips the debate
            chair_context = reviews
        else:
            debate = self.forced_debate()
            debate.agent = self._debate_moderator()
            debate.context = reviews
            tasks.append(debate)
            extra_agents.append(debate.agent)
            chair_context = [debate]

        consensus, dissent = self.consensus_synthesis(), self.dissent_synthesis()
        consensus.context = chair_context
        dissent.context = chair_context           # independent of the consensus chair on purpose
        tasks += [consensus, dissent, self.html_report(), self.journal_record()]

        return self._crew(
            list(self.advisors) + ["moderator", "consensus_chair", "dissent_chair"] + extra_agents,
            tasks,
        )
