"""Council journal: search tool (step 2), atomic append and outcome tracking (step 10)."""
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "we", "i", "should", "is", "it", "our", "my"}


def journal_path() -> Path:
    """Resolved on every call so tests and dry runs can point COUNCIL_JOURNAL elsewhere."""
    return Path(os.getenv("COUNCIL_JOURNAL", "journal/council.jsonl"))


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive lock on a sidecar .lock file. Works on Windows, macOS and Linux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # retries for ~10s, then raises
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def read_entries(path: Path | None = None) -> list[dict]:
    path = path or journal_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- Step 2: search ----------

class JournalSearchInput(BaseModel):
    query: str = Field(description="The question to find similar prior runs for.")
    limit: int = Field(default=5, description="Maximum number of runs to return.")


class JournalSearchTool(BaseTool):
    name: str = "council_journal_search"
    description: str = (
        "Searches the council journal for prior runs on similar questions. "
        "Returns run_id, date, question, recommendation, outcome and a similarity score."
    )
    args_schema: Type[BaseModel] = JournalSearchInput

    def _run(self, query: str, limit: int = 5) -> str:
        # Keyword (Jaccard) similarity. Swap in embeddings once the journal grows.
        q = _tokens(query)
        scored = []
        for e in read_entries():
            t = _tokens(e.get("question", ""))
            if q and t:
                sim = len(q & t) / len(q | t)
                if sim >= 0.15:
                    scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "No relevant prior runs found."
        keep = ("run_id", "timestamp", "question", "recommendation", "confidence", "dissent", "outcome", "outcome_notes")
        return json.dumps(
            [{"similarity": round(s, 2), **{k: e.get(k) for k in keep}} for s, e in scored[:limit]],
            indent=2,
        )


# ---------- Step 10: atomic append + outcome tracking ----------

def append_entry(entry: dict, path: Path | None = None) -> None:
    """Append one JSON line under an exclusive lock, in a single write, then fsync."""
    path = path or journal_path()
    line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with _locked(path):
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)


def record_outcome(run_id: str, outcome: str, notes: str = "", path: Path | None = None) -> bool:
    """Set the outcome of a past run. Rewrites via temp file + os.replace so readers never see a partial file."""
    path = path or journal_path()
    if not path.exists():
        return False
    with _locked(path):
        entries, found = read_entries(path), False
        for e in entries:
            if e.get("run_id") == run_id:
                e["outcome"], e["outcome_notes"], found = outcome, notes, True
        if found:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
            ) as tmp:
                for e in entries:
                    tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
        return found
