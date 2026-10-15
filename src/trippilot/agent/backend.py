"""The model-facing interface the planner talks to.

Keeping this small lets the planner run against Microsoft Foundry in
production and against scripted fakes in tests.
"""

from dataclasses import dataclass, field
from typing import List, Protocol, Tuple


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # raw JSON from the model; may be malformed


@dataclass
class ModelTurn:
    tool_calls: List[ToolCall] = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class AgentSession(Protocol):
    """One conversation with the agent."""

    def send_user_message(self, text: str) -> ModelTurn: ...

    # (call_id, JSON output) pairs for every tool call in the previous turn.
    def send_tool_results(self, results: List[Tuple[str, str]]) -> ModelTurn: ...

    def close(self) -> None: ...


class AgentBackend(Protocol):
    def new_session(self) -> AgentSession: ...
