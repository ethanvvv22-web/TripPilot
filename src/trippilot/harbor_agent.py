"""Run TripPilot as a custom Harbor agent.

    harbor run -p eval/harbor/tasks -a trippilot.harbor_agent:TripPilotAgent -n 2 -o eval/harbor/jobs -y

The agent loop runs on the host (it calls Microsoft Foundry); only the final
itinerary is written into the task container, where Harbor's verifier scores it.
Set TRIPPILOT_TOOLS=legacy to benchmark the unbatched tool style.
"""

import asyncio
import json
import os
import uuid
from datetime import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .agent.foundry import FoundryAgent
from .agent.planner import PlanResult, TripPlanner, instructions, tool_specs
from .harbor_tasks import parse_spec
from .models import Itinerary, Place, Stop, TripRequest
from .dataset import load as load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = "/app/itinerary.json"


def run_spec(spec: Dict[str, Any], batched: bool) -> PlanResult:
    load_dotenv(REPO_ROOT / ".env")
    agent = FoundryAgent(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        instructions=instructions(batched),
        tool_specs=tool_specs(batched),
        # Harbor runs trials concurrently; a shared name would let one trial
        # delete the agent another trial is still using.
        agent_name=f"trippilot-{uuid.uuid4().hex[:12]}",
    )
    try:
        dataset = load_dataset(str(REPO_ROOT / spec["dataset"]) if spec.get("dataset") else None)
        planner = TripPlanner(agent, dataset.provider(), dataset.travel_minutes, batched=batched)
        request = TripRequest(**spec["request"])
        if spec.get("replan"):
            replan = spec["replan"]
            base = Itinerary(date=request.date, travel_mode=request.travel_mode,
                             stops=[Stop(**s) for s in replan["base_stops"]])
            return planner.replan(request, base, time.fromisoformat(replan["now"]), replan["change"],
                                  dataset.by_id())
        return planner.plan(request)
    finally:
        agent.delete()


class TripPilotAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "trippilot"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass  # nothing to install: the agent loop runs on the host

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        spec = parse_spec(instruction)
        batched = os.environ.get("TRIPPILOT_TOOLS", "batched") != "legacy"
        result = await asyncio.to_thread(run_spec, spec, batched)

        stops = result.itinerary.stops if result.itinerary else []
        output = {"stops": [json.loads(s.model_dump_json()) for s in stops], "status": result.status}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        local = self.logs_dir / "itinerary.json"
        local.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        await environment.upload_file(local, OUTPUT_PATH)

        # The agent's trajectory: every tool call with its arguments and output.
        trace = [{"tool": r.name, "arguments": r.arguments, "output": r.output, "ms": round(r.duration_ms, 1)}
                 for r in result.trace]
        (self.logs_dir / "tool_calls.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")

        context.n_input_tokens = result.input_tokens
        context.n_output_tokens = result.output_tokens
        context.metadata = {
            "status": result.status,
            "submissions": result.submissions,
            "turns": result.turns,
            "tool_calls": len(result.trace),
            "model_seconds": round(result.model_seconds, 1),
            "tools": "batched" if batched else "legacy",
        }
