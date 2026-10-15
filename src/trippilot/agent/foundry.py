"""Microsoft Foundry Agent Service backend.

Follows the prompt-agent function-calling pattern from the Foundry docs:
the agent definition holds the instructions and function tools, and each
planning run is one conversation driven through the Responses API.
"""

from typing import Any, Dict, List, Optional, Tuple

from .backend import ModelTurn, ToolCall


class FoundryAgent:
    def __init__(
        self,
        endpoint: str,
        model: str,
        instructions: str,
        tool_specs: List[Dict[str, Any]],
        agent_name: str = "trippilot-planner",
        credential: Optional[Any] = None,
    ):
        # Imported here so the rest of the package works without the Azure SDK.
        from azure.ai.projects import AIProjectClient
        from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
        from azure.identity import DefaultAzureCredential

        self.project = AIProjectClient(endpoint=endpoint, credential=credential or DefaultAzureCredential())
        self.openai = self.project.get_openai_client()
        tools = [
            FunctionTool(name=s["name"], description=s["description"], parameters=s["parameters"], strict=True)
            for s in tool_specs
        ]
        self.agent = self.project.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(model=model, instructions=instructions, tools=tools),
        )

    def new_session(self) -> "FoundrySession":
        return FoundrySession(self.openai, self.agent.name)

    def delete(self) -> None:
        self.project.agents.delete_version(agent_name=self.agent.name, agent_version=self.agent.version)


class FoundrySession:
    def __init__(self, openai: Any, agent_name: str):
        self.openai = openai
        self.agent_name = agent_name
        self.conversation = openai.conversations.create()

    def send_user_message(self, text: str) -> ModelTurn:
        return self._respond(text)

    def send_tool_results(self, results: List[Tuple[str, str]]) -> ModelTurn:
        return self._respond([
            {"type": "function_call_output", "call_id": call_id, "output": output}
            for call_id, output in results
        ])

    def close(self) -> None:
        self.openai.conversations.delete(conversation_id=self.conversation.id)

    def _respond(self, input: Any) -> ModelTurn:
        response = self.openai.responses.create(
            input=input,
            conversation=self.conversation.id,
            extra_body={"agent_reference": {"name": self.agent_name, "type": "agent_reference"}},
        )
        calls = [
            ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
        usage = getattr(response, "usage", None)
        return ModelTurn(
            tool_calls=calls,
            text=response.output_text or "",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
