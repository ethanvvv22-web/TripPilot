"""OpenTelemetry tracing for agent runs.

Code in the package only uses the OpenTelemetry API, which is a no-op until a
tracer provider is configured. Entry points (CLI, UI, eval runner) call
`setup_tracing` to export spans, e.g. as JSON lines for later analysis.

Span names and attributes follow the OpenTelemetry GenAI semantic conventions
where they apply (`gen_ai.operation.name`, `gen_ai.tool.name`, token usage).
"""

import json
import threading
from typing import Optional, Sequence

from opentelemetry import trace

tracer = trace.get_tracer("trippilot")


def setup_tracing(jsonl_path: Optional[str] = None, console: bool = False) -> None:
    """Install a global tracer provider; call once at program start."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "trippilot"}))
    if jsonl_path:
        provider.add_span_processor(SimpleSpanProcessor(JsonLinesSpanExporter(jsonl_path)))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


class JsonLinesSpanExporter:
    """Appends each finished span to a file as one compact JSON object per line."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def export(self, spans: Sequence) -> "SpanExportResult":
        from opentelemetry.sdk.trace.export import SpanExportResult

        lines = [json.dumps(json.loads(span.to_json()), ensure_ascii=False) for span in spans]
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
