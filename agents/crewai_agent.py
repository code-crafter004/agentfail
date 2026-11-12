"""CrewAI adapter: single agent, same tools, same prompt text, same trace format as LangGraph.

How parity is kept (this is the part to read before changing anything):
- Tools: every CrewAI tool only calls gateway.call(...), so tool events are identical across frameworks.
  The tool schema the model sees is our own spec (same JSON schema as LangGraph). CrewAI's own argument
  validation is switched off (permissive schema) so that bad arguments reach the gateway and are logged
  as SCHEMA_INVALID, exactly like in the LangGraph run.
- LLM turns: logged from CrewAI's before and after LLM call hooks as llm_request and llm_response.
- Step limit: counted in the before hook. At the limit we log MAX_STEPS_EXCEEDED and stop the run, the
  same as the LangGraph adapter. CrewAI's own max_iter is only a safety net above our limit.
- Tool result caching and agent retries are turned off, so repeated calls and injected faults reach the gateway.
- CrewAI's text mode blocks an identical tool call made twice in a row ("I tried reusing the same input").
  That would stop an agent from retrying a timed out API call, so the guard is switched off during our runs.
  (Gemini and Groq use native function calling, which does not have this guard.)

Known differences (record them in the report): CrewAI wraps the task in its own prompt template, and it
has its own provider retry policy for rate limits, so provider retries are not logged as retry events.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest import mock

# Turn off CrewAI telemetry before it is imported (no network calls, no prompts about tracing).
for _k, _v in (("CREWAI_DISABLE_TELEMETRY", "true"), ("OTEL_SDK_DISABLED", "true"), ("CREWAI_TRACING_ENABLED", "false")):
    os.environ.setdefault(_k, _v)

from crewai import Agent, Crew, Process, Task as CrewTask  # noqa: E402
from crewai.hooks.dispatch import HookAborted  # noqa: E402
from crewai.hooks.llm_hooks import (  # noqa: E402
    register_after_llm_call_hook,
    register_before_llm_call_hook,
    unregister_after_llm_call_hook,
    unregister_before_llm_call_hook,
)
from crewai.tools import BaseTool  # noqa: E402
from crewai.tools.tool_usage import ToolUsage  # noqa: E402
from pydantic import BaseModel, ConfigDict, PrivateAttr  # noqa: E402

from agents.common import text_of  # noqa: E402
from agents.prompts import SYSTEM_PROMPT, build_task_prompt  # noqa: E402
from tools.gateway import TOOL_SPECS, ToolGateway  # noqa: E402
from tracing.collector import TraceCollector  # noqa: E402


class StepLimitExceeded(HookAborted):
    """HookAborted is the one exception CrewAI lets out of a hook (others are swallowed)."""

    def __init__(self) -> None:
        super().__init__("step limit reached")


def _permissive_args(spec: dict) -> type[BaseModel]:
    """Accept any arguments (validation happens in the gateway) but advertise our exact JSON schema."""
    schema = {**spec["parameters"], "title": spec["name"]}

    class Args(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *a: Any, **k: Any) -> dict:  # what the LLM and the prompt see
            return json.loads(json.dumps(schema))

    Args.__name__ = f"{spec['name']}_args"
    return Args


class GatewayTool(BaseTool):
    name: str
    description: str
    _gateway: ToolGateway = PrivateAttr()
    _state: Any = PrivateAttr()

    def _run(self, **kwargs: Any) -> str:
        payload = self._gateway.call(self.name, kwargs, parent_event_id=self._state.last_event)
        return json.dumps(payload, default=str)


def build_tools(gateway: ToolGateway, state: Any) -> list[GatewayTool]:
    tools = []
    for spec in gateway.specs():
        t = GatewayTool(
            name=spec["name"], description=spec["description"], args_schema=_permissive_args(spec),
            cache_function=lambda *_a, **_k: False,
        )
        t._gateway = gateway
        t._state = state
        tools.append(t)
    return tools


def _split_response(resp: Any) -> tuple[str, list | None]:
    """CrewAI hands us a string, or tool call objects when the model used native function calling."""
    if resp is None:
        return "", None
    if isinstance(resp, str):
        return resp, None
    if isinstance(resp, list):
        calls = []
        for c in resp:
            fn = getattr(c, "function", None) or (c.get("function") if isinstance(c, dict) else None)
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
            if name:
                calls.append({"name": name, "args": args})
        return "", calls or None
    return text_of(resp) if hasattr(resp, "content") else str(resp), None


_PROVIDER_HINTS = ("api key", "api_key", "401", "403", "404", "429", "quota", "rate limit", "resource_exhausted",
                   "connection", "resolve", "nodename", "timeout", "timed out", "overloaded", "503", "502", "unavailable")


def _is_provider_error(e: Exception) -> bool:
    mod = type(e).__module__.split(".")[0]
    s = str(e).lower()
    return mod in ("openai", "google", "httpx", "httpcore", "litellm", "grpc") or any(h in s for h in _PROVIDER_HINTS)


def run_crewai_agent(llm, gateway: ToolGateway, collector: TraceCollector, task, backoff_s: float = 3.0) -> dict:
    agent_id = "agent"
    start = collector.emit("run_start", agent_id=agent_id, content=task.task_id)
    state = SimpleNamespace(steps=0, last_event=start.event_id, last_req=start.event_id, limit_hit=False)

    def before(ctx):
        if state.steps >= task.maximum_steps:
            state.limit_hit = True
            ev = collector.emit("error", agent_id=agent_id, status="error",
                                error={"code": "MAX_STEPS_EXCEEDED", "message": f"Step limit {task.maximum_steps} reached"})
            state.last_event = ev.event_id
            raise StepLimitExceeded()
        state.steps += 1
        req = collector.emit("llm_request", parent=state.last_event, agent_id=agent_id,
                             content=f"messages={len(ctx.messages)}", meta={"step": state.steps})
        state.last_req = req.event_id
        return None

    def after(ctx):
        text, calls = _split_response(ctx.response)
        ev = collector.emit("llm_response", parent=state.last_req, agent_id=agent_id, content=text,
                            arguments=calls, status="ok")
        state.last_event = ev.event_id
        return None

    tools = build_tools(gateway, state)
    agent = Agent(
        role="Autonomous task agent",
        goal="Complete the assigned task using the provided tools.",
        backstory=SYSTEM_PROMPT,
        llm=llm, tools=tools, verbose=False, allow_delegation=False, cache=False,
        max_iter=task.maximum_steps + 5, max_retry_limit=0, respect_context_window=True,
    )
    crew_task = CrewTask(description=build_task_prompt(task), expected_output="A short final summary of what was done.",
                         agent=agent)
    crew = Crew(agents=[agent], tasks=[crew_task], process=Process.sequential, verbose=False, cache=False,
                memory=False, tracing=False)

    register_before_llm_call_hook(before)
    register_after_llm_call_hook(after)
    # Switch off both CrewAI repeat guards: the text mode one (blocks an identical call) and the executor one
    # (force-ends the run after a streak of identical calls). Both would change how an agent can retry.
    guards = [
        mock.patch.object(ToolUsage, "_check_tool_repeated_usage", lambda self, calling: False),
        mock.patch("crewai.agents.agent_executor.repeated_tool_call_limit_reached", lambda counts: False),
    ]
    for g in guards:
        g.start()
    try:
        result = crew.kickoff()
        final = (getattr(result, "raw", None) or "").strip() if result is not None else ""
        fa = collector.emit("final_answer", parent=state.last_event, agent_id=agent_id, content=final)
        state.last_event = fa.event_id
        return {"status": "completed", "steps": state.steps, "final_answer": final}
    except Exception as e:  # noqa: BLE001
        if state.limit_hit or isinstance(e, StepLimitExceeded) or isinstance(getattr(e, "__cause__", None), StepLimitExceeded):
            return {"status": "max_steps", "steps": state.steps, "final_answer": None}
        if _is_provider_error(e):
            collector.emit("error", parent=state.last_req, agent_id=agent_id, status="error",
                           error={"code": "PROVIDER_ERROR", "message": str(e)[:300]})
            return {"status": "provider_error", "steps": state.steps, "final_answer": None}
        raise  # unknown crash: the controller records RUN_CRASHED and keeps the partial trace
    finally:
        for g in guards:
            g.stop()
        unregister_before_llm_call_hook(before)
        unregister_after_llm_call_hook(after)
