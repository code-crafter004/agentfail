"""LangGraph adapter: single agent, plan-act-observe loop.

Graph:  agent --(tool calls?)--> tools --> agent ... --> END
Every LLM turn and every tool call is logged by us (not by framework callbacks),
so the trace format is identical across frameworks.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.common import invoke_with_retry, text_of
from agents.prompts import SYSTEM_PROMPT, build_task_prompt
from tools.gateway import ToolGateway
from tracing.collector import TraceCollector


class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    status: str  # running | completed | max_steps | provider_error
    last_event: str


def run_langgraph_agent(model, gateway: ToolGateway, collector: TraceCollector, task, backoff_s: float = 3.0) -> dict:
    llm = model.bind_tools(gateway.openai_tools())
    agent_id = "agent"

    def agent_node(state: State) -> dict:
        if state["steps"] >= task.maximum_steps:
            ev = collector.emit("error", agent_id=agent_id, status="error",
                                error={"code": "MAX_STEPS_EXCEEDED", "message": f"Step limit {task.maximum_steps} reached"})
            return {"status": "max_steps", "last_event": ev.event_id}
        req = collector.emit("llm_request", parent=state["last_event"], agent_id=agent_id,
                             content=f"messages={len(state['messages'])}", meta={"step": state["steps"] + 1})
        try:
            resp = invoke_with_retry(llm, state["messages"], collector, req.event_id, task.maximum_retries, backoff_s, agent_id)
        except Exception as e:  # noqa: BLE001
            ev = collector.emit("error", parent=req.event_id, agent_id=agent_id, status="error",
                                error={"code": "PROVIDER_ERROR", "message": str(e)[:300]})
            return {"status": "provider_error", "last_event": ev.event_id}
        calls = list(getattr(resp, "tool_calls", None) or [])
        finish = (getattr(resp, "response_metadata", None) or {}).get("finish_reason")
        out = collector.emit(
            "llm_response", parent=req.event_id, agent_id=agent_id, content=text_of(resp),
            arguments=[{"name": c["name"], "args": c["args"]} for c in calls] or None, status="ok",
            meta={"finish_reason": str(finish)} if finish else {},
        )
        update: dict[str, Any] = {"messages": [resp], "steps": state["steps"] + 1, "last_event": out.event_id}
        if not calls:
            fa = collector.emit("final_answer", parent=out.event_id, agent_id=agent_id, content=text_of(resp))
            update.update(status="completed", last_event=fa.event_id)
        return update

    def tools_node(state: State) -> dict:
        last = state["messages"][-1]
        msgs = []
        for c in last.tool_calls:
            payload = gateway.call(c["name"], c["args"], parent_event_id=state["last_event"])
            msgs.append(ToolMessage(content=json.dumps(payload, default=str), tool_call_id=c["id"], name=c["name"]))
        return {"messages": msgs}

    def route(state: State) -> str:
        if state["status"] != "running":
            return END
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    app = g.compile()

    start = collector.emit("run_start", agent_id=agent_id, content=task.task_id)
    init: State = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=build_task_prompt(task))],
        "steps": 0,
        "status": "running",
        "last_event": start.event_id,
    }
    final = app.invoke(init, {"recursion_limit": task.maximum_steps * 3 + 10})
    answer = ""
    for m in reversed(final["messages"]):
        if m.type == "ai":
            answer = text_of(m)
            break
    return {"status": final["status"], "steps": final["steps"], "final_answer": answer if final["status"] == "completed" else None}
