"""
A minimal, dependency-free LangGraph-style orchestrator.

If `langgraph` is installed the same node/edge definitions can be ported 1:1;
this implementation keeps the project runnable offline (and keeps the trace
first-class, which the dashboard displays as the agent's reasoning chain).

    graph = StateGraph()
    graph.add_node("intent", intent_node)
    graph.add_conditional_edges("intent", router, {...})
    app = graph.compile()
    state = app.invoke({"question": "...", "user_id": 1})
"""
from __future__ import annotations

import time
from typing import Callable

from common.logging_utils import get_logger

log = get_logger("graph")

END = "__end__"


class StateGraph:
    def __init__(self, name: str = "fintwin"):
        self.name = name
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Callable, dict]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Callable) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Callable, mapping: dict) -> "StateGraph":
        self.conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def compile(self) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("entry point not set")
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.g = graph

    def invoke(self, state: dict, max_steps: int = 12) -> dict:
        state = dict(state)
        state.setdefault("trace", [])
        state.setdefault("tool_calls", [])
        current = self.g.entry
        steps = 0
        while current and current != END and steps < max_steps:
            fn = self.g.nodes.get(current)
            if fn is None:
                raise ValueError(f"unknown node '{current}'")
            t0 = time.time()
            try:
                state = fn(state) or state
                status = "ok"
            except Exception as exc:                       # noqa: BLE001
                state.setdefault("errors", []).append(f"{current}: {exc}")
                status = f"error: {exc}"
                log.warning("node %s failed: %s", current, exc)
            state["trace"].append({
                "node": current, "status": status, "ms": round((time.time() - t0) * 1000, 1),
            })
            steps += 1
            if current in self.g.conditional:
                router, mapping = self.g.conditional[current]
                key = router(state)
                current = mapping.get(key, END)
            else:
                current = self.g.edges.get(current, END)
        state["steps"] = steps
        return state

    def mermaid(self) -> str:
        lines = ["graph TD"]
        for src, dst in self.g.edges.items():
            lines.append(f"  {src} --> {dst}")
        for src, (_, mapping) in self.g.conditional.items():
            for key, dst in mapping.items():
                lines.append(f"  {src} --{key}--> {dst}")
        return "\n".join(lines)
