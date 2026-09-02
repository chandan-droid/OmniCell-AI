"""
swarm.py — OmniCell-AI Phase 3 | LangGraph Diagnostic Swarm
─────────────────────────────────────────────────────────────
Defines the multi-agent StateGraph:

  ┌──────────────┐    symptom found    ┌────────────────┐
  │  Supervisor  │ ─────────────────── │    Biologist   │
  │  (Router)    │                     │  (LLM + Tool)  │
  └──────────────┘                     └────────────────┘
         │  no anomaly
         ▼
        END

State:
  - telemetry          (dict)  Raw sensor payload from Kafka
  - symptom            (str)   Detected symptom name, or ""
  - diagnosis          (dict)  Neo4j GraphRAG result payload
  - recommended_action (str)   Executable action string

Usage:
  from swarm import build_graph
  app = build_graph()
  result = app.invoke({"telemetry": {...}})
"""

from __future__ import annotations

import json
import os
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from graph_tool import query_knowledge_graph

# ── LLM Configuration (xAI Grok via OpenAI-compatible API) ───────────────────────
# langchain-openai works unchanged — Grok mirrors the OpenAI REST spec.
_XAI_API_KEY  = os.getenv("XAI_API_KEY",  "")
_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
_XAI_MODEL    = os.getenv("XAI_MODEL",    "grok-2-1212")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Agent State
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Shared state threaded through the entire LangGraph execution."""
    telemetry:           dict    # Raw telemetry payload from Kafka
    symptom:             str     # Observed symptom name ("" if healthy)
    diagnosis:           dict    # GraphRAG result from Neo4j
    recommended_action:  str     # Executable action string (e.g. 'trace_pump_on')


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Thresholds (easily tuned without touching logic)
# ═══════════════════════════════════════════════════════════════════════════════

LACTATE_THRESHOLD_MMOL = float(os.getenv("LACTATE_THRESHOLD_MMOL", "2.0"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Supervisor Node (Triage Router)
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_node(state: AgentState) -> AgentState:
    """
    Inspects telemetry for known anomaly triggers.

    Current rules (priority order):
      1. lactate_mmolL > LACTATE_THRESHOLD_MMOL  → 'Lactate Spike'
      2. (extensible) glucose_gL > threshold      → 'Glucose Surge'

    Sets state["symptom"] if a rule fires; otherwise leaves it empty.
    """
    telemetry = state.get("telemetry", {})
    symptom   = ""

    lactate = telemetry.get("lactate_mmolL", 0.0)
    if lactate > LACTATE_THRESHOLD_MMOL:
        symptom = "Lactate Spike"
        print(f"  [Supervisor] Lactate={lactate:.3f} mmol/L > {LACTATE_THRESHOLD_MMOL} → flagging '{symptom}'")
    else:
        print(f"  [Supervisor] All vitals nominal (Lactate={lactate:.3f} mmol/L). No anomaly.")

    return {**state, "symptom": symptom}


def supervisor_router(state: AgentState) -> Literal["biologist", "__end__"]:
    """Conditional edge: route to Biologist if a symptom was flagged, else END."""
    return "biologist" if state.get("symptom") else "__end__"


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Biologist Node (LLM + GraphRAG Tool)
# ═══════════════════════════════════════════════════════════════════════════════

_BIOLOGIST_SYSTEM_PROMPT = """\
You are an expert process biologist embedded in an autonomous bioreactor control system.

Your ONLY job is to call the `query_knowledge_graph` tool with the provided symptom name
to retrieve the root-cause anomaly and corrective treatment from the Neo4j knowledge graph.

After calling the tool, parse the JSON result and update your response with:
  - The anomaly identified
  - The treatment name
  - The exact `action` string from the tool output (do NOT paraphrase it)

Be concise and factual. This is a safety-critical industrial system.
"""


def biologist_node(state: AgentState) -> AgentState:
    """
    LLM agent bound with the query_knowledge_graph tool.

    Passes state["symptom"] to the GraphRAG tool, parses the JSON result,
    and writes diagnosis + recommended_action back into state.
    """
    symptom = state.get("symptom", "")
    print(f"  [Biologist]  Querying knowledge graph for symptom: '{symptom}' …")

    # ── Build LLM pointed at xAI Grok endpoint
    llm   = ChatOpenAI(
        model    = _XAI_MODEL,
        api_key  = _XAI_API_KEY,
        base_url = _XAI_BASE_URL,
        temperature = 0,
    )
    tools = [query_knowledge_graph]
    llm_with_tools = llm.bind_tools(tools)

    # ── System + human messages
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=_BIOLOGIST_SYSTEM_PROMPT),
        HumanMessage(content=f"Symptom detected: '{symptom}'. Diagnose and recommend action."),
    ]

    # ── Agentic loop: let the LLM call the tool and process results
    from langchain_core.messages import AIMessage, ToolMessage

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Execute any tool calls the LLM requested
    diagnosis: dict         = {}
    recommended_action: str = ""

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            # Only call our registered tool
            if tool_call["name"] == "query_knowledge_graph":
                raw_result = query_knowledge_graph.invoke(tool_call["args"])

                # Append ToolMessage so the LLM sees the result
                messages.append(
                    ToolMessage(
                        content=raw_result,
                        tool_call_id=tool_call["id"],
                    )
                )

                # Parse structured payload
                payload = json.loads(raw_result)
                diagnosis = {
                    "anomaly":   payload.get("anomaly"),
                    "treatment": payload.get("treatment"),
                    "raw":       payload.get("raw", []),
                }
                recommended_action = payload.get("action", "")

                print(f"  [Biologist]  Anomaly   : {diagnosis.get('anomaly')}")
                print(f"  [Biologist]  Treatment : {diagnosis.get('treatment')}")
                print(f"  [Biologist]  Action    : {recommended_action}")

    return {
        **state,
        "diagnosis":          diagnosis,
        "recommended_action": recommended_action,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Graph Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> "CompiledStateGraph":
    """
    Assembles and compiles the LangGraph StateGraph.

    Topology:
      START → supervisor → (conditional) → biologist → END
                                        → END
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("biologist",  biologist_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Conditional edge from Supervisor
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "biologist": "biologist",
            "__end__":   END,
        },
    )

    # Biologist always terminates after one pass
    builder.add_edge("biologist", END)

    return builder.compile()
