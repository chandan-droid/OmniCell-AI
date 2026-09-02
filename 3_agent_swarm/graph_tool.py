"""
graph_tool.py — OmniCell-AI Phase 3 | GraphRAG Tool
─────────────────────────────────────────────────────
Exposes the Neo4j knowledge graph as a LangChain @tool so the
Biologist agent can perform structured GraphRAG retrieval.

Tool:  query_knowledge_graph(symptom: str) -> dict
Query: Traverses (Symptom)-[:INDICATES]->(Anomaly)-[:SUPPRESSES]->(Enzyme)<-[:RESTORES]-(Treatment)
       and returns the anomaly name, treatment name, and executable action string.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool
from neo4j import GraphDatabase

# ── Neo4j connection (resolved from environment with safe defaults) ─────────────
_NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "BioprocessSecurePassword2026")

# Cypher query mandated by the architecture spec
_CYPHER = """
MATCH (s:Symptom {name: $symptom})-[:INDICATES]->(a:Anomaly)-[:SUPPRESSES]->(e)<-[:RESTORES]-(t:Treatment)
RETURN a.name AS anomaly, t.name AS treatment, t.action AS action
"""


def _run_query(symptom: str) -> list[dict[str, Any]]:
    """Execute the GraphRAG Cypher query and return raw records."""
    driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(_CYPHER, symptom=symptom)
            return [record.data() for record in result]
    finally:
        driver.close()


@tool
def query_knowledge_graph(symptom: str) -> str:
    """
    GraphRAG retrieval tool for the OmniCell-AI diagnostic swarm.

    Queries the Neo4j bioreactor failure knowledge graph to identify the root-cause
    Anomaly and recommended Treatment for a given observed Symptom.

    Args:
        symptom: The observed bioreactor symptom name (e.g., 'Lactate Spike').

    Returns:
        A JSON string containing:
          - anomaly:   Root-cause anomaly identified in the knowledge graph.
          - treatment: Name of the recommended corrective treatment.
          - action:    Executable action command string (e.g., 'trace_pump_on').
          - raw:       Full list of matched records from Neo4j.

    Example output:
        {
          "anomaly":   "Overflow Metabolism",
          "treatment": "Reduce Feed Pump Rate",
          "action":    "trace_pump_on",
          "raw":       [{"anomaly": "Overflow Metabolism", ...}]
        }
    """
    records = _run_query(symptom)

    if not records:
        return json.dumps({
            "anomaly":   None,
            "treatment": None,
            "action":    None,
            "raw":       [],
            "error":     f"No knowledge graph entry found for symptom: '{symptom}'",
        })

    # Surface the primary (first) match — graph seeding ensures 1:1 for known symptoms
    primary = records[0]
    return json.dumps({
        "anomaly":   primary.get("anomaly"),
        "treatment": primary.get("treatment"),
        "action":    primary.get("action"),
        "raw":       records,
    })
