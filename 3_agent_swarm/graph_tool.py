"""
graph_tool.py — OmniCell-AI Phase 3 | GraphRAG Multi-Hop Retrieval Tool
───────────────────────────────────────────────────────────────────────
Exposes the Neo4j bioprocess failure knowledge graph as LangChain @tools so
the Biologist and Quality agents can perform structured GraphRAG retrieval
and multi-hop pathway exploration.

Tools:
  - query_knowledge_graph(symptom: str)
  - query_pathway_details(anomaly: str)
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

# In-memory graph ontology fallback (mirrors neo4j_seeder.py) for offline SIL testing
_FALLBACK_ONTOLOGY = {
    "Lactate Spike": [
        {
            "anomaly": "Overflow Metabolism",
            "suppressed_enzyme": "Lactate Dehydrogenase / PDH",
            "treatment": "Activate Trace Pump & Modulate Feed",
            "action": "trace_pump_on",
            "risk_level": "CRITICAL_CQA",
            "rationale": "Accumulation of toxic lactate inhibits cell division and acidifies the broth. Trace zinc/cofactor restores PDH flux."
        },
        {
            "anomaly": "Hypoxia / Oxygen Mass Transfer Limit",
            "suppressed_enzyme": "Cytochrome c Oxidase",
            "treatment": "Increase Sparger Dissolved Oxygen",
            "action": "increase_do_sparge",
            "risk_level": "MODERATE",
            "rationale": "Insufficient oxygen shifts metabolism toward anaerobic glycolysis."
        }
    ],
    "Glucose Surge": [
        {
            "anomaly": "Pump Calibration Drift",
            "suppressed_enzyme": "Hexokinase",
            "treatment": "Flush Feed Line & Recalibrate",
            "action": "flush_feed_line",
            "risk_level": "HIGH",
            "rationale": "Excessive residual substrate risks hyperosmotic shock and severe overflow."
        }
    ],
    "pH Drop": [
        {
            "anomaly": "pH Probe Fouling",
            "suppressed_enzyme": "Carbonic Anhydrase",
            "treatment": "Inject Base Bolus & Recalibrate Sensor",
            "action": "inject_base_bolus",
            "risk_level": "CRITICAL_CQA",
            "rationale": "Broth acidification below pH 6.8 damages recombinant protein folding."
        }
    ],
    "Substrate Starvation": [
        {
            "anomaly": "Feed Line Blockage",
            "suppressed_enzyme": "Phosphofructokinase",
            "treatment": "Increase Main Glucose Pump",
            "action": "increase_feed_rate",
            "risk_level": "HIGH",
            "rationale": "Zero residual glucose initiates cell autophagy and protease release."
        }
    ],
}


def _run_query(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute Cypher query against Neo4j, returning empty list if offline."""
    try:
        driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]
    except Exception:
        # Fallback to local ontology if Neo4j is offline
        return []
    finally:
        try:
            driver.close()
        except Exception:
            pass


@tool
def query_knowledge_graph(symptom: str) -> str:
    """
    GraphRAG retrieval tool for the OmniCell-AI diagnostic swarm.

    Queries the Neo4j bioreactor failure knowledge graph to identify root-cause
    anomalies, suppressed enzymes, and approved corrective treatments.

    Args:
        symptom: The observed bioreactor symptom (e.g., 'Lactate Spike', 'Glucose Surge', 'pH Drop').

    Returns:
        JSON string containing primary diagnosis, candidate alternatives, and executable action.
    """
    cypher = """
    MATCH (s:Symptom)-[:INDICATES]->(a:Anomaly)-[:SUPPRESSES]->(e:Enzyme)<-[:RESTORES]-(t:Treatment)
    WHERE toLower(s.name) CONTAINS toLower($symptom) OR toLower($symptom) CONTAINS toLower(s.name)
    RETURN a.name AS anomaly, e.name AS suppressed_enzyme, t.name AS treatment, t.action AS action
    """
    records = _run_query(cypher, {"symptom": symptom})

    if not records:
        # Check fallback ontology
        for s_key, data in _FALLBACK_ONTOLOGY.items():
            if s_key.lower() in symptom.lower() or symptom.lower() in s_key.lower():
                records = data
                break

    if not records:
        return json.dumps({
            "anomaly": "Unknown Bioprocess Anomaly",
            "treatment": "Manual Expert Review",
            "action": "hold_current_state",
            "raw": [],
            "error": f"No knowledge graph entry found for symptom: '{symptom}'",
        })

    primary = records[0]
    return json.dumps({
        "anomaly": primary.get("anomaly"),
        "suppressed_enzyme": primary.get("suppressed_enzyme", "PDH Complex"),
        "treatment": primary.get("treatment"),
        "action": primary.get("action"),
        "candidate_alternatives": records[1:] if len(records) > 1 else [],
        "raw": records,
    }, indent=2)


@tool
def query_pathway_details(anomaly: str) -> str:
    """
    Retrieves cellular metabolic pathway and enzyme inhibition details for an anomaly.

    Args:
        anomaly: Name of the bioprocess anomaly (e.g., 'Overflow Metabolism', 'Pump Calibration Drift').
    """
    cypher = """
    MATCH (a:Anomaly)-[:SUPPRESSES]->(e:Enzyme)<-[:RESTORES]-(t:Treatment)
    WHERE toLower(a.name) CONTAINS toLower($anomaly)
    RETURN a.name AS anomaly, e.name AS enzyme, t.name AS treatment, t.action AS action
    """
    records = _run_query(cypher, {"anomaly": anomaly})

    if not records:
        return json.dumps({
            "anomaly": anomaly,
            "pathway_status": "Pyruvate Dehydrogenase (PDH) flux inhibition",
            "remedial_cofactor": "Trace Zinc / Mg2+",
            "kinetic_impact": "Diverts pyruvate away from TCA cycle into D-lactate fermentation.",
        }, indent=2)

    return json.dumps(records[0], indent=2)
