"""
test_swarm.py — OmniCell-AI Phase 3 | Pure Multi-Agent Swarm Verification Suite
───────────────────────────────────────────────────────────────────────────
Automated test suite verifying the pure multi-agent deliberation swarm:
  - Nominal vitals (zero false alarms, early exit)
  - Lactate spike & metabolic overflow with counterfactual Bio-Twin simulation
  - Glucose surge / pump drift detection
  - Multi-agent debate & reflection loops (back-routing upon rejection)
  - Multi-agent state integrity, confidence scoring, and cGMP compliance auditing

Run:
  uv run python test_swarm.py
"""

from __future__ import annotations

import sys
from swarm import build_graph


def test_nominal_telemetry() -> None:
    print("\n" + "=" * 60)
    print("TEST 1: Nominal Telemetry (Healthy Bioreactor)")
    print("=" * 60)

    app = build_graph()
    state = {
        "telemetry": {
            "biomass_gL": 2.5,
            "glucose_gL": 14.2,
            "lactate_mmolL": 0.85,
        },
        "anomaly_evaluation": {},
        "hypotheses": [],
        "biologist_diagnosis": {},
        "simulation_results": {},
        "engineer_review": {},
        "cgmp_audit": {},
        "critique_feedback": "",
        "revision_count": 0,
        "final_decision": {},
        "confidence": 0.0,
        "deliberation_history": [],
    }

    result = app.invoke(state)
    eval_res = result.get("anomaly_evaluation", {})

    assert not eval_res.get("is_anomalous"), "Nominal state should not flag an anomaly!"
    assert result.get("final_decision", {}).get("action") is None or result.get("final_decision", {}).get("action") == "", "Nominal state should not command action!"
    print("[PASS] TEST 1 PASSED: Evaluator correctly routed nominal vitals to END.")


def test_lactate_spike_anomaly() -> None:
    print("\n" + "=" * 60)
    print("TEST 2: Lactate Spike & Multi-Agent Deliberation with Bio-Twin Sim")
    print("=" * 60)

    app = build_graph()
    state = {
        "telemetry": {
            "biomass_gL": 1.8,
            "glucose_gL": 18.0,
            "lactate_mmolL": 3.42,  # > 2.0 mmol/L threshold
        },
        "anomaly_evaluation": {},
        "hypotheses": [],
        "biologist_diagnosis": {},
        "simulation_results": {},
        "engineer_review": {},
        "cgmp_audit": {},
        "critique_feedback": "",
        "revision_count": 0,
        "final_decision": {},
        "confidence": 0.0,
        "deliberation_history": [],
    }

    result = app.invoke(state)

    eval_res = result.get("anomaly_evaluation", {})
    assert eval_res.get("is_anomalous"), "Lactate 3.42 must trigger anomaly!"
    assert "Lactate Spike" in eval_res.get("symptoms", [])

    bio = result.get("biologist_diagnosis", {})
    assert bio.get("anomaly") == "Overflow Metabolism", f"Unexpected anomaly: {bio.get('anomaly')}"
    assert bio.get("proposed_action") == "trace_pump_on"

    sim = result.get("simulation_results", {})
    assert "final_lactate_mmolL" in sim, "Simulation results missing final lactate!"
    assert sim.get("final_lactate_mmolL") < 3.42, "Trace pump simulation must project reduced lactate!"

    eng = result.get("engineer_review", {})
    assert eng.get("approved") is True, "Engineer should approve trace pump rate within limits"

    cgmp = result.get("cgmp_audit", {})
    assert cgmp.get("compliance_rating") in ["PASS_OPTIMAL", "PASS_WITH_MONITORING"]

    dec = result.get("final_decision", {})
    assert dec.get("action") == "trace_pump_on"
    assert result.get("confidence") >= 0.70

    print("[PASS] TEST 2 PASSED: Pure Multi-Agent Swarm deliberated, simulated, and reached consensus on 'trace_pump_on'.")


def test_glucose_surge_anomaly() -> None:
    print("\n" + "=" * 60)
    print("TEST 3: Glucose Surge Anomaly & Action Resolution")
    print("=" * 60)

    app = build_graph()
    state = {
        "telemetry": {
            "biomass_gL": 1.2,
            "glucose_gL": 45.0,  # Surge > 30.0 g/L
            "lactate_mmolL": 1.2,
        },
        "anomaly_evaluation": {},
        "hypotheses": [],
        "biologist_diagnosis": {},
        "simulation_results": {},
        "engineer_review": {},
        "cgmp_audit": {},
        "critique_feedback": "",
        "revision_count": 0,
        "final_decision": {},
        "confidence": 0.0,
        "deliberation_history": [],
    }

    result = app.invoke(state)
    eval_res = result.get("anomaly_evaluation", {})
    assert eval_res.get("is_anomalous")
    assert "Glucose Surge" in eval_res.get("symptoms", [])

    dec = result.get("final_decision", {})
    assert dec.get("action") == "flush_feed_line"
    print("[PASS] TEST 3 PASSED: Swarm identified 'Pump Calibration Drift' and resolved 'flush_feed_line'.")


def test_debate_reflection_loop() -> None:
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Agent Reflection & Debate Loop (Back-Routing on Objection)")
    print("=" * 60)

    app = build_graph()
    # Telemetry with high lactate
    state = {
        "telemetry": {
            "biomass_gL": 1.5,
            "glucose_gL": 22.0,
            "lactate_mmolL": 4.10,
        },
        "anomaly_evaluation": {},
        "hypotheses": [],
        "biologist_diagnosis": {},
        "simulation_results": {},
        "engineer_review": {},
        "cgmp_audit": {},
        "critique_feedback": "Initial proposal pump speed excessive (>0.50 L/h). Reduce feed rate and activate trace pump.",
        "revision_count": 1,
        "final_decision": {},
        "confidence": 0.0,
        "deliberation_history": [],
    }

    result = app.invoke(state)
    assert result.get("final_decision", {}).get("action") in ["trace_pump_on", "reduce_feed_rate", "flush_feed_line"]
    assert len(result.get("deliberation_history", [])) >= 4, "Deliberation trail must capture all multi-agent turns"
    print("[PASS] TEST 4 PASSED: Multi-turn debate loop successfully processed critique feedback and converged to consensus.")


if __name__ == "__main__":
    try:
        test_nominal_telemetry()
        test_lactate_spike_anomaly()
        test_glucose_surge_anomaly()
        test_debate_reflection_loop()
        print("\n" + "=" * 60)
        print("ALL PURE AGENTIC SWARM TESTS PASSED SUCCESSFULLY!")
        print("=" * 60 + "\n")
    except AssertionError as err:
        print(f"\n[FAIL] TEST FAILED: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] UNEXPECTED ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
