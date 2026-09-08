"""
sim_tool.py — OmniCell-AI Phase 3 | Counterfactual Bio-Twin Simulator Tool
─────────────────────────────────────────────────────────────────────────
Exposes the digital twin simulation environment as a LangChain @tool so that
the Biologist and Process Engineer agents can test hypothetical pump and
nutrient adjustments in a virtual sandbox before applying actions to hardware.

Tool: simulate_counterfactual_intervention
Predicts:
  - Final Biomass concentration (g/L)
  - Glucose depletion / accumulation (g/L)
  - Lactate byproduct concentration (mmol/L)
  - Critical Quality Attribute (CQA) safety compliance
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from langchain_core.tools import tool

# Attempt to locate and import 2_simulation_env if available
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "2_simulation_env"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)


def _run_twin_simulation(
    initial_biomass: float,
    initial_glucose: float,
    initial_lactate: float,
    glucose_feed_rate: float,
    base_feed_rate: float,
    trace_feed_rate: float,
    horizon_hours: float = 2.0,
    has_metabolic_drift: bool = True,
) -> dict[str, Any]:
    """
    Executes a forward Euler dFBA simulation over the specified horizon.
    Uses BioreactorTwinEnv if available; falls back to an exact analytical
    kinetic model of E. coli overflow metabolism.
    """
    try:
        from cobra_metabolism import BiologicalEngine
        engine = BiologicalEngine("textbook")
        if has_metabolic_drift and trace_feed_rate <= 0.005:
            engine.induce_metabolic_drift("PDH", knock_down_fraction=0.15)
        elif trace_feed_rate > 0.005:
            engine.reset_metabolic_drift()
        use_cobra = True
    except Exception:
        use_cobra = False

    # Simulation constants
    dt = 0.1  # 6-minute step
    steps = max(1, int(horizon_hours / dt))
    
    # State vectors
    X = float(initial_biomass)
    S = float(initial_glucose)
    L = float(initial_lactate)
    
    V = 1.0  # Liter
    feed_conc = 400.0  # g/L in feed reservoir
    
    # Clamp input rates to realistic pump ranges [0.0, 0.50] L/h
    f_glc = max(0.0, min(0.50, float(glucose_feed_rate)))
    f_trace = max(0.0, min(0.02, float(trace_feed_rate)))

    trajectory = []
    
    for step_i in range(steps):
        if use_cobra:
            max_uptake = 10.0 * (S / (0.5 + max(0.0, S)))
            constraints = {
                "EX_glc__D_e": max_uptake,
                "EX_o2_e": 100.0,
            }
            rates = engine.solve_fba(constraints)
            mu = rates["mu"]
            q_s = rates["q_glucose"]
            q_l = rates["q_lactate"]
        else:
            # Analytical Michaelis-Menten kinetics fallback
            mu_max = 0.55 if f_trace > 0.005 or not has_metabolic_drift else 0.20
            Ks = 0.5  # g/L
            mu = mu_max * (S / (Ks + S)) if S > 0.01 else 0.0
            q_s = (mu / 0.45) + 0.02  # Yield Y_x/s = 0.45
            # Overflow lactate generation when glucose is high and PDH is hindered
            if has_metabolic_drift and f_trace <= 0.005 and S > 10.0:
                q_l = 0.85 * (S / (1.0 + S))
            else:
                q_l = 0.0 if f_trace > 0.005 else 0.05

        # Differential Mass Balance
        dX = (mu * X - (f_glc / V) * X) * dt
        dS = (-q_s * X + (f_glc / V) * (feed_conc - S)) * dt
        dL = (q_l * X - (f_glc / V) * L) * dt

        X = max(0.0, min(50.0, X + dX))
        S = max(0.0, min(100.0, S + dS))
        L = max(0.0, min(50.0, L + dL))

        trajectory.append({
            "step": step_i + 1,
            "time_h": round((step_i + 1) * dt, 2),
            "biomass_gL": round(X, 3),
            "glucose_gL": round(S, 3),
            "lactate_mmolL": round(L, 3),
        })

    # cGMP Compliance check: QbD threshold for Lactate is typically 2.0 - 5.0 mmol/L
    qbd_compliant = L <= 2.0
    lactate_trend = "decreasing" if L < initial_lactate else ("stable" if abs(L - initial_lactate) < 0.1 else "increasing")

    return {
        "horizon_hours": horizon_hours,
        "final_biomass_gL": round(X, 3),
        "final_glucose_gL": round(S, 3),
        "final_lactate_mmolL": round(L, 3),
        "biomass_growth_pct": round(((X - initial_biomass) / max(0.01, initial_biomass)) * 100, 2),
        "lactate_trend": lactate_trend,
        "qbd_compliant": qbd_compliant,
        "summary": (
            f"Simulated {horizon_hours}h lookahead: Biomass {initial_biomass:.2f}->{X:.2f} g/L, "
            f"Lactate {initial_lactate:.2f}->{L:.2f} mmol/L ({lactate_trend}), "
            f"QbD Compliant: {qbd_compliant}"
        ),
    }


@tool
def simulate_counterfactual_intervention(
    action_type: str,
    glucose_feed_rate: float = 0.10,
    trace_feed_rate: float = 0.0,
    initial_lactate: float = 3.0,
    initial_biomass: float = 2.0,
    initial_glucose: float = 15.0,
    horizon_hours: float = 2.0,
) -> str:
    """
    Counterfactual Bio-Twin simulation tool for the OmniCell-AI Swarm.

    Simulates the biological impact of proposed control actions on the Digital Twin
    (CobraPy dFBA + Euler physics) BEFORE applying them to physical hardware.

    Args:
        action_type: Name of the proposed action (e.g., 'trace_pump_on', 'reduce_feed_rate', 'nominal_hold').
        glucose_feed_rate: Proposed volumetric glucose feed rate in L/h [0.0 to 0.50].
        trace_feed_rate: Proposed trace nutrient / zinc feed rate in L/h [0.0 to 0.02].
        initial_lactate: Current observed lactate in mmol/L.
        initial_biomass: Current observed biomass in g/L.
        initial_glucose: Current observed glucose in g/L.
        horizon_hours: Virtual lookahead duration in hours (e.g., 2.0).

    Returns:
        JSON string with simulated trajectory metrics (biomass, glucose, lactate, QbD compliance).
    """
    # Map semantic action keywords to physical flow adjustments
    act_lower = action_type.lower()
    g_feed = glucose_feed_rate
    t_feed = trace_feed_rate

    if "trace_pump_on" in act_lower or "trace" in act_lower or "zinc" in act_lower:
        t_feed = max(0.01, t_feed)
        g_feed = min(0.20, g_feed)
    elif "reduce_feed" in act_lower or "cut_feed" in act_lower:
        g_feed = max(0.02, g_feed * 0.5)
    elif "flush" in act_lower:
        g_feed = 0.0
        t_feed = 0.01

    results = _run_twin_simulation(
        initial_biomass=initial_biomass,
        initial_glucose=initial_glucose,
        initial_lactate=initial_lactate,
        glucose_feed_rate=g_feed,
        base_feed_rate=0.02,
        trace_feed_rate=t_feed,
        horizon_hours=horizon_hours,
        has_metabolic_drift=True,
    )
    
    results["action_evaluated"] = action_type
    results["applied_glucose_feed_Lh"] = g_feed
    results["applied_trace_feed_Lh"] = t_feed

    return json.dumps(results, indent=2)
