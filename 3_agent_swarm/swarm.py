"""
swarm.py — OmniCell-AI Phase 3 | Pure Multi-Agent Deliberation & Reflection Swarm
─────────────────────────────────────────────────────────────────────────────
Orchestrates a collaborative, multi-persona StateGraph with true ReAct tool-binding,
autonomous LLM agent reasoning, and multi-turn reflection/debate loops.

Topology:
  ┌─────────────────┐
  │ Anomaly         │ ── (Nominal) ──► END
  │ Evaluator Agent │ ◄── Autonomous LLM Multivariate Perception
  └────────┬────────┘
           │ (Anomaly Detected)
           ▼
  ┌─────────────────┐ ◄───────────────────────────────┐
  │ Biologist Agent │ ◄── ReAct Loop: bind_tools(...) │ (Reflection / Debate Loop
  └────────┬────────┘                                 │  if Rejected by Engineer
           ▼                                          │  or cGMP Auditor)
  ┌─────────────────┐                                 │
  │ Bioprocess      │ ── (Rejected & Rev < 3) ────────┤
  │ Engineer Agent  │ ◄── LLM Hardware & Hydrodynamic │
  └────────┬────────┘     Safety Guardrail            │
           │ (Approved)                               │
           ▼                                          │
  ┌─────────────────┐                                 │
  │ cGMP Quality    │ ── (Non-compliant & Rev < 3) ───┘
  │ Auditor Agent   │ ◄── LLM FDA 21 CFR Part 11 / QbD Design Space Audit
  └────────┬────────┘
           │ (Approved)
           ▼
  ┌─────────────────┐
  │ Arbitrator Node │ ◄── LLM Consensus Synthesis & Action Directive Dispatch
  └────────┬────────┘
           ▼
          END
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from graph_tool import query_knowledge_graph, query_pathway_details
from sim_tool import simulate_counterfactual_intervention

# ── LLM Configuration ────────────────────────────────────────────────────────
_XAI_API_KEY    = os.getenv("XAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
_XAI_BASE_URL   = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
_XAI_MODEL      = os.getenv("XAI_MODEL", os.getenv("OPENAI_MODEL", "grok-2-1212"))

# ── Bioprocess Operating Boundaries (Reference Context for Agents) ────────────
LACTATE_THRESHOLD_MMOL = float(os.getenv("LACTATE_THRESHOLD_MMOL", "2.0"))
GLUCOSE_MIN_GL         = float(os.getenv("GLUCOSE_MIN_GL", "2.0"))
GLUCOSE_MAX_GL         = float(os.getenv("GLUCOSE_MAX_GL", "30.0"))
MAX_DEBATE_REVISIONS   = 3


def _get_llm(temperature: float = 0.0) -> ChatOpenAI | None:
    """Instantiates an LLM instance if an API key is configured with robust endpoint resolution."""
    api_key = os.getenv("XAI_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", ""))).strip()
    if not api_key:
        return None
    try:
        if api_key.startswith("gsk_"):
            base_url = "https://api.groq.com/openai/v1"
            model = os.getenv("XAI_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
            if "gpt-oss" in model or "grok" in model:
                model = "llama-3.3-70b-versatile"
        elif api_key.startswith("xai-"):
            base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
            model = os.getenv("XAI_MODEL", "grok-2-1212")
        else:
            base_url = os.getenv("OPENAI_BASE_URL", os.getenv("XAI_BASE_URL", "https://api.openai.com/v1"))
            model = os.getenv("OPENAI_MODEL", os.getenv("XAI_MODEL", "gpt-4o-mini"))

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "request_timeout": 12.0,
            "max_retries": 1,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    except Exception:
        return None


def _parse_llm_json(content: str) -> dict[str, Any] | None:
    """Extracts and parses JSON from an LLM response string."""
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        raw_json = match.group(1) if match else content.strip()
        return json.loads(raw_json)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Shared Swarm State
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Rich multi-agent state threaded through the entire deliberation lifecycle."""
    telemetry:            dict[str, Any]  # Sensor snapshot (biomass, glucose, lactate)
    anomaly_evaluation:   dict[str, Any]  # Multivariate triage assessment & flagged symptoms
    hypotheses:           list[str]       # Candidate biological & mechanical failure modes
    biologist_diagnosis:  dict[str, Any]  # Root-cause analysis + proposed remedy from Biologist
    simulation_results:   dict[str, Any]  # Lookahead digital twin counterfactual trajectory
    engineer_review:      dict[str, Any]  # Equipment limits & hydrodynamic feasibility review
    cgmp_audit:           dict[str, Any]  # FDA/QbD compliance scorecard & critical quality check
    critique_feedback:    str             # Rejection rationale triggering debate/revision
    revision_count:       int             # Counter to prevent infinite debate loops
    final_decision:       dict[str, Any]  # Synthesized execution directive
    confidence:           float           # Consensus confidence score [0.0 - 1.0]
    deliberation_history: list[str]       # Multi-agent audit trail log


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Node 1: Anomaly Evaluator Agent (LLM Multivariate Triage)
# ═══════════════════════════════════════════════════════════════════════════════

_EVALUATOR_PROMPT = """\
You are an expert bioprocess monitoring AI agent embedded in an autonomous bioreactor control system.
Analyze the provided multi-variate telemetry snapshot across metabolic and substrate parameters.

Standard reference baselines:
- Biomass (X): 0.1 to 10.0 g/L
- Glucose (S): 2.0 to 30.0 g/L (Optimal: 10 - 20 g/L)
- Lactate (L): < 2.0 mmol/L (CQA Threshold: 2.0 mmol/L, Toxic: > 5.0 mmol/L)

Evaluate whether the telemetry represents a nominal fermentation state or an emerging biological/mechanical anomaly.
Formulate 2-3 candidate failure hypotheses if anomalous.

Return ONLY a valid JSON object matching this schema:
{
  "is_anomalous": boolean,
  "symptoms": ["string"],
  "primary_symptom": "string (e.g. 'Lactate Spike', 'Glucose Surge', 'Substrate Starvation' or '')",
  "severity": "NOMINAL" | "WARNING" | "CRITICAL_CQA",
  "hypotheses": ["string"],
  "physiological_rationale": "string"
}
"""


def evaluator_node(state: AgentState) -> AgentState:
    """
    LLM-powered Evaluator Agent:
    Evaluates multivariate telemetry using deep bioprocess reasoning rather than rigid single-value thresholds.
    """
    telem = state.get("telemetry", {})
    history = list(state.get("deliberation_history", []))

    lactate = float(telem.get("lactate_mmolL", telem.get("lactate", 0.0)))
    glucose = float(telem.get("glucose_gL", telem.get("glucose", 15.0)))
    biomass = float(telem.get("biomass_gL", telem.get("biomass", 1.0)))

    llm = _get_llm(temperature=0.0)
    evaluation = None

    if llm:
        try:
            messages = [
                SystemMessage(content=_EVALUATOR_PROMPT),
                HumanMessage(content=f"Current Bioreactor Telemetry:\n{json.dumps(telem, indent=2)}"),
            ]
            response = llm.invoke(messages)
            evaluation = _parse_llm_json(response.content)
        except Exception as exc:
            print(f"  [Evaluator] LLM inference note: {exc}, using grounded bioprocess evaluation engine.")

    # Fallback to multivariate bioprocess heuristic model if LLM offline / unconfigured
    if not evaluation or not isinstance(evaluation, dict) or "is_anomalous" not in evaluation:
        flagged_symptoms = []
        severity = "NOMINAL"
        hypotheses = []

        if lactate > LACTATE_THRESHOLD_MMOL:
            flagged_symptoms.append("Lactate Spike")
            hypotheses.extend(["Overflow Metabolism (Warburg/Crabtree)", "PDH Enzyme Suppression", "Hypoxia Stress"])
            severity = "CRITICAL_CQA" if lactate > 3.0 else "WARNING"

        if glucose < GLUCOSE_MIN_GL:
            flagged_symptoms.append("Substrate Starvation")
            hypotheses.append("Feed Pump Blockage or Depletion")
            severity = "WARNING" if severity == "NOMINAL" else severity
        elif glucose > GLUCOSE_MAX_GL:
            flagged_symptoms.append("Glucose Surge")
            hypotheses.append("Pump Calibration Drift or Line Siphoning")
            severity = "WARNING" if severity == "NOMINAL" else severity

        evaluation = {
            "is_anomalous": len(flagged_symptoms) > 0,
            "symptoms": flagged_symptoms,
            "primary_symptom": flagged_symptoms[0] if flagged_symptoms else "",
            "severity": severity,
            "hypotheses": hypotheses,
            "physiological_rationale": f"Multivariate analysis: Lactate={lactate:.2f} mM, Glucose={glucose:.2f} g/L, Biomass={biomass:.2f} g/L",
        }

    evaluation["vitals_summary"] = f"Lactate={lactate:.2f} mM | Glucose={glucose:.2f} g/L | Biomass={biomass:.2f} g/L"

    log_entry = (
        f"[Evaluator] Vitals: {evaluation['vitals_summary']} -> "
        f"Status: {evaluation.get('severity')} (Flagged: {', '.join(evaluation.get('symptoms', [])) if evaluation.get('symptoms') else 'None'})"
    )
    print(f"  {log_entry}")
    history.append(log_entry)

    return {
        **state,
        "anomaly_evaluation": evaluation,
        "hypotheses": evaluation.get("hypotheses", []),
        "critique_feedback": "",
        "revision_count": 0,
        "deliberation_history": history,
    }


def evaluator_router(state: AgentState) -> Literal["biologist", "__end__"]:
    """Conditional edge: route to multi-agent deliberation if anomaly detected, else exit."""
    evaluation = state.get("anomaly_evaluation", {})
    return "biologist" if evaluation.get("is_anomalous", False) else "__end__"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Node 2: Biologist Agent (Pure ReAct Tool-Calling & Revision Loop)
# ═══════════════════════════════════════════════════════════════════════════════

_BIOLOGIST_SYSTEM_PROMPT = """\
You are an expert Process Biologist Agent embedded in an autonomous bioreactor control system.
You have full autonomy to inspect the failure knowledge graph and test candidate actions in the virtual Digital Twin.

Your available tools:
1. `query_knowledge_graph(symptom)`: Traverses the biological failure ontology in Neo4j.
2. `query_pathway_details(anomaly)`: Checks specific metabolic enzyme suppressions.
3. `simulate_counterfactual_intervention(action_type, ...)`: Runs a forward lookahead simulation on the CobraPy dFBA Digital Twin.

Workflow:
1. Call `query_knowledge_graph` to retrieve root-cause anomalies and candidate actions.
2. Call `simulate_counterfactual_intervention` to test if the candidate action resolves the anomaly in simulation.
3. Once satisfied with the simulation trajectory, output your final diagnosis in strict JSON:
{
  "anomaly": "string",
  "suppressed_enzyme": "string",
  "treatment": "string",
  "proposed_action": "string",
  "scientific_rationale": "string"
}
"""


def biologist_node(state: AgentState) -> AgentState:
    """
    Pure ReAct Biologist Agent:
    Autonomously invokes tools (Knowledge Graph traversal, Pathway check, Counterfactual Simulation)
    via LangChain bind_tools, and addresses critique feedback if in a revision loop.
    """
    evaluation = state.get("anomaly_evaluation", {})
    telem = state.get("telemetry", {})
    critique = state.get("critique_feedback", "")
    revision_count = state.get("revision_count", 0)
    history = list(state.get("deliberation_history", []))
    primary_symptom = evaluation.get("primary_symptom", "Lactate Spike")

    print(f"  [Biologist] Investigating root cause for symptom: '{primary_symptom}' (Revision #{revision_count}) ...")

    tools = [query_knowledge_graph, query_pathway_details, simulate_counterfactual_intervention]
    tools_by_name = {t.name: t for t in tools}

    llm = _get_llm(temperature=0.0)
    diagnosis: dict[str, Any] | None = None
    sim_results: dict[str, Any] = {}

    if llm:
        try:
            llm_with_tools = llm.bind_tools(tools)
            user_prompt = f"Observed symptom: '{primary_symptom}'. Telemetry: {json.dumps(telem)}"
            if critique:
                user_prompt += f"\n\nCRITICAL PEER REVIEW FEEDBACK (from previous turn):\n{critique}\nPlease adjust your diagnosis and proposed action to satisfy this requirement."

            messages: list[Any] = [
                SystemMessage(content=_BIOLOGIST_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            # Autonomous ReAct loop (up to 4 tool iterations)
            for _ in range(4):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        t_name = tool_call["name"]
                        t_args = tool_call["args"]
                        tool_fn = tools_by_name.get(t_name)
                        if tool_fn:
                            raw_out = tool_fn.invoke(t_args)
                            messages.append(ToolMessage(content=str(raw_out), tool_call_id=tool_call["id"]))
                            if t_name == "simulate_counterfactual_intervention":
                                try:
                                    sim_results = json.loads(raw_out)
                                except Exception:
                                    pass
                else:
                    diagnosis = _parse_llm_json(response.content)
                    break
        except Exception as exc:
            print(f"  [Biologist] ReAct execution note: {exc}, using grounded fallback reasoning.")

    # Grounded fallback execution if LLM offline / response unparsed
    if not diagnosis or not isinstance(diagnosis, dict) or "proposed_action" not in diagnosis:
        # Autonomous tool invocation
        raw_graph = query_knowledge_graph.invoke({"symptom": primary_symptom})
        graph_payload = json.loads(raw_graph)
        anomaly = graph_payload.get("anomaly", "Overflow Metabolism")
        treatment = graph_payload.get("treatment", "Activate Trace Pump")
        action = graph_payload.get("action", "trace_pump_on")
        suppressed_enzyme = graph_payload.get("suppressed_enzyme", "PDH Complex")

        lactate = float(telem.get("lactate_mmolL", telem.get("lactate", 3.0)))
        biomass = float(telem.get("biomass_gL", telem.get("biomass", 2.0)))
        glucose = float(telem.get("glucose_gL", telem.get("glucose", 15.0)))

        sim_raw = simulate_counterfactual_intervention.invoke({
            "action_type": action,
            "initial_lactate": lactate,
            "initial_biomass": biomass,
            "initial_glucose": glucose,
            "horizon_hours": 2.0,
        })
        sim_results = json.loads(sim_raw)

        diagnosis = {
            "anomaly": anomaly,
            "suppressed_enzyme": suppressed_enzyme,
            "treatment": treatment,
            "proposed_action": action,
            "scientific_rationale": (
                f"Observed {primary_symptom} indicates {anomaly}, suppressing {suppressed_enzyme}. "
                f"Proposing {treatment} (action: '{action}'). "
                f"Bio-Twin simulation projects lactate trend: {sim_results.get('lactate_trend')} "
                f"with final lactate at {sim_results.get('final_lactate_mmolL')} mmol/L."
            ),
        }

    log_biologist = (
        f"[Biologist] Root Cause: {diagnosis.get('anomaly')} ({diagnosis.get('suppressed_enzyme')}) | "
        f"Proposal: {diagnosis.get('proposed_action')} | Sim Outcome: {sim_results.get('summary', 'Simulated')}"
    )
    print(f"  {log_biologist}")
    history.append(log_biologist)

    return {
        **state,
        "biologist_diagnosis": diagnosis,
        "simulation_results": sim_results,
        "critique_feedback": "",
        "deliberation_history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Node 3: Bioprocess Engineer Agent (LLM Hardware & Hydrodynamics)
# ═══════════════════════════════════════════════════════════════════════════════

_ENGINEER_PROMPT = """\
You are an expert Bioprocess Automation & Hardware Engineer Agent.
Review the Biologist's proposed action and parameter adjustments against physical bioreactor equipment constraints:

Physical Boundaries:
- Main Glucose Feed Pump: [0.0 to 0.50 L/h] (Max slew rate: 0.10 L/h per step)
- Trace Nutrient / Cofactor Pump: [0.0 to 0.02 L/h]
- Base Buffer Pump: [0.0 to 0.10 L/h]
- Dilution Rate Washout Limit: D = F_total / V <= 0.50 h^-1 (Tank Volume V = 1.0 L)
- Hydrodynamic Shear Stress: Agitation acceleration must remain within cell viability bounds.

Evaluate whether to APPROVE or REJECT the proposed action. If rejecting, provide specific engineering feedback so the Biologist can revise.

Respond in strict JSON:
{
  "approved": boolean,
  "equipment_status": "NOMINAL" | "WARNING" | "LIMIT_EXCEEDED",
  "engineering_notes": "string",
  "critique_for_biologist": "string (empty if approved, detailed instructions if rejected)"
}
"""


def engineer_node(state: AgentState) -> AgentState:
    """
    LLM-powered Bioprocess Engineer Agent:
    Cognitively reviews physical pump capacities, dilution washout dynamics, and actuator safety.
    """
    diagnosis = state.get("biologist_diagnosis", {})
    action = diagnosis.get("proposed_action", "hold_current_state")
    history = list(state.get("deliberation_history", []))

    llm = _get_llm(temperature=0.0)
    review = None

    if llm:
        try:
            context = {
                "proposed_action": action,
                "biologist_diagnosis": diagnosis,
                "simulation_results": state.get("simulation_results", {}),
            }
            messages = [
                SystemMessage(content=_ENGINEER_PROMPT),
                HumanMessage(content=f"Biologist Proposal:\n{json.dumps(context, indent=2)}"),
            ]
            response = llm.invoke(messages)
            review = _parse_llm_json(response.content)
        except Exception as exc:
            print(f"  [Engineer] LLM review note: {exc}, using grounded engineering model.")

    if not review or not isinstance(review, dict) or "approved" not in review:
        # Grounded engineering safety rules
        approved = True
        notes = []
        if action == "trace_pump_on":
            notes.append("Trace cofactor feed commanded at 0.010 L/h (safe envelope [0.0 - 0.02 L/h]). Dilution rate D=0.01 h^-1.")
        elif action == "flush_feed_line":
            notes.append("Feed line flush authorized for 30s pulse.")
        elif action == "inject_base_bolus":
            notes.append("Base bolus restricted to 5 mL increments to prevent localized pH shock.")
        else:
            notes.append("Standard operating regime confirmed.")

        review = {
            "approved": approved,
            "equipment_status": "NOMINAL",
            "engineering_notes": "; ".join(notes),
            "critique_for_biologist": "",
        }

    status_str = "APPROVED" if review.get("approved") else "REJECTED"
    log_eng = f"[Engineer] Equipment Audit: {status_str} for action '{action}'. Notes: {review.get('engineering_notes')}"
    print(f"  {log_eng}")
    history.append(log_eng)

    critique = review.get("critique_for_biologist", "") if not review.get("approved") else ""
    rev_count = state.get("revision_count", 0) + (1 if not review.get("approved") else 0)

    return {
        **state,
        "engineer_review": review,
        "critique_feedback": critique,
        "revision_count": rev_count,
        "deliberation_history": history,
    }


def engineer_router(state: AgentState) -> Literal["cgmp_auditor", "biologist"]:
    """Conditional Edge: Proceed to Auditor if approved; loop back to Biologist if rejected."""
    review = state.get("engineer_review", {})
    rev_count = state.get("revision_count", 0)
    if review.get("approved", True) or rev_count >= MAX_DEBATE_REVISIONS:
        return "cgmp_auditor"
    print(f"  [Debate Loop] Engineer rejected proposal -> Routing back to Biologist (Revision {rev_count}/{MAX_DEBATE_REVISIONS})")
    return "biologist"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Node 4: cGMP Quality Auditor Agent (LLM FDA / QbD Compliance)
# ═══════════════════════════════════════════════════════════════════════════════

_CGMP_PROMPT = """\
You are an expert cGMP Quality Assurance & Regulatory Auditor Agent (FDA 21 CFR Part 11 / ICH Q8 QbD).
Review the Biologist's proposal, simulation lookahead, and Engineer's review against Critical Quality Attributes (CQAs):

Regulatory Design Space Rules:
- Broth Lactate CQA: Must remain <= 2.0 mmol/L (or demonstrate a continuous downward trajectory toward <= 2.0 mmol/L).
- Substrate Availability: Maintain glucose within 2.0 to 30.0 g/L to prevent culture starvation.
- Prevent irreversible product quality degradation (e.g. host cell protein contamination from cell lysis).

Evaluate whether the proposed strategy complies with Design Space parameters.

Respond in strict JSON:
{
  "approved": boolean,
  "cgmp_compliant": boolean,
  "compliance_rating": "PASS_OPTIMAL" | "PASS_WITH_MONITORING" | "NON_CONFORMING",
  "risk_grade": "LOW" | "MODERATE" | "CRITICAL",
  "regulatory_design_space": "WITHIN_DESIGN_SPACE" | "EXTENDED_MONITORING" | "OUT_OF_SPECIFICATION",
  "audit_feedback": "string",
  "critique_for_biologist": "string (empty if approved, regulatory objection if rejected)"
}
"""


def cgmp_auditor_node(state: AgentState) -> AgentState:
    """
    LLM-powered cGMP Quality Auditor Agent:
    Evaluates regulatory compliance and Critical Quality Attributes under FDA / QbD standards.
    """
    sim_results = state.get("simulation_results", {})
    diagnosis = state.get("biologist_diagnosis", {})
    history = list(state.get("deliberation_history", []))

    llm = _get_llm(temperature=0.0)
    audit = None

    if llm:
        try:
            context = {
                "diagnosis": diagnosis,
                "simulation_results": sim_results,
                "engineer_review": state.get("engineer_review", {}),
            }
            messages = [
                SystemMessage(content=_CGMP_PROMPT),
                HumanMessage(content=f"Deliberation Dossier:\n{json.dumps(context, indent=2)}"),
            ]
            response = llm.invoke(messages)
            audit = _parse_llm_json(response.content)
        except Exception as exc:
            print(f"  [cGMP Auditor] LLM audit note: {exc}, using grounded regulatory model.")

    if not audit or not isinstance(audit, dict) or "approved" not in audit:
        final_lactate = sim_results.get("final_lactate_mmolL", 0.0)
        qbd_compliant = sim_results.get("qbd_compliant", True)

        if final_lactate <= 2.0:
            rating = "PASS_OPTIMAL"
            risk = "LOW"
            approved = True
        elif final_lactate <= 5.0:
            rating = "PASS_WITH_MONITORING"
            risk = "MODERATE"
            approved = True
        else:
            rating = "NON_CONFORMING"
            risk = "CRITICAL"
            approved = False

        audit = {
            "approved": approved,
            "cgmp_compliant": qbd_compliant,
            "compliance_rating": rating,
            "risk_grade": risk,
            "regulatory_design_space": "WITHIN_DESIGN_SPACE" if qbd_compliant else "EXTENDED_MONITORING",
            "audit_feedback": f"Predicted Lactate: {final_lactate:.2f} mmol/L (CQA limit <= 2.0)",
            "critique_for_biologist": "" if approved else "Simulation shows lactate exceeds 5.0 mM toxic limit. Propose a more aggressive feed reduction.",
        }

    status_str = "APPROVED" if audit.get("approved") else "NON_CONFORMING"
    log_cgmp = (
        f"[cGMP Auditor] CQA Evaluation: {audit.get('compliance_rating')} (Risk: {audit.get('risk_grade')}, Status: {status_str}) | "
        f"{audit.get('audit_feedback')} | Design Space: {audit.get('regulatory_design_space')}"
    )
    print(f"  {log_cgmp}")
    history.append(log_cgmp)

    critique = audit.get("critique_for_biologist", "") if not audit.get("approved") else ""
    rev_count = state.get("revision_count", 0) + (1 if not audit.get("approved") else 0)

    return {
        **state,
        "cgmp_audit": audit,
        "critique_feedback": critique,
        "revision_count": rev_count,
        "deliberation_history": history,
    }


def cgmp_router(state: AgentState) -> Literal["arbitrator", "biologist"]:
    """Conditional Edge: Proceed to Arbitrator if approved; loop back to Biologist if non-conforming."""
    audit = state.get("cgmp_audit", {})
    rev_count = state.get("revision_count", 0)
    if audit.get("approved", True) or rev_count >= MAX_DEBATE_REVISIONS:
        return "arbitrator"
    print(f"  [Compliance Loop] cGMP Auditor rejected proposal -> Routing back to Biologist (Revision {rev_count}/{MAX_DEBATE_REVISIONS})")
    return "biologist"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Node 5: Executive Arbitrator (LLM Consensus Synthesis)
# ═══════════════════════════════════════════════════════════════════════════════

_ARBITRATOR_PROMPT = """\
You are the Executive Bioprocess Arbitrator Node.
Synthesize the multi-agent deliberation between the Biologist, Process Engineer, and cGMP Auditor into a final executable directive.

Respond in strict JSON:
{
  "action": "string",
  "anomaly": "string",
  "treatment": "string",
  "confidence": float (0.0 to 1.0),
  "consensus_status": "UNANIMOUS_CONSENSUS" | "CONDITIONAL_APPROVAL" | "EMERGENCY_OVERRIDE",
  "synthesis_summary": "string"
}
"""


def arbitrator_node(state: AgentState) -> AgentState:
    """
    Executive Arbitrator Persona:
      Synthesizes the multi-agent deliberation into a unified, confidence-scored directive.
    """
    diagnosis = state.get("biologist_diagnosis", {})
    engineer = state.get("engineer_review", {})
    cgmp = state.get("cgmp_audit", {})
    sim = state.get("simulation_results", {})
    history = list(state.get("deliberation_history", []))

    action = diagnosis.get("proposed_action", "hold_current_state")
    eng_ok = engineer.get("approved", True)
    cgmp_ok = cgmp.get("approved", True)

    llm = _get_llm(temperature=0.0)
    arb_decision = None

    if llm:
        try:
            context = {
                "biologist_diagnosis": diagnosis,
                "engineer_review": engineer,
                "cgmp_audit": cgmp,
                "simulation_results": sim,
                "revision_count": state.get("revision_count", 0),
            }
            messages = [
                SystemMessage(content=_ARBITRATOR_PROMPT),
                HumanMessage(content=f"Deliberation Summary Dossier:\n{json.dumps(context, indent=2)}"),
            ]
            response = llm.invoke(messages)
            arb_decision = _parse_llm_json(response.content)
        except Exception as exc:
            print(f"  [Arbitrator] LLM synthesis note: {exc}, using structured consensus engine.")

    if not arb_decision or not isinstance(arb_decision, dict) or "action" not in arb_decision:
        confidence = 0.95 if (eng_ok and cgmp_ok) else (0.75 if eng_ok else 0.40)
        status = "UNANIMOUS_CONSENSUS" if (eng_ok and cgmp_ok) else "CONDITIONAL_APPROVAL"
        arb_decision = {
            "action": action,
            "anomaly": diagnosis.get("anomaly", "Unknown"),
            "treatment": diagnosis.get("treatment", "Unknown"),
            "confidence": confidence,
            "consensus_status": status,
            "synthesis_summary": diagnosis.get("scientific_rationale", ""),
        }

    final_decision = {
        "action": arb_decision.get("action", action),
        "anomaly": arb_decision.get("anomaly", diagnosis.get("anomaly", "Unknown")),
        "treatment": arb_decision.get("treatment", diagnosis.get("treatment", "Unknown")),
        "confidence": float(arb_decision.get("confidence", 0.90)),
        "consensus_status": arb_decision.get("consensus_status", "APPROVED"),
        "rationale": arb_decision.get("synthesis_summary", ""),
        "simulated_trajectory": sim.get("summary", ""),
        "execution_payload": {
            "action_command": arb_decision.get("action", action),
            "trace_feed_Lh": sim.get("applied_trace_feed_Lh", 0.01),
            "glucose_feed_Lh": sim.get("applied_glucose_feed_Lh", 0.10),
            "authorized_by": ["Biologist", "Process_Engineer", "cGMP_Auditor"],
        },
    }

    log_arb = (
        f"[Arbitrator] Consensus Reached ({final_decision['consensus_status']}, Confidence: {final_decision['confidence']*100:.0f}%). "
        f"DISPATCHING ACTION: >> {final_decision['action'].upper()} <<"
    )
    print(f"  {log_arb}")
    history.append(log_arb)

    return {
        **state,
        "final_decision": final_decision,
        "confidence": final_decision["confidence"],
        "deliberation_history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Multi-Agent StateGraph Assembly (with Reflection / Debate Loops)
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> Any:
    """
    Assembles and compiles the pure Multi-Agent Deliberation & Reflection StateGraph.

    Topology:
      START → evaluator → (conditional: anomaly?)
                               ├── True  ──► biologist ◄─────────────┐
                               │                 │                   │
                               │                 ▼                   │ (Re-try / Debate)
                               │              engineer ──────────────┤
                               │                 │ (Approved)        │
                               │                 ▼                   │
                               │             cgmp_auditor ───────────┘
                               │                 │ (Approved)
                               │                 ▼
                               │             arbitrator
                               │                 │
                               │                 ▼
                               └── False ──►    END
    """
    builder = StateGraph(AgentState)

    # Register Nodes
    builder.add_node("evaluator",    evaluator_node)
    builder.add_node("biologist",    biologist_node)
    builder.add_node("engineer",     engineer_node)
    builder.add_node("cgmp_auditor", cgmp_auditor_node)
    builder.add_node("arbitrator",   arbitrator_node)

    # Entry Point
    builder.set_entry_point("evaluator")

    # Conditional Routing from Evaluator (Triage)
    builder.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "biologist": "biologist",
            "__end__":   END,
        },
    )

    # Biologist routes to Engineer
    builder.add_edge("biologist", "engineer")

    # Engineer routes to cGMP Auditor (if approved) or loops back to Biologist (if rejected)
    builder.add_conditional_edges(
        "engineer",
        engineer_router,
        {
            "cgmp_auditor": "cgmp_auditor",
            "biologist":    "biologist",
        },
    )

    # cGMP Auditor routes to Arbitrator (if approved) or loops back to Biologist (if non-conforming)
    builder.add_conditional_edges(
        "cgmp_auditor",
        cgmp_router,
        {
            "arbitrator": "arbitrator",
            "biologist":  "biologist",
        },
    )

    # Arbitrator exits
    builder.add_edge("arbitrator", END)

    return builder.compile()
