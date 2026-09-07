# OmniCell-AI Phase 3: Pure Multi-Agent Deliberation & Reflection Swarm

## Module Overview

The **Diagnostic Swarm** (`3_agent_swarm`) is the autonomous multi-agent clinical reasoning, counterfactual simulation, and cyber-physical governance layer of the OmniCell-AI bioprocess platform.

Rather than relying on linear pipelines or pre-ordained tool execution scripts, this module implements a **Pure Level-5 Multi-Agent Swarm** orchestrated via **LangGraph**:
1. **Autonomous Tool Binding (`bind_tools`)**: Agents decide *which* tools to call, inspect intermediate results, and iterate autonomously.
2. **Cognitive Multi-Persona Committee**: All agent personas (Evaluator, Biologist, Engineer, cGMP Auditor, Arbitrator) are powered by dedicated LLM system roles and schemas.
3. **Dynamic Reflection & Debate Loops**: When an engineering or regulatory objection arises, the graph back-routes to previous agents with structured critique dossiers until consensus is reached.

---

## 🏛️ Multi-Agent Swarm Architecture & Reflection Topology

```text
                     Kafka Telemetry Stream ('omnicell-telemetry')
                                       │
                                       ▼
                             [ watchdog.py ]
                         (Kafka Consumer Loop)
                                       │
                                       ▼
                          [ swarm.py : LangGraph ]
                        ┌─────────────────────────┐
                        │   evaluator_node        │  ◄── LLM Multivariate Perception & Anomaly Scoring
                        └───────────┬─────────────┘
                                    │
                       (Is Multivariate Anomaly?)
                                    ├── No ───► [ END / Nominal ]
                                    │
                                    └── Yes ──► Flags Symptoms
                                                  │
                                                  ▼
                        ┌─────────────────────────┐ ◄───────────────────────────────┐
                        │   biologist_node        │ ◄── Pure ReAct: bind_tools(...) │
                        └───────────┬─────────────┘     (GraphRAG + Bio-Twin Sim)   │ (Reflection /
                                    │                                               │  Debate Loop)
                                    ▼                                               │
                        ┌─────────────────────────┐                                 │
                        │   engineer_node         │ ── (Rejected: Rev < 3) ─────────┤
                        └───────────┬─────────────┘ ◄── LLM Hydrodynamic & Hardware │
                                    │ (Approved)        Guardrail                   │
                                    ▼                                               │
                        ┌─────────────────────────┐                                 │
                        │   cgmp_auditor_node     │ ── (Non-compliant: Rev < 3) ────┘
                        └───────────┬─────────────┘ ◄── LLM FDA 21 CFR Part 11 /
                                    │ (Approved)        QbD Design Space Audit
                                    ▼
                        ┌─────────────────────────┐
                        │   arbitrator_node       │ ◄── LLM Consensus Synthesis & Action Directive
                        └───────────┬─────────────┘
                                    │
                                    ▼
                         [ Final Action Directive ]
                  (e.g., ACTION: >> TRACE_PUMP_ON << [95% Confidence])
```

---

## 🔬 The 5 Autonomous Agent Personas

### 1. Anomaly Evaluator Agent (`evaluator_node`)
* **Role:** Autonomous multivariate perception & trend analysis.
* **Mechanism:** Prompts an LLM with `_EVALUATOR_PROMPT` to analyze multi-parameter telemetry ($[\text{Lactate}]$, $[\text{Glucose}]$, $[\text{Biomass}]$, $\text{pH}$), formulate 2–3 rival hypotheses, and assign a severity grade (`NOMINAL`, `WARNING`, `CRITICAL_CQA`).

### 2. Metabolic Biologist Agent (`biologist_node`)
* **Role:** Pure ReAct reasoning with dynamic tool selection.
* **Mechanism:** Uses `llm.bind_tools([query_knowledge_graph, query_pathway_details, simulate_counterfactual_intervention])`.
* **Behavior:** The LLM autonomously chooses which tools to execute, receives `ToolMessage` payloads, tests candidate interventions in the Bio-Twin, and incorporates peer review critiques if running inside a revision loop.

### 3. Bioprocess Engineer Agent (`engineer_node`)
* **Role:** LLM hardware feasibility & hydrodynamic safety guardrail.
* **Mechanism:** Evaluates pump flow envelopes ($F_{\text{glucose}} \le 0.50\text{ L/h}$, $F_{\text{trace}} \le 0.02\text{ L/h}$), dilution washout constraints ($F/V \le 0.50\text{ h}^{-1}$), and actuator slew rates.
* **Debate Routing:** If hardware limits are violated, emits `approved = False` with specific engineering critiques and routes back to the Biologist.

### 4. cGMP Quality Auditor Agent (`cgmp_auditor_node`)
* **Role:** LLM regulatory compliance & Quality by Design (QbD) validation.
* **Mechanism:** Evaluates whether the proposed strategy and simulated trajectory maintain broth lactate within CQA boundaries ($\le 2.0\text{ mmol/L}$) under FDA 21 CFR Part 11 / ICH Q8 guidelines.
* **Debate Routing:** If the trajectory is non-conforming, triggers back-routing to the Biologist for therapeutic re-planning.

### 5. Executive Arbitrator (`arbitrator_node`)
* **Role:** LLM consensus synthesis and execution dispatch.
* **Mechanism:** Synthesizes the full multi-turn deliberation dossier, validates unanimous or conditional approval, and emits the final machine-executable directive with a confidence rating.

---

## 📦 StateGraph State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    telemetry:            dict[str, Any]  # Sensor snapshot (biomass, glucose, lactate, pH)
    anomaly_evaluation:   dict[str, Any]  # Multivariate triage assessment & flagged symptoms
    hypotheses:           list[str]       # Candidate biological & mechanical failure modes
    biologist_diagnosis:  dict[str, Any]  # Root-cause analysis + proposed remedy from Biologist
    simulation_results:   dict[str, Any]  # Lookahead digital twin counterfactual trajectory
    engineer_review:      dict[str, Any]  # Equipment limits & hydrodynamic feasibility review
    cgmp_audit:           dict[str, Any]  # FDA/QbD compliance scorecard & critical quality check
    critique_feedback:    str             # Peer review critique triggering debate/revision
    revision_count:       int             # Counter preventing infinite debate loops (max 3)
    final_decision:       dict[str, Any]  # Synthesized execution directive
    confidence:           float           # Consensus confidence score [0.0 - 1.0]
    deliberation_history: list[str]       # Multi-agent audit trail log
```

---

## 🚀 Verification & Testing

Run the automated verification suite:

```powershell
cd 3_agent_swarm
uv run python test_swarm.py
```

**Test Coverage:**
* **Test 1: Nominal Telemetry** — Zero false alarms on healthy vitals with early exit.
* **Test 2: Lactate Spike Anomaly** — ReAct tool-calling, Bio-Twin simulation ($3.42 \rightarrow 2.80\text{ mmol/L}$), engineer approval, and cGMP compliance scorecard.
* **Test 3: Glucose Surge Anomaly** — Detection of pump calibration drift and line flushing authorization.
* **Test 4: Multi-Agent Reflection & Debate Loop** — Multi-turn back-routing on peer review objections, revision processing, and convergence to consensus.

---

## 📡 Running the Live Kafka Watchdog

```powershell
python watchdog.py
```
