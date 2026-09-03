# OmniCell-AI Phase 3: LangGraph Diagnostic Swarm & GraphRAG

## Module Overview

The **Diagnostic Swarm** (`3_agent_swarm`) represents the clinical diagnosis and anomaly response layer of the OmniCell-AI autonomous bioreactor platform. While the Phase 4 RL controller handles routine continuous feeding, biological systems frequently drift or encounter metabolic failure modes (such as overflow metabolism, lactate spikes, or probe fouling) that cannot be safely managed by an ungrounded neural network alone.

This module combines:
1. **Deterministic Edge Triage (Supervisor):** A zero-latency state router that filters routine telemetry and flags biochemical violations against process thresholds.
2. **Agentic LLM Biologist (xAI Grok):** An intelligent agent bound to custom retrieval tools that reasons about observed failure symptoms.
3. **Neo4j GraphRAG (Knowledge Graph):** A deterministic biological knowledge graph connecting symptoms to root-cause cellular anomalies, affected metabolic enzymes, and approved corrective treatments.
4. **Real-time Kafka Watchdog:** A background consumer continuously monitoring the `omnicell-telemetry` stream to triage and emit prioritized industrial alerts.

---

## 🏛️ System Architecture

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
                       │     supervisor_node     │
                       └─────────────────────────┘
                                      │
                         (Is lactate > 2.0 mmol/L?)
                                ├── No ───► [ END / Nominal ]
                                │
                                └── Yes ──► Flags: "Lactate Spike"
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │     biologist_node      │  ◄── (xAI Grok LLM)
                                 └─────────────────────────┘
                                              │
                                   Invokes tool call
                                              │
                                              ▼
                                    [ graph_tool.py ]
                              (query_knowledge_graph)
                                              │
                                       Cypher Traversal
                                              │
                                              ▼
                                    [ Neo4j Database ]
                                (Seeded by neo4j_seeder.py)
                                              │
                                       Returns Payload
                               (Anomaly, Treatment, Action)
                                              │
                                              ▼
                       [ _emit_alert in watchdog.py ]
                 (e.g., ACTION: ► TRACE_PUMP_ON ◄)
```

---

## 🔬 Multi-Agent StateGraph Mechanics

The swarm is orchestrated using **LangGraph** (`swarm.py`), guaranteeing deterministic execution state and verifiable reasoning steps:

### 1. Shared State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    telemetry:           dict    # Raw sensor measurements from Kafka (biomass, glucose, lactate, pH)
    symptom:             str     # Observed failure symptom flagged by Supervisor ("" if nominal)
    diagnosis:           dict    # Structured GraphRAG response from Neo4j
    recommended_action:  str     # Exact machine-executable action string (e.g., 'trace_pump_on')
```

### 2. The Nodes

* **`supervisor_node` (Triage Router):**
  * Evaluates incoming telemetry against critical safety limits (e.g., `lactate_mmolL > LACTATE_THRESHOLD_MMOL`).
  * If vitals are within standard operating bounds, execution terminates immediately at `END` (conserving LLM compute and latency).
  * If a threshold is violated, it writes the symptom name (e.g., `"Lactate Spike"`) into `state["symptom"]` and routes to the Biologist.
* **`biologist_node` (LLM + GraphRAG Tool):**
  * Powered by xAI Grok (or OpenAI-compatible API) with strict zero-temperature constraints.
  * Bound to the `query_knowledge_graph` tool.
  * When invoked, it formulates a structured tool call to retrieve the exact anomaly and treatment mapped in Neo4j, preventing LLM hallucination in safety-critical operations.

---

## 🕸️ Knowledge Graph & GraphRAG Schema (`Neo4j`)

The Neo4j database stores validated bioprocess failure ontologies. It is seeded automatically via `neo4j_seeder.py`.

### Graph Ontology

```text
 (Symptom) ──[:INDICATES]──► (Anomaly) ──[:SUPPRESSES]──► (Enzyme)
                                                            ▲
                                                            │ [:RESTORES]
                                                            │
                                                      (Treatment)
```

### Cypher Traversal (`graph_tool.py`)

When the Biologist agent queries the knowledge base, it executes the following graph traversal:

```cypher
MATCH (s:Symptom {name: $symptom})-[:INDICATES]->(a:Anomaly)-[:SUPPRESSES]->(e)<-[:RESTORES]-(t:Treatment)
RETURN a.name AS anomaly, t.name AS treatment, t.action AS action
```

### Seeded Anomalies & Treatments

| Symptom | Identified Anomaly | Suppressed Enzyme | Corrective Treatment | Machine Action |
| :--- | :--- | :--- | :--- | :--- |
| **Lactate Spike** | Overflow Metabolism | Lactate Dehydrogenase | Reduce Feed Pump Rate | `trace_pump_on` |
| **Glucose Surge** | Pump Calibration Drift | Hexokinase | Flush Feed Line | `flush_feed_line` |
| **pH Drop** | pH Probe Fouling | Carbonic Anhydrase | Inject Base Bolus | `inject_base_bolus` |

---

## 📡 Kafka Telemetry Watchdog (`watchdog.py`)

The watchdog runs as an industrial daemon listening on `omnicell-telemetry`:
* Consumes messages with auto-commit and configurable poll timeouts.
* Deserializes JSON payloads emitted by the Go Edge Ingestion service.
* Invokes the compiled LangGraph workflow synchronously on each pulse.
* Emits high-visibility ANSI console alerts with the exact remedial action.

---

## 🚀 Setup & Execution Guide

### Step 1: Initialize Python Environment

```powershell
cd 3_agent_swarm
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Copy the template configuration:
```powershell
copy .env.example .env
```

Edit `.env` with your credentials:
```ini
# LLM Provider (xAI Grok or any OpenAI-compatible endpoint)
XAI_API_KEY=your_api_key_here
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-2-1212

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=BioprocessSecurePassword2026

# Kafka Broker
KAFKA_BOOTSTRAP=localhost:9092
KAFKA_TOPIC=omnicell-telemetry
KAFKA_GROUP_ID=omnicell-swarm-watchdog
KAFKA_POLL_MS=1000

# Safety Limits
LACTATE_THRESHOLD_MMOL=2.0
```

### Step 3: Seed the Neo4j Knowledge Graph

Ensure Neo4j is running (`docker-compose up -d neo4j`), then seed the graph nodes and relations:

```powershell
python neo4j_seeder.py
```
> Inspect the graph in your browser at **http://localhost:7474**.

### Step 4: Run the Telemetry Watchdog

```powershell
python watchdog.py
```

**Sample Output on Anomaly Detection:**
```text
🚀 OmniCell-AI Diagnostic Watchdog starting …
✅ Swarm ready.
👂 Listening on topic 'omnicell-telemetry' …

[MSG #14] ts=2026-09-03T21:40:02Z | lactate=3.410 mmol/L | glucose=19.2 g/L | pH=7.05
  [Supervisor] Lactate=3.410 mmol/L > 2.0 → flagging 'Lactate Spike'
  [Biologist]  Querying knowledge graph for symptom: 'Lactate Spike' …
  [Biologist]  Anomaly   : Overflow Metabolism
  [Biologist]  Treatment : Reduce Feed Pump Rate
  [Biologist]  Action    : trace_pump_on

════════════════════════════════════════════════════════════
  ⚠️  OMNICELL ALERT
  Anomaly   : Overflow Metabolism
  Treatment : Reduce Feed Pump Rate
  ACTION    : ► TRACE_PUMP_ON ◄
  Telemetry : Lactate=3.410 mmol/L  |  Glucose=19.2 g/L  |  pH=7.05
════════════════════════════════════════════════════════════
```
