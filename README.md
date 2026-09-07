# OmniCell-AI: Foundation-Driven Autonomous Bioreactor Control with cGMP-Compliant Multi-Agent Alignment

**OmniCell-AI** is an industrial Software-in-the-Loop (SIL) platform for autonomous bioreactor monitoring, metabolic dynamic flux balance analysis (dFBA), and real-time Quality by Design (QbD) control.

---

## 🏗️ System Architecture

The project is structured into 5 core execution layers:

```
omnicell-ai-sil/
├── 1_edge_ingestion/      # Go-based telemetry ingestion engine & Kafka producer
├── 2_simulation_env/      # SIL Bio-Twin simulation (CobraPy + OpenAI Gymnasium)
├── 3_agent_swarm/         # Diagnostic Brain (LangGraph multi-agent system + Knowledge Graph)
├── 4_safe_controller/     # Safe DRL Controller (Ray RLlib / CVXPY safety layer)
├── 5_frontend_dashboard/  # Real-time Web Dashboard for telemetry & control monitoring
└── docker-compose.yml     # Infrastructure stack (Kafka, TimescaleDB, Neo4j, Qdrant, Triton)
```

---

## 📋 Prerequisites

* **Python:** `3.12` (Managed automatically via `uv`)
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (`pip install uv` or `winget install ast-sh.uv`)
* **Docker & Docker Compose:** For running Kafka, TimescaleDB, Neo4j, Qdrant, and Triton.
* **Go:** `1.21+` (Required for Phase 1 Edge Ingestion)

---

## 🚀 Running Instructions

### Step 1: Start Infrastructure Services (Docker)

Spin up the message broker, time-series data lake, graph database, vector store, and inference server:

```powershell
docker-compose up -d
```

To verify running services:
```powershell
docker-compose ps
```

* **Kafka Broker:** `localhost:9092`
* **TimescaleDB:** `localhost:5432` (`omnicell_admin`)
* **Neo4j Dashboard:** `http://localhost:7474`
* **Qdrant Vector DB:** `http://localhost:6333`
* **Triton Inference:** `http://localhost:8000`

---

### Step 2: Set Up & Run Module 2 (SIL Bio-Twin Simulation)

1. Navigate to the simulation environment directory:
   ```powershell
   cd 2_simulation_env
   ```

2. Create a virtual environment using Python 3.12 (uv auto-fetches Python 3.12 if not present):
   ```powershell
   uv venv --python 3.12
   ```

3. Activate the virtual environment:
   ```powershell
   # Windows PowerShell
   .venv\Scripts\activate

   # Linux/macOS
   source .venv/bin/activate
   ```

4. Install dependencies from `requirements.txt`:
   ```powershell
   uv pip install -r requirements.txt
   ```

5. Run System Verification Testing:
   ```powershell
   python test_environment.py
   ```

**Expected Output:**
```text
Initial State: Biomass=0.10, Glucose=20.00
Final State: Biomass=6.35, Glucose=73.43, Lactate=0.00
Verification Passed: dFBA and Euler Physics successfully integrated.
```

---

### Step 3: Run Edge Ingestion (Module 1)

*(When implemented)*
```powershell
cd 1_edge_ingestion
go run main.go
```

---

### Step 4: Run Multi-Agent Swarm — Phase 3 (LangGraph Deliberation Swarm & Digital Twin GraphRAG)

#### 4a. Install Dependencies
```powershell
cd 3_agent_swarm
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

#### 4b. Configure Environment
```powershell
copy .env.example .env
# Open .env and set OPENAI_API_KEY, Neo4j password, etc.
```

#### 4c. Seed the Neo4j Knowledge Graph (run once)
```powershell
python neo4j_seeder.py
```
> Verify the graph at **http://localhost:7474** — you should see `Anomaly`, `Symptom`, `Enzyme`, and `Treatment` nodes connected by `CAUSES`, `INDICATES`, `SUPPRESSES`, and `RESTORES` relationships.

#### 4d. Run Multi-Agent Swarm Verification Suite
```powershell
uv run python test_swarm.py
```

#### 4e. Start the Kafka Diagnostic Watchdog
```powershell
python watchdog.py
```

**Expected output (multi-agent deliberation scenario):**
```text
[MSG #4] ts=2026-09-06 21:40:02 | lactate=3.42 mmol/L | glucose=18.0 g/L | pH=6.95
  [Evaluator] Vitals: Lactate=3.42 mM | Glucose=18.00 g/L | Biomass=1.80 g/L | pH=6.95 -> Status: CRITICAL_CQA (Flagged: Lactate Spike)
  [Biologist] Investigating root cause for symptom: 'Lactate Spike' ...
  [Biologist] Root Cause: Overflow Metabolism (Lactate Dehydrogenase / PDH) | Proposal: trace_pump_on | Sim Outcome: Simulated 2.0h lookahead: Biomass 1.80->4.29 g/L, Lactate 3.42->2.80 mmol/L (decreasing)
  [Engineer] Equipment Audit: APPROVED for action 'trace_pump_on'. Notes: Trace cofactor feed commanded at 0.010 L/h.
  [cGMP Auditor] CQA Evaluation: PASS_WITH_MONITORING (Risk: MODERATE) | Predicted Lactate: 2.80 mmol/L (CQA limit <= 2.0)
  [Arbitrator] Consensus Reached (CONDITIONAL_APPROVAL, Confidence: 75%). DISPATCHING ACTION: >> TRACE_PUMP_ON <<

══════════════════════════════════════════════════════════════════════
  🧬 OMNICELL MULTI-AGENT CONSENSUS ALERT
  Anomaly         : Overflow Metabolism
  Treatment       : Activate Trace Pump & Modulate Feed
  ACTION DIRECTIVE: ► TRACE_PUMP_ON ◄
  Confidence      : 75% (CONDITIONAL_APPROVAL)
  Bio-Twin Sim    : Simulated 2.0h lookahead: Biomass 1.80->4.29 g/L, Lactate 3.42->2.80 mmol/L (decreasing)
  cGMP Audit      : PASS_WITH_MONITORING | Risk: MODERATE
  Telemetry       : Lactate=3.42 mmol/L | Glucose=18.0 g/L | Biomass=1.8 g/L
══════════════════════════════════════════════════════════════════════
```

---

## 🧪 Module Reference Summary

| Module | Core Technology | Primary Responsibility |
| :--- | :--- | :--- |
| **`1_edge_ingestion`** | Go, Apache Kafka | High-throughput telemetry ingestion & noise injection |
| **`2_simulation_env`** | CobraPy, Gymnasium, NumPy | Dynamic Flux Balance Analysis (dFBA) & Euler physics twin |
| **`3_agent_swarm`** | LangGraph, Neo4j, Qdrant | Pure multi-agent deliberation, GraphRAG & counterfactual simulation |
| **`4_safe_controller`** | Ray RLlib, CVXPY | Constrained Reinforcement Learning for feeding pump control |
| **`5_frontend_dashboard`** | HTML/JS, WebSockets | Process analytical technology (PAT) visual monitoring |

---

## 🔮 Future Scope (Project Roadmap)

Based on the architectural constraints and deliberate exclusions of the current MVP, the enterprise roadmap includes:

### 1. 3D Multi-Physics & Fluid Dynamics Integration

* **Computational Fluid Dynamics (CFD):** Transition from a 1D "perfectly mixed" assumption to a 3D spatial model to calculate true mixing times, dead zones, and impeller shear stress on the cells.
* **Dynamic Gas Transfer Kinetics ($k_L a$):** Replace static oxygen boundaries with active sparger modeling, calculating bubble size distribution, gas hold-up, and real-time volumetric mass transfer.
* **Thermodynamic Modeling:** Implement metabolic heat generation and dynamic cooling jacket compensation algorithms, moving away from the static 37°C assumption.

### 2. Reinforcement Learning (RL) Expansion

* **Expanded Action Space:** Upgrade the Gym environment so the AI agent can control physical hardware variables—such as Agitator RPM, Sparger Airflow rates, and Cooling Jacket flow—rather than just chemical pumps.
* **Continuous Control Policies:** Train advanced RL algorithms (like PPO or SAC) to dynamically optimize these new variables in real-time to maximize biomass yield.

### 3. GraphRAG & AI Diagnostic Scaling

* **Distractor Anomalies:** Expand the Neo4j knowledge base with thousands of overlapping hardware failures (e.g., agitator motor stalls, DO probe calibration drift). This forces the LangGraph swarm to perform complex logical elimination (checking multiple Kafka streams) rather than following a single linear path.
* **Enterprise Cell Lines:** Swap the textbook *E. coli* CobraPy payload for a commercial Chinese Hamster Ovary (CHO) cell model (which contains 6,000+ reactions) to mirror true biopharmaceutical manufacturing complexity.

---

## 📄 License & Confidentiality

Internal proprietary codebase for OmniCell-AI SIL Simulation Architecture.

