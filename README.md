# OmniCell-AI: Autonomous Bioreactor Control Architecture

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

### Step 4: Run Agent Swarm & Controller (Modules 3 & 4)

*(When implemented)*
```powershell
cd 3_agent_swarm
python main.py
```

---

### Step 5: Start Frontend Dashboard (Module 5)

*(When implemented)*
```powershell
cd 5_frontend_dashboard
npm run dev
```

---

## 🧪 Module Reference Summary

| Module | Core Technology | Primary Responsibility |
| :--- | :--- | :--- |
| **`1_edge_ingestion`** | Go, Apache Kafka | High-throughput telemetry ingestion & noise injection |
| **`2_simulation_env`** | CobraPy, Gymnasium, NumPy | Dynamic Flux Balance Analysis (dFBA) & Euler physics twin |
| **`3_agent_swarm`** | LangGraph, Neo4j, Qdrant | Diagnostic reasoning, anomaly detection & root-cause analysis |
| **`4_safe_controller`** | Ray RLlib, CVXPY | Constrained Reinforcement Learning for feeding pump control |
| **`5_frontend_dashboard`** | HTML/JS, WebSockets | Process analytical technology (PAT) visual monitoring |

---

## 📄 License & Confidentiality

Internal proprietary codebase for OmniCell-AI SIL Simulation Architecture.
