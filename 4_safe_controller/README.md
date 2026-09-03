# OmniCell-AI Phase 4: Safe DRL Bioreactor Controller & Triton Inference

## Module Overview

The **Safe Controller** module (`4_safe_controller`) is the autonomous actuation engine of the OmniCell-AI platform. In an industrial bioprocess (e.g., monoclonal antibody or recombinant protein production), maintaining cell viability and maximizing product titer while preventing toxic byproduct accumulation (such as lactate or ammonia) requires continuous, closed-loop feeding control.

Training a standard Reinforcement Learning (RL) agent directly on physical equipment risks equipment damage or catastrophic batch loss. In this architecture:
1. The RL agent is trained **offline** inside the **Phase 2 Bio-Twin Software-in-the-Loop (SIL)** simulation.
2. The learned policy is shielded by a **mathematical safety filter** (Control Barrier Functions / CVXPY QP solver).
3. The trained policy is exported as an **ONNX graph** (`model.onnx`).
4. The model is hosted in **NVIDIA Triton Inference Server** (`model_repository/`) for microsecond-latency gRPC inference.

---

## 🏛️ System & Deployment Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │               2_simulation_env (SIL Bio-Twin)          │
       └───────────────────────────┬────────────────────────────┘
                                   │  Gymnasium API (obs, reward, done)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        4_safe_controller (Offline)                     │
│                                                                        │
│   [ train.py ]  ──► PPO / SAC Agent (Stable-Baselines3 / Ray RLlib)    │
│                           │                                            │
│                           ▼ (Trained PyTorch Policy)                   │
│   [ safety_filter.py ] ──► Quadratic Program / CBF Validation          │
│                           │                                            │
│                           ▼                                            │
│   [ export_onnx.py ]   ──► Exports to ONNX Format                      │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼ Generated Artifact
┌────────────────────────────────────────────────────────────────────────┐
│                         model_repository/                              │
│   └── omnicell_rl_controller/                                          │
│       ├── config.pbtxt             (Triton tensor shape configuration) │
│       └── 1/                                                           │
│           └── model.onnx           (Universal Neural Network Weights)  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼ Mounted into Docker
┌────────────────────────────────────────────────────────────────────────┐
│               docker-compose: triton-server (:8000 / :8001)            │
│      - High-throughput gRPC / HTTP dynamic batching inference          │
│      - Queried in real-time by Edge Ingestion / Kafka Consumer Loop    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Scientific & Algorithmic Foundation

### 1. Markov Decision Process (MDP) Definition

* **State / Observation Space ($\mathcal{S} \in \mathbb{R}^3$):**
  * $X$: Viable Biomass Concentration ($\text{g/L}$)
  * $S$: Substrate / Glucose Concentration ($\text{g/L}$)
  * $L$: Toxic Byproduct / Lactate Concentration ($\text{mmol/L}$)
* **Action Space ($\mathcal{A} \in [0, 1]$):**
  * $F_{in}$: Normalized Volumetric Feed Pump Speed ($\text{L/h}$)
* **Reward Formulation ($R_t$):**
  $$R_t = w_1 \cdot \Delta X - w_2 \cdot \max(0, L_t - L_{\text{threshold}})^2 - w_3 \cdot \|a_t - a_{t-1}\|^2$$
  * $w_1 \cdot \Delta X$: Incentivizes exponential biomass growth.
  * $w_2 \cdot (L_t - L_{\text{thresh}})^2$: Heavily penalizes metabolic overflow and acidification.
  * $w_3 \cdot \|\Delta a\|^2$: Promotes smooth actuator control to prevent mechanical pump wear.

### 2. Safety Layer (Action Shielding via CVXPY)

To ensure strict compliance with bioprocess Quality by Design (QbD) operating envelopes, the raw RL policy output $a_{\text{rl}}$ is projected into an admissible safe control set $\mathcal{C}_{\text{safe}}$ via a real-time Quadratic Program (QP):

$$\min_{a} \frac{1}{2} \| a - a_{\text{rl}} \|_2^2$$

$$\text{subject to: } a_{\min} \le a \le a_{\max}, \quad \nabla h(s)^T f(s, a) \ge -\gamma h(s)$$

Where $h(s) \ge 0$ defines the barrier function guaranteeing lactate and glucose remain within safe physiological bounds.

---

## 📁 Repository Layout

When fully implemented, the directory structure for Phase 4 and the Triton inference store:

```text
omnicell-ai-sil/
├── 4_safe_controller/
│   ├── README.md               # This reference documentation
│   ├── pyproject.toml          # uv project dependencies
│   ├── requirements.txt        # SB3, PyTorch, CVXPY, ONNX, ONNXRuntime
│   ├── train.py                # PPO/SAC training pipeline with BioTwinEnv
│   ├── evaluate.py             # Policy benchmarking against baseline PID/fed-batch
│   ├── safety_shield.py        # CVXPY barrier safety filter
│   └── export_onnx.py          # Torch-to-ONNX conversion script
│
└── model_repository/           # Mounted to /models in Triton Server
    └── omnicell_rl_controller/
        ├── config.pbtxt        # Triton model definition
        └── 1/
            └── model.onnx      # Exported neural network
```

---

## ⚙️ Triton `config.pbtxt` Specification

Below is the configuration that must accompany `model.onnx` inside `model_repository/omnicell_rl_controller/config.pbtxt`:

```protobuf
name: "omnicell_rl_controller"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "bioreactor_state"
    data_type: TYPE_FP32
    dims: [ 3 ]  # [Biomass, Glucose, Lactate]
  }
]

output [
  {
    name: "pump_action"
    data_type: TYPE_FP32
    dims: [ 1 ]  # [Feed Pump Speed: 0.0 to 1.0]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU   # Set to KIND_GPU if NVIDIA GPU drivers are enabled
  }
]
```

---

## 🚀 Execution Workflow

### Step 1: Set Up Python Environment

```powershell
cd 4_safe_controller
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Step 2: Train the Controller

```powershell
python train.py --timesteps 500000 --algo PPO
```

### Step 3: Export Policy to ONNX

```powershell
python export_onnx.py --checkpoint ./checkpoints/best_model.zip --output ../model_repository/omnicell_rl_controller/1/model.onnx
```

### Step 4: Start Triton Server in Docker

From the repository root:

```powershell
docker-compose up -d triton-server
```

Verify server status:
* **HTTP Health:** `curl http://localhost:8000/v2/health/ready`
* **Model Metadata:** `curl http://localhost:8000/v2/models/omnicell_rl_controller`
