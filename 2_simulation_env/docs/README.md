# OmniCell-AI Bio-Twin: Software-in-the-Loop Simulation

## Module Overview

The **Bio-Twin Simulation** module acts as the deterministic mathematical ground truth for the OmniCell-AI control architecture. Because training a Deep Reinforcement Learning (DRL) agent on a physical, commercial-scale bioreactor is extremely dangerous and cost-prohibitive, this module provides a **Software-in-the-Loop (SIL)** sandbox.

It couples **genome-scale metabolic modeling (via CobraPy)** with **macroscopic mass-balance kinetics** inside a standardized **OpenAI Gymnasium** interface. This ensures that the RL agent learns to navigate genuine thermodynamic constraints and biological physics before it is ever deployed to production hardware.

---

## Mathematical & Scientific Foundation

To prevent the AI from exploiting software bugs or "hallucinating" impossible physics, the environment strictly separates cellular biochemistry from tank fluid dynamics.

### 1. The Cellular Logic: Dynamic Flux Balance Analysis (dFBA)

The internal metabolic state of the cell is calculated at each time step using Flux Balance Analysis (FBA). The algorithm solves a linear programming problem to optimize for cell growth based on current environmental constraints:

$$\max v_{biomass}$$

$$\text{subject to }\mathbf{S}\mathbf{v}=0, \quad \mathbf{v}_{min}\leq\mathbf{v}\leq\mathbf{v}_{max}$$

* $\mathbf{S}$: The stoichiometric matrix of the target cell line (e.g., *E. coli* or CHO).
* $\mathbf{v}$: The vector of metabolic fluxes (reaction rates).
* $\mathbf{v}_{min}, \mathbf{v}_{max}$: Dynamic uptake bounds dictated by the physical tank environment.

### 2. The Physical Wrapper: Macroscopic Mass Balance

The bioreactor tank physics are calculated using Euler integration of ordinary differential equations (ODEs). This calculates the accumulation of biomass ($X$), substrate/glucose ($S$), and waste/lactate ($L$) over time ($t$):

$$\frac{dX}{dt}=\mu X-\frac{F}{V}X$$

$$\frac{dS}{dt}=-q_s X+\frac{F}{V}(S_{in}-S)$$

$$\frac{dL}{dt}=q_l X-\frac{F}{V}L$$

* $\mu$: Specific growth rate (calculated dynamically by CobraPy).
* $q_s, q_l$: Substrate consumption and lactate excretion rates (calculated by CobraPy).
* $F$: Volumetric feed rate (controlled by the RL Agent).
* $V$: Tank volume.

---

## Implementation Plan & Engineering Logic

The module is strictly separated into specific execution layers to maintain clean architecture.

### Step 1: Environment Provisioning (`uv`)

* **The Action:** The module uses `uv` to create an isolated Python virtual environment, defining strict lockfiles for `cobra`, `gymnasium`, and `numpy`.
* **The Reason:** Resolves complex scientific dependency trees instantly and guarantees zero cross-contamination with the LangGraph or RLlib environments. It ensures 100% reproducibility when deployed to edge inferencing servers.

### Step 2: The Biological Engine (`cobra_metabolism.py`)

* **The Action:** A stateless Python class that loads the genomic model (e.g., `textbook` or `iCHOv1`) and exposes a single optimization method.
* **The Reason:** Separation of concerns. The biological solver must be entirely unaware of the RL agent or the physical tank size. This plug-and-play architecture allows you to swap an *E. coli* model for a mammalian cell model without rewriting any downstream control logic.

### Step 3: Macroscopic Physics & Gym API (`bioreactor_gym_env.py`)

* **The Action:** A custom class inheriting from `gym.Env` that handles the Euler physics integration and calculates the all-important Reward function.
* **The Reason:** RL algorithms (like PPO or SAC) require a standardized interface (`reset()` and `step(action)`). The physics wrapper enforces the mass-balance laws, translating the AI's abstract mathematical requests into physical constraints.

### Step 4: System Verification (`test_environment.py`)

* **The Action:** A standalone loop that triggers the environment without any external ML agents.
* **The Reason:** Mathematical integration errors, tensor shape mismatches, or infinite ODE loops must be caught here. If the simulation is mathematically unstable, the downstream RL agent will fail to converge.

---

## API Interfaces

### Observation Space (The AI's "Eyes")

The environment limits the AI's vision to realistic, physically measurable sensor limits (e.g., what a Raman spectrometer would output).

* **Format:** `spaces.Box(low, high, shape=(3,), dtype=np.float32)`
* **Index 0:** Biomass Concentration (g/L)
* **Index 1:** Glucose Concentration (g/L)
* **Index 2:** Lactate Concentration (mmol/L)

### Action Space (The AI's "Hands")

The environment forces the RL agent to operate within a normalized continuous space controlling 3 physical pump channels:

* **Format:** `spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)`
* **Index 0:** Glucose Feed Rate (Mapped to 0.0 – 0.50 L/h)
* **Index 1:** Base Buffer Rate for pH stabilization (Mapped to 0.0 – 0.10 L/h)
* **Index 2:** Trace Element / Micronutrient Feed for enzyme cofactor injection (Mapped to 0.0 – 0.02 L/h)

### Synthetic Genetic Drift & Chemical Bypass Engine

To train downstream diagnostic agent swarms and safe controllers on metabolic bottleneck recovery:
* **Genetic Drift Trigger:** At simulation step 150, Pyruvate Dehydrogenase (`PDH`) flux is artificially suppressed (`knock_down_fraction=0.15`), simulating epigenetic silencing and cellular aging.
* **Chemical Bypass Recovery:** If the RL agent actuates trace nutrient feed above `0.005 L/h` (Index 2), pathway enzyme activity is restored (`knock_down_fraction=1.0`), restoring optimal biomass growth.

### Gymnasium Environment Registration (`__init__.py`)

Registered with Gymnasium for Ray/RLlib distributed worker rollouts:
* **Environment ID:** `OmniCellBioreactor-v0`
* **Entry Point:** `bioreactor_gym_env:BioreactorTwinEnv`
* **Max Episode Steps:** `500`

### The Reward Function (The Incentive)

To ensure FDA Quality by Design (QbD) compliance, the reward function mathematically incentivizes safety over pure yield optimization:

$$R_t=(\mu_t \times 10)-(L_t \times 2)-P_{crash}$$

* **Growth Reward:** Positive points for a high specific growth rate ($\mu_t$).
* **Toxicity Penalty:** Negative points scaling with lactate accumulation ($L_t$).
* **Catastrophic Penalty ($P_{crash}$):** A massive negative scalar (e.g., $-1000$) applied immediately if the batch exceeds safe viability thresholds, triggering an episode truncation to train the AI against reckless feeding.

---

## Execution & Testing Instructions

1. **Activate the Environment:**
```powershell
cd 2_simulation_env
.venv\Scripts\activate
```

2. **Run System Verification Testing:**
Verifies the Euler mass-balance wrapper, dFBA kinetics, and OpenAI Gym tensor formatting.
```powershell
python test_environment.py
```

3. **Launch Real-Time Graphical Dashboard:**
Animates real-time biological curves (Biomass, Glucose, Lactate) and visually observes genetic drift crash at Step 150.
```powershell
python live_dashboard.py
```

---

## Upstream/Downstream Connectivity

* **Downstream (To Kafka):** The output of `self._get_obs()` serves as the baseline "ground truth" accessed by the Go Edge Ingestion Engine (Phase 1), which adds hardware noise before publishing to the data lake.
* **Upstream (From RLlib):** The `step(action)` method receives its continuous array input directly from the Ray/RLlib PPO execution loop (Phase 4), subject to CVXPY safety bounds.
