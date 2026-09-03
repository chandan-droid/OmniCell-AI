# OmniCell-AI Phase 5: Real-time PAT Frontend Dashboard

## Module Overview

The **Frontend Dashboard** (`5_frontend_dashboard`) serves as the operator-facing Process Analytical Technology (PAT) monitoring cockpit for OmniCell-AI. Designed for bioprocess engineers and plant operators, it provides real-time visualization of culture health, metabolic fluxes, actuator telemetry, and multi-agent clinical diagnostic alerts.

---

## 🏛️ Planned Architecture

* **Framework:** React / Next.js or lightweight Vanilla JS with WebSocket streaming.
* **Charts & Telemetry:** WebGL / Canvas-based real-time line charts (Chart.js / uPlot) plotting biomass, glucose, lactate, and feeding rates.
* **Alarm Console:** Live event log displaying diagnostic notifications and corrective actions emitted by the Phase 3 LangGraph Swarm.
* **Control Override:** Safe operator manual intervention switches with bioprocess audit logging.
