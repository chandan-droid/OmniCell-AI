"""
watchdog.py — OmniCell-AI Phase 3 | Kafka Diagnostic Watchdog
───────────────────────────────────────────────────────────────
Continuously consumes telemetry from the 'omnicell-telemetry' Kafka topic,
feeds each payload into the LangGraph multi-agent deliberation swarm, and
prints multi-agent consensus alerts to stdout.

Run:
  python watchdog.py

Environment variables (optional, override via .env or shell):
  KAFKA_BOOTSTRAP  = localhost:9092
  KAFKA_TOPIC      = omnicell-telemetry
  KAFKA_GROUP_ID   = omnicell-swarm-watchdog
  KAFKA_POLL_MS    = 1000
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

# Load .env from the module's own directory (if present)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from swarm import build_graph  # Import after dotenv so API keys are set

# ── Kafka Configuration ────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "omnicell-telemetry")
KAFKA_GROUP_ID  = os.getenv("KAFKA_GROUP_ID",  "omnicell-swarm-watchdog")
KAFKA_POLL_MS   = int(os.getenv("KAFKA_POLL_MS", "1000"))

CONSUMER_CONFIG = {
    "bootstrap.servers":        KAFKA_BOOTSTRAP,
    "group.id":                 KAFKA_GROUP_ID,
    "auto.offset.reset":        "latest",      # Skip historical backlog on first start
    "enable.auto.commit":       True,
    "session.timeout.ms":       30_000,
    "max.poll.interval.ms":    300_000,
}

# ── Graceful shutdown flag ─────────────────────────────────────────────────────
_RUNNING = True

def _handle_sigterm(signum, frame) -> None:  # noqa: ANN001
    global _RUNNING
    print("\n⏹  Received shutdown signal — draining and exiting …")
    _RUNNING = False

signal.signal(signal.SIGINT,  _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_message(raw_value: bytes) -> dict | None:
    """Safely parse a Kafka message payload as JSON."""
    if not raw_value:
        return None
    try:
        return json.loads(raw_value.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"  [Watchdog] ⚠️  Malformed message — skipping. Reason: {exc}")
        return None


def _emit_agent_alert(result: dict[str, Any], telemetry: dict[str, Any]) -> None:
    """Pretty-print a full multi-agent consensus deliberation alert to stdout."""
    decision = result.get("final_decision", {})
    action = decision.get("action", "")
    if not action or action == "hold_current_state":
        return

    confidence = result.get("confidence", 0.0) * 100
    cgmp = result.get("cgmp_audit", {})
    eng = result.get("engineer_review", {})
    sim = result.get("simulation_results", {})

    border = "═" * 70
    print(f"\n{border}")
    print(f"  🧬 OMNICELL MULTI-AGENT CONSENSUS ALERT")
    print(f"  Anomaly         : {decision.get('anomaly', 'Unknown')}")
    print(f"  Treatment       : {decision.get('treatment', 'Unknown')}")
    print(f"  ACTION DIRECTIVE: ► {action.upper()} ◄")
    print(f"  Confidence      : {confidence:.0f}% ({decision.get('consensus_status', 'APPROVED')})")
    print(f"  Bio-Twin Sim    : {sim.get('summary', 'Sim verified')}")
    print(f"  cGMP Audit      : {cgmp.get('compliance_rating', 'PASS')} | Risk: {cgmp.get('risk_grade', 'LOW')}")
    print(f"  Telemetry       : Lactate={telemetry.get('lactate_mmolL', telemetry.get('lactate', '?'))} mmol/L | "
          f"Glucose={telemetry.get('glucose_gL', telemetry.get('glucose', '?'))} g/L | "
          f"Biomass={telemetry.get('biomass_gL', telemetry.get('biomass', '?'))} g/L")
    print(f"{border}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Watchdog Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_watchdog() -> None:
    """Continuous Kafka consumer + LangGraph multi-agent deliberation dispatcher."""
    print("🚀 OmniCell-AI Multi-Agent Diagnostic Watchdog starting …")
    print(f"   Broker : {KAFKA_BOOTSTRAP}")
    print(f"   Topic  : {KAFKA_TOPIC}")
    print(f"   Group  : {KAFKA_GROUP_ID}\n")

    print("🔧 Compiling Multi-Agent LangGraph Swarm …")
    app = build_graph()
    print("✅ Multi-Agent Swarm ready.\n")

    consumer = Consumer(CONSUMER_CONFIG)
    consumer.subscribe([KAFKA_TOPIC])
    print(f"👂 Listening on topic '{KAFKA_TOPIC}' … (Ctrl+C to stop)\n")

    messages_processed = 0
    alerts_fired       = 0

    try:
        while _RUNNING:
            msg = consumer.poll(timeout=KAFKA_POLL_MS / 1000.0)

            if msg is None:
                continue

            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                elif err.fatal():
                    raise KafkaException(err)
                else:
                    print(f"  [Watchdog] ⚠️  Non-fatal Kafka error: {err}")
                    continue

            telemetry = _parse_message(msg.value())
            if telemetry is None:
                continue

            messages_processed += 1
            ts = telemetry.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"\n[MSG #{messages_processed}] ts={ts} | "
                  f"lactate={telemetry.get('lactate_mmolL', telemetry.get('lactate', '?'))} mmol/L | "
                  f"glucose={telemetry.get('glucose_gL', telemetry.get('glucose', '?'))} g/L | "
                  f"biomass={telemetry.get('biomass_gL', telemetry.get('biomass', '?'))} g/L")

            initial_state = {
                "telemetry":            telemetry,
                "anomaly_evaluation":   {},
                "hypotheses":           [],
                "biologist_diagnosis":  {},
                "simulation_results":   {},
                "engineer_review":      {},
                "cgmp_audit":           {},
                "final_decision":       {},
                "confidence":           0.0,
                "deliberation_history": [],
            }

            try:
                result = app.invoke(initial_state)
            except Exception as exc:
                print(f"  [Watchdog] ❌ Swarm deliberation error: {exc}")
                continue

            decision = result.get("final_decision", {})
            action = decision.get("action", "")
            if action and action != "hold_current_state":
                alerts_fired += 1
                _emit_agent_alert(result=result, telemetry=telemetry)

    except KafkaException as exc:
        print(f"\n❌ Fatal Kafka exception: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"\n📊 Watchdog shutdown — {messages_processed} messages processed, "
              f"{alerts_fired} alerts fired.")
        consumer.close()


if __name__ == "__main__":
    run_watchdog()
