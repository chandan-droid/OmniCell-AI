"""
watchdog.py — OmniCell-AI Phase 3 | Kafka Diagnostic Watchdog
───────────────────────────────────────────────────────────────
Continuously consumes telemetry from the 'omnicell-telemetry' Kafka topic,
feeds each payload into the LangGraph diagnostic swarm, and prints
actionable alerts to stdout.

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
    """
    Safely parse a Kafka message payload as JSON.

    Returns None if the payload is empty or malformed.
    """
    if not raw_value:
        return None
    try:
        return json.loads(raw_value.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"  [Watchdog] ⚠️  Malformed message — skipping. Reason: {exc}")
        return None


def _emit_alert(recommended_action: str, diagnosis: dict, telemetry: dict) -> None:
    """Pretty-print a diagnostic alert to stdout."""
    border = "═" * 60
    print(f"\n{border}")
    print(f"  ⚠️  OMNICELL ALERT")
    print(f"  Anomaly   : {diagnosis.get('anomaly', 'Unknown')}")
    print(f"  Treatment : {diagnosis.get('treatment', 'Unknown')}")
    print(f"  ACTION    : ► {recommended_action.upper()} ◄")
    print(f"  Telemetry : Lactate={telemetry.get('lactate_mmolL', '?')} mmol/L"
          f"  |  Glucose={telemetry.get('glucose_gL', '?')} g/L"
          f"  |  pH={telemetry.get('pH', '?')}")
    print(f"{border}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Watchdog Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_watchdog() -> None:
    """
    Continuous Kafka consumer + LangGraph swarm dispatcher.

    Loop:
      1. Poll Kafka for a message (timeout: KAFKA_POLL_MS)
      2. Skip on timeout / transient error
      3. Raise on fatal Kafka error
      4. Parse JSON payload
      5. Invoke LangGraph swarm
      6. Emit alert if a recommended_action is present
    """
    print("🚀 OmniCell-AI Diagnostic Watchdog starting …")
    print(f"   Broker : {KAFKA_BOOTSTRAP}")
    print(f"   Topic  : {KAFKA_TOPIC}")
    print(f"   Group  : {KAFKA_GROUP_ID}\n")

    # Compile LangGraph app (done once — expensive LLM binding)
    print("🔧 Compiling LangGraph diagnostic swarm …")
    app = build_graph()
    print("✅ Swarm ready.\n")

    consumer = Consumer(CONSUMER_CONFIG)
    consumer.subscribe([KAFKA_TOPIC])
    print(f"👂 Listening on topic '{KAFKA_TOPIC}' … (Ctrl+C to stop)\n")

    messages_processed = 0
    alerts_fired       = 0

    try:
        while _RUNNING:
            # ── Poll ────────────────────────────────────────────────────────
            msg = consumer.poll(timeout=KAFKA_POLL_MS / 1000.0)

            if msg is None:
                # Normal timeout — no message in this window
                continue

            # ── Kafka-level errors ──────────────────────────────────────────
            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    # Reached end of partition — not fatal
                    continue
                elif err.fatal():
                    raise KafkaException(err)
                else:
                    print(f"  [Watchdog] ⚠️  Non-fatal Kafka error: {err}")
                    continue

            # ── Parse payload ───────────────────────────────────────────────
            telemetry = _parse_message(msg.value())
            if telemetry is None:
                continue

            messages_processed += 1
            ts = telemetry.get("timestamp", "?")
            print(f"[MSG #{messages_processed}] ts={ts} | "
                  f"lactate={telemetry.get('lactate_mmolL', '?')} mmol/L | "
                  f"glucose={telemetry.get('glucose_gL', '?')} g/L | "
                  f"pH={telemetry.get('pH', '?')}")

            # ── Invoke LangGraph swarm ──────────────────────────────────────
            initial_state = {
                "telemetry":          telemetry,
                "symptom":            "",
                "diagnosis":          {},
                "recommended_action": "",
            }

            try:
                result = app.invoke(initial_state)
            except Exception as exc:
                print(f"  [Watchdog] ❌ Swarm invocation error: {exc}")
                continue

            # ── Emit alert if actionable ────────────────────────────────────
            action = result.get("recommended_action", "")
            if action:
                alerts_fired += 1
                _emit_alert(
                    recommended_action=action,
                    diagnosis=result.get("diagnosis", {}),
                    telemetry=telemetry,
                )

    except KafkaException as exc:
        print(f"\n❌ Fatal Kafka exception: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"\n📊 Watchdog shutdown — {messages_processed} messages processed, "
              f"{alerts_fired} alerts fired.")
        consumer.close()


if __name__ == "__main__":
    run_watchdog()


