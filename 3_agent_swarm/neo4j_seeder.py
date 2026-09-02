"""
neo4j_seeder.py — OmniCell-AI Phase 3
──────────────────────────────────────
Seeds the local Neo4j instance with the bioreactor failure knowledge graph.

Schema:
  (Anomaly)-[:CAUSES]->(Symptom)
  (Symptom)-[:INDICATES]->(Anomaly)
  (Anomaly)-[:SUPPRESSES]->(Enzyme)
  (Treatment)-[:RESTORES]->(Enzyme)

Run once before starting the swarm:
  python neo4j_seeder.py
"""

from neo4j import GraphDatabase
import os

# ── Connection ─────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "BioprocessSecurePassword2026")

# ── Knowledge Base ─────────────────────────────────────────────────────────────
SEED_QUERIES = [
    # ── Clear stale data
    "MATCH (n) DETACH DELETE n",

    # ── Constraints (idempotent re-runs)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Anomaly)   REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom)   REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Enzyme)    REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",

    # ── Nodes — Anomalies
    "MERGE (:Anomaly {name: 'Overflow Metabolism',  description: 'Excess glucose causes overflow metabolism, producing lactate as a toxic byproduct'})",
    "MERGE (:Anomaly {name: 'Pump Calibration Drift', description: 'Feeding pump dispenses above set-point volume due to peristaltic tube wear'})",
    "MERGE (:Anomaly {name: 'pH Probe Fouling',     description: 'Protein deposits on pH probe cause false-high readings, suppressing base addition'})",

    # ── Nodes — Symptoms
    "MERGE (:Symptom {name: 'Lactate Spike',   description: 'Rapid accumulation of L-lactate in culture broth above 2 mmol/L'})",
    "MERGE (:Symptom {name: 'Glucose Surge',   description: 'Glucose concentration exceeds 25 mmol/L due to uncontrolled feed bolus'})",
    "MERGE (:Symptom {name: 'pH Drop',         description: 'Bulk pH falls below 6.8 inhibiting cell growth'})",

    # ── Nodes — Enzymes
    "MERGE (:Enzyme {name: 'Lactate Dehydrogenase', pathway: 'Glycolysis overflow branch'})",
    "MERGE (:Enzyme {name: 'Hexokinase',            pathway: 'Central carbon metabolism entry'})",
    "MERGE (:Enzyme {name: 'Carbonic Anhydrase',    pathway: 'CO2 / bicarbonate equilibrium'})",

    # ── Nodes — Treatments
    """
    MERGE (:Treatment {
        name:        'Reduce Feed Pump Rate',
        action:      'trace_pump_on',
        description: 'Reduce feed pump duty cycle by 30% and re-assay glucose in 15 min'
    })
    """,
    """
    MERGE (:Treatment {
        name:        'Flush Feed Line',
        action:      'flush_feed_line',
        description: 'Trigger feed line purge sequence to clear blockage or calibration offset'
    })
    """,
    """
    MERGE (:Treatment {
        name:        'Inject Base Bolus',
        action:      'inject_base_bolus',
        description: 'Activate base pump (NaHCO3) to restore pH to 7.2 set-point'
    })
    """,

    # ── Relationships — Causes / Indicates
    "MATCH (a:Anomaly {name:'Overflow Metabolism'}),  (s:Symptom {name:'Lactate Spike'})   MERGE (a)-[:CAUSES]->(s) MERGE (s)-[:INDICATES]->(a)",
    "MATCH (a:Anomaly {name:'Pump Calibration Drift'}),(s:Symptom {name:'Glucose Surge'})  MERGE (a)-[:CAUSES]->(s) MERGE (s)-[:INDICATES]->(a)",
    "MATCH (a:Anomaly {name:'Pump Calibration Drift'}),(s:Symptom {name:'Lactate Spike'})  MERGE (a)-[:CAUSES]->(s) MERGE (s)-[:INDICATES]->(a)",
    "MATCH (a:Anomaly {name:'pH Probe Fouling'}),      (s:Symptom {name:'pH Drop'})        MERGE (a)-[:CAUSES]->(s) MERGE (s)-[:INDICATES]->(a)",

    # ── Relationships — Suppresses
    "MATCH (a:Anomaly {name:'Overflow Metabolism'}),   (e:Enzyme {name:'Lactate Dehydrogenase'}) MERGE (a)-[:SUPPRESSES]->(e)",
    "MATCH (a:Anomaly {name:'Pump Calibration Drift'}),(e:Enzyme {name:'Hexokinase'})            MERGE (a)-[:SUPPRESSES]->(e)",
    "MATCH (a:Anomaly {name:'pH Probe Fouling'}),      (e:Enzyme {name:'Carbonic Anhydrase'})    MERGE (a)-[:SUPPRESSES]->(e)",

    # ── Relationships — Restores
    "MATCH (t:Treatment {name:'Reduce Feed Pump Rate'}),(e:Enzyme {name:'Lactate Dehydrogenase'}) MERGE (t)-[:RESTORES]->(e)",
    "MATCH (t:Treatment {name:'Flush Feed Line'}),      (e:Enzyme {name:'Hexokinase'})             MERGE (t)-[:RESTORES]->(e)",
    "MATCH (t:Treatment {name:'Inject Base Bolus'}),    (e:Enzyme {name:'Carbonic Anhydrase'})     MERGE (t)-[:RESTORES]->(e)",
]


def seed_graph() -> None:
    """Connect to Neo4j and execute all seed queries."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            for i, query in enumerate(SEED_QUERIES, 1):
                session.run(query)
                print(f"  [OK] Query {i:02d}/{len(SEED_QUERIES)}")
        print("\n✅ Neo4j knowledge graph seeded successfully.")
    except Exception as exc:
        print(f"\n❌ Seeding failed: {exc}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    print("🌱 Seeding OmniCell-AI knowledge graph into Neo4j …\n")
    seed_graph()
