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
