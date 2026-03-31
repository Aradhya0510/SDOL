# SDOL Documentation

Complete documentation for the Semantic Data Orchestration Layer.

---

## Documentation Index

| Document | File | Purpose |
|----------|------|---------|
| **Getting Started** | [getting-started.md](getting-started.md) | Installation, setup, and basic working examples — start here |
| **Architecture** | [architecture.md](architecture.md) | Detailed system design: nine composable layers, data flow, type system, directory structure, and design principles |
| **Typed Connectors Guide** | [typed-connectors-guide.md](typed-connectors-guide.md) | Three-tier connector architecture, built-in connectors (OLAP/OLTP/Document), QueryExecutor protocol, building custom connectors, suitability scoring |
| **Databricks Guide** | [databricks-guide.md](databricks-guide.md) | Databricks-specific integration: DBSQL (Photon/OLAP), Lakebase (OLTP), and Vector Search (Document), Unity Catalog, cross-paradigm queries, writing production QueryExecutors |
| **Implementation Spec** | [implementation-spec.md](implementation-spec.md) | Original milestone-based specification used to build SDOL — 13 milestones covering project bootstrap through end-to-end integration |

---

## Recommended Reading Order

1. **[Getting Started](getting-started.md)** — get the project running and understand core concepts
2. **[Architecture](architecture.md)** — understand how the layers interact and where code lives
3. **[Typed Connectors Guide](typed-connectors-guide.md)** — learn the connector system and how to extend it
4. **[Databricks Guide](databricks-guide.md)** — if using Databricks, read this for DBSQL + Lakebase + Vector Search specifics

The [Implementation Spec](implementation-spec.md) is a reference document — useful for understanding design decisions and original requirements, but not necessary for day-to-day usage.

---

## Project Overview

SDOL sits between AI agents and their data sources. Agents declare *what* they want (typed intents) — never *how* to get it. SDOL handles routing, execution, provenance tracking, trust scoring, conflict resolution, and epistemic context generation automatically.

For a high-level introduction, see the [main README](../README.md).
