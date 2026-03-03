# Roadmap – qdrant-proxy

Stand: 2026-02-25

## Ziele
- Performantes, sicheres und nachvollziehbares Vektor-Retrieval
- Vereinfachte Betriebsführung bei wachsenden Collections
- Höhere Relevanz durch bessere Indizierungs- und Query-Strategien

## Kurzfristig (0–2 Monate)
- **Collection-Templates** (Dimension, distance, payload-index presets)
- **Payload-Filter-Validierung** für konsistente Query-Semantik
- **Bulk-Upsert-Endpunkt** mit Backpressure und Chunked-Import
- **Observability-Basics**: Query-Latenz, top-k Größe, Fehlerarten, Timeout-Rate
- **Sichere Defaults** (auth checks, request-size limits, sensible logs maskieren)

## Mittelfristig (2–4 Monate)
- **Reindex-/Migration-Workflows** ohne langen Downtime-Fenster
- **Named Vector Strategien** (dense + sparse + multimodal) per API steuerbar
- **Adaptive HNSW/Quantization Profile** je Collection-Größe und SLA
- **Tenant-Isolation Patterns** (Collection-per-tenant vs payload partitioning)
- **Consistency Checks** zwischen Embedding-Metadaten und Payload-Schema

## Langfristig (4+ Monate)
- **Auto-Tuning für Query-Parameter** basierend auf Relevanz- und Latenzsignalen
- **Lifecycle-Management** für Cold/Hot Collections mit Kostensteuerung
- **Disaster-Recovery Playbooks** (snapshot orchestration, restore drills)
- **Workload-Aware Routing** bei mehreren Qdrant-Backends

## Erfolgskriterien
- p95 Query-Latenz bei Haupt-Workload um >= 20 % reduziert
- Import-Fehlerquote < 0.5 % bei Bulk-Loads
- Betriebsstörungen durch Collection-Migrationen deutlich reduziert
