# Kafka–Spark optional local pipeline (development branch)

This is an **experimental local simulation**, not a bank connection or a production fraud system.

Data flow: generated PCA-like feature vectors in `demo_stream.csv` -> Kafka -> Spark Structured Streaming microbatches -> repository model -> local SQLite -> the Streamlit `Streaming pipeline` page.

Run from the repo root: `docker compose -f streaming/docker-compose.yml up --build`. Open http://localhost:8501 and choose Streaming pipeline. The broker/spark environment requires Docker and internet for Maven packages. Depending on image resolution and architecture, local adjustments may be needed. **This pipeline has not been end-to-end executed in the assistant environment.**

Only synthetic records are permitted. Driver collection is intentionally capped at 1,000 rows per microbatch. There is no personal payment data, horizontal scaling, durable alert service, access-control system for SQLite, or production-grade observability. A score is not a calibrated fraud probability. Latency reflects synthetic event send-to-score completion, *not* payment network processing.

The public Streamlit Community Cloud website cannot connect to your local Docker network or read your local SQLite file. It continues to show the original manual synthetic demo unless you separately deploy an integrated backend.

For repeatable validation: check Kafka and Spark logs, verify `streaming/state/scored.sqlite`, compare event counts, test duplicate event IDs, measure latency percentiles, and restart Spark to evaluate checkpoint recovery before asserting end-to-end success.
