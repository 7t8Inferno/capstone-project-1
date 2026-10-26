FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY streaming/requirements.txt /app/streaming/requirements.txt
RUN pip install --no-cache-dir -r /app/streaming/requirements.txt
COPY forest_model.py fraud_model.pkl demo_stream.csv app.py model_metrics.json /app/
COPY streaming/ /app/streaming/
