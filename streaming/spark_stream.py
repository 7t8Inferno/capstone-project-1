"""Kafka -> Spark Structured Streaming -> bounded driver scoring -> SQLite demo review queue.
Research-only pipeline. Spark microbatches are collected on the driver: NOT a scalable production implementation.
"""
import argparse, json, os, pickle, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType,StructField,StringType,MapType,DoubleType

FEATURES=["Time",*[f"V{i}" for i in range(1,29)],"Amount"]
DDL="""CREATE TABLE IF NOT EXISTS scored_events (
 event_id TEXT PRIMARY KEY, sent_at TEXT NOT NULL, scored_at TEXT NOT NULL,
 amount REAL NOT NULL, score REAL NOT NULL, decision TEXT NOT NULL,
 detection_latency_ms REAL, source TEXT NOT NULL
)"""
def parse_event(raw):
    obj=json.loads(raw)
    if obj.get("source")!="synthetic":raise ValueError("Only synthetic events are permitted")
    f=obj["features"]
    if set(f)!=set(FEATURES):raise ValueError("Feature mismatch")
    values=[float(f[k]) for k in FEATURES]
    import math
    if not all(map(math.isfinite,values)) or values[0]<0 or values[-1]<0:raise ValueError("Invalid values")
    return obj,values

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--broker",default=os.getenv("KAFKA_BOOTSTRAP_SERVERS","kafka:9092"))
    arg.add_argument("--topic",default="synthetic-transactions")
    arg.add_argument("--db",default="/state/scored.sqlite")
    arg.add_argument("--checkpoint",default="/state/checkpoints/synthetic")
    arg.add_argument("--model",default="/app/fraud_model.pkl")
    a=arg.parse_args()
    # Load only the trusted model artifact supplied by this repository.
    with open(a.model,"rb") as f:bundle=pickle.load(f)
    model=bundle["model"]; threshold=float(bundle["threshold"])
    os.makedirs(os.path.dirname(a.db),exist_ok=True)
    conn=sqlite3.connect(a.db,timeout=30)
    conn.execute("PRAGMA journal_mode=WAL");conn.execute(DDL);conn.commit();conn.close()
    spark=SparkSession.builder.appName("CapstoneSyntheticFraud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    schema=StructType([StructField("event_id",StringType()),StructField("sent_at",StringType()),StructField("source",StringType()),StructField("features",MapType(StringType(),DoubleType()))])
    source=(spark.readStream.format("kafka").option("kafka.bootstrap.servers",a.broker)
            .option("subscribe",a.topic).option("startingOffsets","earliest").load())
    messages=source.selectExpr("CAST(value AS STRING) AS payload")
    def score_batch(df,batch_id):
        # Local prototype safeguard: do not collect unbounded transactions.
        entries=df.limit(1001).collect()
        if len(entries)>1000:raise RuntimeError("Microbatch exceeds safe 1000-event driver limit")
        parsed=[]
        for row in entries:
            try: parsed.append(parse_event(row.payload))
            except (KeyError,ValueError,TypeError,OverflowError,json.JSONDecodeError) as ex:
                print("Rejected invalid synthetic event:",str(ex))
        if not parsed:return
        import numpy as np
        xs=np.asarray([v for obj,v in parsed],dtype=np.float32)
        scores=model.predict_proba(xs)[:,1]
        stamp=datetime.now(timezone.utc)
        output=[]
        for (obj,features),s in zip(parsed,scores):
            try:
                sent=datetime.fromisoformat(obj["sent_at"].replace("Z","+00:00"))
                if sent.tzinfo is None:raise ValueError("Timestamp must be timezone-aware")
                latency=max(0,(stamp-sent).total_seconds()*1000)
            except (ValueError,TypeError,KeyError):latency=None
            output.append((str(obj["event_id"]),str(obj["sent_at"]),stamp.isoformat(),features[-1],float(s),
                           "Review" if s>=threshold else "Clear",latency,"synthetic"))
        with sqlite3.connect(a.db,timeout=30) as db:
            db.executemany("INSERT OR IGNORE INTO scored_events VALUES (?,?,?,?,?,?,?,?)",output)
        print(f"Spark microbatch {batch_id}: {len(output)} synthetic events examined")
    query=(messages.writeStream.foreachBatch(score_batch).option("checkpointLocation",a.checkpoint)
           .trigger(processingTime="5 seconds").start())
    query.awaitTermination()
if __name__=="__main__":main()
