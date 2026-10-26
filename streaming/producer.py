"""Publish synthetic feature-only transactions to a Kafka topic (not real payments)."""
import argparse, csv, json, os, time, uuid
from datetime import datetime, timezone
from kafka import KafkaProducer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--broker",default=os.getenv("KAFKA_BOOTSTRAP_SERVERS","kafka:9092"))
    p.add_argument("--topic",default="synthetic-transactions")
    p.add_argument("--csv",default="/app/demo_stream.csv")
    p.add_argument("--interval",type=float,default=0.25)
    a=p.parse_args()
    producer=KafkaProducer(bootstrap_servers=a.broker.split(","),value_serializer=lambda x:json.dumps(x,allow_nan=False).encode("utf-8"),acks="all",retries=5)
    with open(a.csv,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            features={k:float(row[k]) for k in ["Time",*[f"V{i}" for i in range(1,29)],"Amount"]}
            event={"event_id":str(uuid.uuid4()),"sent_at":datetime.now(timezone.utc).isoformat(),"source":"synthetic","features":features}
            producer.send(a.topic,event).get(timeout=30)
            time.sleep(max(0,a.interval))
    producer.flush()
if __name__=="__main__":main()
