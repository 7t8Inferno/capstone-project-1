"""Read-only local Spark result viewer. Never equate synthetic examples with live payments."""
import os,sqlite3
from pathlib import Path
import pandas as pd
DB=Path(os.getenv("STREAMING_DB","/state/scored.sqlite"))
def read_recent(limit=100):
    if not DB.exists():return pd.DataFrame()
    try:
        with sqlite3.connect(f"file:{DB}?mode=ro",uri=True,timeout=2) as db:
            return pd.read_sql_query("SELECT event_id,sent_at,scored_at,amount,score,decision,detection_latency_ms,source FROM scored_events ORDER BY scored_at DESC LIMIT ?",db,params=(int(min(limit,500)),))
    except (sqlite3.Error,OSError):return pd.DataFrame()
