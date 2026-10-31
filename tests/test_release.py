"""Release smoke checks, runnable without Streamlit Cloud or a Kafka broker."""
import ast, hashlib, json, math, pickle, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    def test_python_syntax(self):
        for file in [ROOT/"app.py",ROOT/"forest_model.py",ROOT/"train_model.py",ROOT/"streaming"/"producer.py",ROOT/"streaming"/"spark_stream.py",ROOT/"streaming"/"streaming_status.py"]:
            ast.parse(file.read_text(encoding="utf-8"),filename=str(file))
    def test_artifact_and_metrics(self):
        raw=(ROOT/"fraud_model.pkl").read_bytes()
        expected="232ad91f3ea13dc956662b3718f2381b667590565c4ddd39c7cf5bd5792f6d5f"
        self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
        bundle=pickle.loads(raw)
        info=json.loads((ROOT/"model_metrics.json").read_text())
        self.assertEqual(len(bundle["model"].trees),45)
        self.assertEqual(len(bundle["features"]),30)
        self.assertAlmostEqual(bundle["threshold"],info["decision_threshold"],places=10)
        cm=info["test"]["confusion_matrix"]
        self.assertEqual(sum(cm.values()),info["test"]["test_transactions"])
        self.assertAlmostEqual(cm["tp"]/(cm["tp"]+cm["fp"]),info["test"]["precision"])
        self.assertAlmostEqual(cm["tp"]/(cm["tp"]+cm["fn"]),info["test"]["recall"])
    def test_synthetic_demo_scoring(self):
        bundle=pickle.loads((ROOT/"fraud_model.pkl").read_bytes())
        data=pd.read_csv(ROOT/"demo_stream.csv")
        self.assertEqual(len(data),1200)
        x=data.loc[:,bundle["features"]].astype(float).to_numpy(dtype=np.float32)
        self.assertTrue(np.isfinite(x).all())
        score=bundle["model"].predict_proba(x[:48])
        self.assertEqual(score.shape,(48,2))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score>=0)&(score<=1)).all())
        self.assertTrue(np.allclose(score.sum(axis=1),1))
        self.assertEqual(set(data["profile_type"].unique()),{"Typical profile","Elevated profile"})
if __name__=="__main__":unittest.main()
