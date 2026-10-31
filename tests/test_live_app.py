import ast,hashlib,json,pickle,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ReleaseSmoke(unittest.TestCase):
 def test_python_syntax(self):
  for f in ("app.py","train_model.py","forest_model.py"):
   ast.parse((ROOT/f).read_text(),filename=f)
 def test_model_and_results(self):
  raw=(ROOT/"fraud_model.pkl").read_bytes()
  self.assertEqual(hashlib.sha256(raw).hexdigest(),"232ad91f3ea13dc956662b3718f2381b667590565c4ddd39c7cf5bd5792f6d5f")
  b=pickle.loads(raw);m=json.loads((ROOT/"model_metrics.json").read_text())
  self.assertEqual(len(b["model"].trees),45)
  self.assertEqual(len(b["features"]),30)
  cm=m["test"]["confusion_matrix"]
  self.assertEqual(sum(cm.values()),m["test"]["test_transactions"])
  self.assertAlmostEqual(cm["tp"]/(cm["tp"]+cm["fp"]),m["test"]["precision"])
  self.assertAlmostEqual(cm["tp"]/(cm["tp"]+cm["fn"]),m["test"]["recall"])
 def test_demo_predictions(self):
  b=pickle.loads((ROOT/"fraud_model.pkl").read_bytes())
  d=pd.read_csv(ROOT/"demo_stream.csv")
  self.assertEqual(len(d),1200)
  s=b["model"].predict_proba(d[b["features"]].iloc[:48].to_numpy(dtype=np.float32))
  self.assertEqual(s.shape,(48,2))
  self.assertTrue(np.isfinite(s).all())
  self.assertTrue(np.allclose(s.sum(axis=1),1))
if __name__=="__main__": unittest.main()
