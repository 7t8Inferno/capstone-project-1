"""Tests for public, feature-only CSV upload policy. Uses only pandas/numpy."""
import ast
import io
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
FEATURES = json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))["features"]

def validator():
    module = ast.parse(SOURCE)
    definition = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name == "validate")
    scope = {"io":io,"pd":pd,"np":np,"FEATURES":FEATURES,"MAX_UPLOAD_BYTES":1024*1024,"MAX_UPLOAD_ROWS":1000}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(ROOT/"app.py"),"exec"),scope)
    return scope["validate"]

class UploadedCSV:
    def __init__(self, data):
        self.raw = data if isinstance(data,bytes) else data.encode("utf-8")
    def getvalue(self):
        return self.raw

def frame(n=2):
    data = {k:[0.1]*n for k in FEATURES}
    data["Time"]=[100.0]*n
    data["Amount"]=[25.0]*n
    return pd.DataFrame(data)

class PublicUploadTests(unittest.TestCase):
    def test_no_login_gate(self):
        self.assertNotIn("scoring_access()",SOURCE)
        self.assertNotIn("st.login",SOURCE)
        self.assertNotIn("st.user",SOURCE)
        self.assertIn('st.file_uploader("Batch transaction CSV"',SOURCE)
        self.assertIn("Download sample synthetic CSV",SOURCE)
    def test_valid_csv(self):
        self.assertEqual(len(validator()(UploadedCSV(frame().to_csv(index=False)))),2)
    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError,"Missing required fields"):
            validator()(UploadedCSV(frame().drop(columns=["V28"]).to_csv(index=False)))
    def test_extra_column_or_label(self):
        f=frame();f["Class"]=1
        with self.assertRaisesRegex(ValueError,"Unexpected fields"):
            validator()(UploadedCSV(f.to_csv(index=False)))
    def test_invalid_numeric(self):
        f=frame();f["V1"]=f["V1"].astype(object);f.loc[0,"V1"]="not-a-number"
        with self.assertRaisesRegex(ValueError,"finite number"):
            validator()(UploadedCSV(f.to_csv(index=False)))
    def test_negative_amount(self):
        f=frame();f.loc[0,"Amount"]=-1
        with self.assertRaisesRegex(ValueError,"Time and Amount"):
            validator()(UploadedCSV(f.to_csv(index=False)))
    def test_row_cap(self):
        with self.assertRaisesRegex(ValueError,"no more than 1,000"):
            validator()(UploadedCSV(frame(1001).to_csv(index=False)))
    def test_file_size_cap(self):
        with self.assertRaisesRegex(ValueError,"1 MB"):
            validator()(UploadedCSV(b"x"*(1024*1024+1)))

if __name__ == "__main__":
    unittest.main()
