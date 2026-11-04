"""Headless Streamlit presentation smoke test; no live payment connection required."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=str(Path(__file__).resolve().parents[1]/"app.py")

class MentorDemoTests(unittest.TestCase):
    def launch(self):
        app=AppTest.from_file(APP,default_timeout=90).run()
        self.assertEqual(len(app.exception),0,str(list(app.exception)))
        return app

    def test_home_page(self):
        app=self.launch()
        titles=[t.value for t in app.title]
        self.assertTrue(any("Card transaction fraud monitoring" in t for t in titles),titles)

    def test_live_monitor_process_and_reset(self):
        app=self.launch()
        app.sidebar.radio[0].set_value("Live monitor").run()
        self.assertEqual(len(app.exception),0,str(list(app.exception)))
        buttons=[x for x in app.button if x.label=="Process next batch"]
        self.assertEqual(len(buttons),1)
        buttons[0].click().run()
        self.assertEqual(len(app.exception),0,str(list(app.exception)))
        self.assertGreaterEqual(len(app.dataframe),1)
        reset=[x for x in app.button if x.label=="Reset simulation"]
        self.assertEqual(len(reset),1)
        reset[0].click().run()
        self.assertEqual(len(app.exception),0,str(list(app.exception)))

    def test_model_evaluation(self):
        app=self.launch()
        app.sidebar.radio[0].set_value("Model evaluation").run()
        self.assertEqual(len(app.exception),0,str(list(app.exception)))
        self.assertTrue(any("Model evaluation" in t.value for t in app.title))
if __name__=="__main__":unittest.main()
