import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from src.features.scaler import SimpleStandardScaler

class TestSimpleStandardScaler(unittest.TestCase):
    def test_simple_standard_scaler_math(self):
        df = pd.DataFrame({
            "feat_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feat_b": [100.0, 200.0, 300.0, 400.0, 500.0],
            "other": ["a", "b", "c", "d", "e"]
        })

        scaler = SimpleStandardScaler()
        scaler.fit(df, ["feat_a", "feat_b"])

        scaled = scaler.transform(df)

        self.assertTrue(np.isclose(scaled["feat_a"].mean(), 0.0, atol=1e-6))
        self.assertTrue(np.isclose(scaled["feat_a"].std(ddof=1), 1.0, atol=1e-6))
        self.assertTrue(np.isclose(scaled["feat_b"].mean(), 0.0, atol=1e-6))
        self.assertTrue(np.isclose(scaled["feat_b"].std(ddof=1), 1.0, atol=1e-6))

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "scaler_test.json")
            scaler.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_scaler = SimpleStandardScaler.load(save_path)
            loaded_scaled = loaded_scaler.transform(df)
            pd.testing.assert_frame_equal(scaled, loaded_scaled)

if __name__ == "__main__":
    unittest.main()
