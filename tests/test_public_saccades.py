import unittest

import pandas as pd

from gaze_analysis.public_saccades import (
    compute_features,
    select_primary_saccades,
    simulate_directional_restriction,
)


class PublicSaccadeTests(unittest.TestCase):
    def test_primary_selection_rejects_anticipatory_and_secondary_rows(self):
        frame = pd.DataFrame([
            dict(SUBID=1, BLOCK=1, TRIAL=1, PHASE=2, TOT=1, CUR=1, ECC=10,
                 DX=9, DY=0, AMP=9, LAT=180, DIFF=1, ANSW=1, ANTC=0),
            dict(SUBID=1, BLOCK=1, TRIAL=1, PHASE=2, TOT=2, CUR=2, ECC=10,
                 DX=1, DY=0, AMP=1, LAT=250, DIFF=0, ANSW=1, ANTC=0),
            dict(SUBID=1, BLOCK=1, TRIAL=2, PHASE=2, TOT=1, CUR=1, ECC=10,
                 DX=9, DY=0, AMP=9, LAT=60, DIFF=1, ANSW=1, ANTC=1),
        ])
        selected = select_primary_saccades(frame)
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected.iloc[0]["TRIAL"]), 1)

    def test_synthetic_restriction_increases_gain_asymmetry(self):
        base = pd.DataFrame([
            dict(SUBID=1, BLOCK=1, TRIAL=1, PHASE=2, ECC=10, TOT_L=1, TOT_R=1,
                 DX_L=9.5, DX_R=9.5, DY_L=0.1, DY_R=0.1, AMP_L=9.5, AMP_R=9.5,
                 LAT_L=180, LAT_R=182, DIFF_L=1, DIFF_R=1),
        ])
        healthy = compute_features(base)
        synthetic = simulate_directional_restriction(healthy, amplitude_scale=0.5)
        self.assertGreater(
            float(synthetic.iloc[0]["GAIN_ASYMMETRY"]),
            float(healthy.iloc[0]["GAIN_ASYMMETRY"]),
        )
        self.assertEqual(int(synthetic.iloc[0]["LABEL"]), 1)


if __name__ == "__main__":
    unittest.main()
