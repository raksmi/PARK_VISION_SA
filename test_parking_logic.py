"""
Unit tests for parking_logic.py.

Run with:
    python -m unittest test_parking_logic.py -v

These cover the congestion-threshold boundaries (Low < 40%, Moderate
40-75%, High > 75%) and the edge case of an image with no detections,
which is the kind of automated test the assignment's "Testing and
Feedback" step asks for, in addition to manually testing the app on
unseen images (see predict.py / sample_images/).
"""

import unittest
from parking_logic import parking_insights


class TestParkingInsights(unittest.TestCase):

    def test_no_slots_detected(self):
        result = parking_insights(occupied=0, empty=0)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["congestion"], "No data")

    def test_low_congestion(self):
        # 2/10 occupied = 20% -> Low
        result = parking_insights(occupied=2, empty=8)
        self.assertEqual(result["total"], 10)
        self.assertEqual(result["available"], 8)
        self.assertAlmostEqual(result["occupancy"], 20.0)
        self.assertEqual(result["congestion"], "Low")

    def test_moderate_congestion_lower_boundary(self):
        # exactly 40% -> Moderate (per "40-75% -> Moderate" spec)
        result = parking_insights(occupied=4, empty=6)
        self.assertAlmostEqual(result["occupancy"], 40.0)
        self.assertEqual(result["congestion"], "Moderate")

    def test_moderate_congestion_upper_boundary(self):
        # exactly 75% -> still Moderate
        result = parking_insights(occupied=75, empty=25)
        self.assertAlmostEqual(result["occupancy"], 75.0)
        self.assertEqual(result["congestion"], "Moderate")

    def test_high_congestion(self):
        # 90% -> High
        result = parking_insights(occupied=90, empty=10)
        self.assertEqual(result["congestion"], "High")
        self.assertIn("another area", result["recommendation"].lower())

    def test_full_lot(self):
        result = parking_insights(occupied=20, empty=0)
        self.assertEqual(result["available"], 0)
        self.assertEqual(result["occupancy"], 100.0)
        self.assertEqual(result["congestion"], "High")


if __name__ == "__main__":
    unittest.main()
