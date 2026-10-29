import unittest
from garment_analysis.cleaning import clean_column_name

class ColumnNameTests(unittest.TestCase):
    def test_clean_column_name(self):
        self.assertEqual(clean_column_name("Rain test Average"), "Rain_test_Average")
        self.assertEqual(clean_column_name("  Spray-test (Average) "), "Spraytest_Average")
