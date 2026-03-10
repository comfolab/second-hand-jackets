from garment_analysis.cleaning import clean_column_name

def test_clean_column_name():
    assert clean_column_name("Rain test Average") == "Rain_test_Average"
    assert clean_column_name("  Spray-test (Average) ") == "Spraytest_Average"