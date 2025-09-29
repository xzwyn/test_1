# tests/test_parser.py

import unittest
import json
import tempfile
import os

# Adjust import path to access the src module
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.parser import extract_paragraphs_from_json

# A sample JSON structure mimicking the Azure DI output
SAMPLE_JSON_CONTENT = {
    "analyzeResult": {
        "content": "Sample Title. This is the first sentence. This is the second sentence. Page 1 A complex paragraph.",
        "paragraphs": [
            {"role": "pageHeader", "spans": [{"offset": 52, "length": 6}]},
            {"role": "title", "spans": [{"offset": 0, "length": 13}]},
            {"spans": [{"offset": 14, "length": 29}]}, # "This is the first sentence."
            {"role": "text", "spans": [{"offset": 44, "length": 29}]}, # "This is the second sentence."
            {"spans": [{"offset": 74, "length": 21}]} # "A complex paragraph." - merged
        ]
    }
}

class TestJsonParser(unittest.TestCase):

    def setUp(self):
        """Create a temporary JSON file for testing."""
        # Create a named temporary file that stays open
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding='utf-8')
        json.dump(SAMPLE_JSON_CONTENT, self.temp_file)
        self.temp_file.close()
        self.temp_filename = self.temp_file.name

    def tearDown(self):
        """Remove the temporary file after the test."""
        os.remove(self.temp_filename)

    def test_extraction_and_stitching(self):
        """
        Tests if the parser correctly filters roles, splits headings, and stitches paragraphs.
        """
        # Run the parser on our temporary file
        result = extract_paragraphs_from_json(self.temp_filename)

        # Define what the clean output should look like
        expected_output = [
            "Sample Title.",
            "This is the first sentence. This is the second sentence.", # These two should be stitched
            "A complex paragraph."
        ]

        # Assert that the output matches the expectation
        self.assertEqual(result, expected_output)
        self.assertEqual(len(result), 3, "Should have found 3 distinct content blocks.")
        self.assertNotIn("Page 1", " ".join(result), "Page header should have been filtered out.")

if __name__ == '__main__':
    unittest.main()