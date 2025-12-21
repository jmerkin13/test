import unittest
from unittest.mock import patch, MagicMock
import os
import json
import sys

# Add the directory to path so we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gemini_tool import cli

class TestGeminiCLI(unittest.TestCase):

    @patch('gemini_tool.cli.os.path.exists')
    def test_load_credentials_missing_file(self, mock_exists):
        mock_exists.return_value = False

        # Capture stdout/stderr or check for SystemExit
        with self.assertRaises(SystemExit) as cm:
            cli.load_credentials()
        self.assertEqual(cm.exception.code, 1)

    @patch('gemini_tool.cli.os.path.exists')
    @patch('builtins.open')
    @patch('gemini_tool.cli.json.load')
    def test_load_credentials_success(self, mock_json_load, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_json_load.return_value = {
            'token': 'fake_token',
            'refresh_token': 'fake_refresh',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'client_id': 'fake_id',
            'client_secret': 'fake_secret',
            'scopes': ['scope1']
        }

        creds = cli.load_credentials()
        self.assertEqual(creds.token, 'fake_token')
        self.assertEqual(creds.client_id, 'fake_id')

if __name__ == '__main__':
    unittest.main()
