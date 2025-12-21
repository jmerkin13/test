import os
import json
import sys
import argparse
import google.generativeai as genai
from google.oauth2.credentials import Credentials

TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')

def load_credentials():
    if not os.path.exists(TOKEN_FILE):
        print("Error: Not authenticated. Please run 'python3 setup_gemini.py' first.")
        sys.exit(1)

    with open(TOKEN_FILE, 'r') as f:
        data = json.load(f)

    creds = Credentials(
        token=data['token'],
        refresh_token=data.get('refresh_token'),
        token_uri=data.get('token_uri'),
        client_id=data.get('client_id'),
        client_secret=data.get('client_secret'),
        scopes=data.get('scopes')
    )
    return creds

def main():
    parser = argparse.ArgumentParser(description="Gemini CLI Tool")
    parser.add_argument('prompt', nargs='+', help="The prompt to send to Gemini")
    parser.add_argument('--model', default='gemini-pro', help="The model to use (default: gemini-pro)")
    args = parser.parse_args()

    creds = load_credentials()

    # Configure the library with the credentials
    # Note: google-generativeai usually takes an API key, but we are using OAuth.
    # We might need to use the `configure` method differently or use the Request wrapper.
    # However, currently google-generativeai is optimized for API Keys.
    # Let's try passing the credentials to `configure`.
    # Based on documentation, `genai.configure(credentials=creds)` is supported.

    try:
        genai.configure(credentials=creds)
        model = genai.GenerativeModel(args.model)

        prompt_text = " ".join(args.prompt)
        response = model.generate_content(prompt_text)

        print(response.text)

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        # Fallback debug info
        # print(f"Debug Info: Creds valid? {creds.valid}")

if __name__ == '__main__':
    main()
