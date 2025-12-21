import os
import json
import time
import subprocess
import webbrowser
import sys

# Resolve paths relative to the script location, not the current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_DIR = os.path.join(SCRIPT_DIR, 'gemini_tool')
CLIENT_SECRETS_FILE = os.path.join(GEMINI_DIR, 'client_secrets.json')
TOKEN_FILE = os.path.join(GEMINI_DIR, 'token.json')

def prompt_for_secrets():
    print("\n=== Gemini CLI Setup Wizard ===")
    print("To use this tool, you need an OAuth 2.0 Client ID and Client Secret from Google Cloud.")
    print("1. Go to https://console.cloud.google.com/apis/credentials")
    print("2. Create an OAuth 2.0 Client ID (Application type: Web application).")
    print("3. Add 'http://localhost:5000/oauth2callback' to Authorized redirect URIs.")
    print("===============================\n")

    client_id = input("Enter Client ID: ").strip()
    client_secret = input("Enter Client Secret: ").strip()

    data = {
        "web": {
            "client_id": client_id,
            "project_id": "gemini-cli-tool",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost:5000/oauth2callback"]
        }
    }

    with open(CLIENT_SECRETS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved secrets to {CLIENT_SECRETS_FILE}")

def run_auth_server():
    print("Starting authentication server...")
    server_script = os.path.join(GEMINI_DIR, 'server.py')
    # Start the server in a separate process
    server_process = subprocess.Popen([sys.executable, server_script])

    print("\nServer started at http://localhost:5000")
    print("Opening browser...")
    time.sleep(1) # Give server a moment to start

    # Try to open browser, but might fail in headless. That's okay, we print the URL.
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

    print("If the browser did not open, please visit http://localhost:5000 to complete login.")

    return server_process

def wait_for_token(server_process):
    print("Waiting for authentication to complete...")
    try:
        while not os.path.exists(TOKEN_FILE):
            time.sleep(1)
        print("\nAuthentication successful! Token received.")
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
    finally:
        server_process.terminate()
        print("Server stopped.")

def main():
    if not os.path.exists(GEMINI_DIR):
        os.makedirs(GEMINI_DIR)

    if not os.path.exists(CLIENT_SECRETS_FILE):
        prompt_for_secrets()
    else:
        print(f"Found existing {CLIENT_SECRETS_FILE}")
        choice = input("Do you want to use these secrets? (y/n): ").lower()
        if choice == 'n':
            prompt_for_secrets()

    if os.path.exists(TOKEN_FILE):
        print("Token file already exists. You are logged in.")
        choice = input("Do you want to re-authenticate? (y/n): ").lower()
        if choice != 'y':
            print("Setup complete.")
            return

    server_process = run_auth_server()
    wait_for_token(server_process)
    print("\nSetup complete. You can now use the 'gemini' command (after aliasing).")

if __name__ == '__main__':
    main()
