# Gemini CLI Tool

This tool allows you to authenticate with Google Gemini using a local JavaScript GUI and then use a CLI tool in your terminal to interact with the AI.

## Prerequisites

- Python 3.x
- Google Cloud Project with OAuth 2.0 Client ID and Secret configured.

## Installation

1.  Install dependencies:
    ```bash
    pip install flask google-auth google-auth-oauthlib google-generativeai
    ```
    *(Note: These are already installed in the environment)*

## Setup

1.  Run the setup wizard:
    ```bash
    python3 setup_gemini.py
    ```
2.  Follow the prompts to enter your Client ID and Client Secret.
3.  The wizard will launch a browser (or give you a link) to `http://localhost:5000`.
4.  Click "Sign in with Gemini" and complete the Google login flow.
5.  Once authenticated, the wizard will confirm success and exit.

## Usage

You can use the provided shell script wrapper `gemini`:

```bash
./gemini "Tell me a joke"
```

Or run the python script directly:

```bash
python3 gemini_tool/cli.py "Explain quantum computing"
```

## Troubleshooting

-   **Redirect URI Mismatch**: Ensure your Google Cloud Console has `http://localhost:5000/oauth2callback` added as an Authorized Redirect URI for your OAuth Client.
-   **Token Expired**: If you receive authentication errors, delete `gemini_tool/token.json` and run `setup_gemini.py` again.
