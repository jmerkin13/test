from flask import Flask, render_template, redirect, url_for, session, request
import google_auth_oauthlib.flow
import os
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Allow HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), 'client_secrets.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')
SCOPES = ['https://www.googleapis.com/auth/generative-language']

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login')
def login():
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return "Error: client_secrets.json not found. Please run the setup wizard."

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)

    # Indicate where the API server will redirect the user after the user completes
    # the authorization flow. The redirect_uri must match the one configured
    # in the Google Cloud Console.
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session.get('state')
    if not state:
        return "Error: Session state is missing."

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    authorization_response = request.url
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as e:
        return f"Authentication failed: {e}"

    credentials = flow.credentials
    creds_data = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

    with open(TOKEN_FILE, 'w') as f:
        json.dump(creds_data, f)

    return "Authentication successful! You can now close this tab and use the terminal."

if __name__ == '__main__':
    # Running on port 5000 by default
    app.run(host='localhost', port=5000)
