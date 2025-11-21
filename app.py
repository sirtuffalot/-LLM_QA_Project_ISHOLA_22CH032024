import json
import requests
import os
import time
from flask import Flask, render_template, request, jsonify

# Flask App Initialization
app = Flask(__name__)

# --- LLM Configuration ---
# NOTE: The Canvas environment provides this key during execution, but for local use, you would set it via environment variables.
# We initialize it as an empty string, which the Canvas runtime will handle.
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
MAX_RETRIES = 5

def get_llm_answer_api(prompt: str) -> dict:
    """
    Handles the LLM API call logic for the web application.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Enable Google Search grounding
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": "You are a helpful and expert Question-Answering system. Provide a concise, clear, and factual answer based on the query, citing sources if used."}]},
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            candidate = result.get('candidates', [None])[0]

            if not candidate or not candidate.get('content') or not candidate['content'].get('parts'):
                return {"answer": "Error: API response was empty or malformed.", "sources": []}

            text = candidate['content']['parts'][0]['text']
            sources = []
            
            # Extract Grounding Sources
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = [
                    {
                        "uri": attr['web']['uri'],
                        "title": attr['web']['title']
                    }
                    for attr in grounding_metadata['groundingAttributions']
                    if 'web' in attr and 'uri' in attr['web'] and 'title' in attr['web']
                ]

            return {"answer": text, "sources": sources}

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {"answer": f"API Error: Failed to connect to LLM after {MAX_RETRIES} attempts. {e}", "sources": []}
        except Exception as e:
             return {"answer": f"An unexpected error occurred: {e}", "sources": []}


# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_llm():
    """Endpoint to handle question submission and call the LLM."""
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"answer": "Please provide a question.", "sources": []}), 400

    # Call the LLM API
    response = get_llm_answer_api(question)

    return jsonify(response)

# Required for Render deployment
if __name__ == '__main__':
    # Running on localhost for testing
    app.run(debug=True, host='0.0.0.0', port=5000)