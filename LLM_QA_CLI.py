import json
import requests
import re
import time

# --- Configuration ---
# NOTE: In a real-world scenario, replace this empty string with your actual API key
# The Canvas environment provides this key during execution, but for local use, you would set it here.
API_KEY = ""
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
MAX_RETRIES = 5

def preprocess_question(question: str) -> str:
    """
    Applies basic preprocessing steps to the user's question.
    - Lowercasing
    - Removes punctuation (except for spaces)
    - Tokenization (implicitly by splitting on space, though not fully standard tokenization)
    """
    # 1. Lowercasing
    text = question.lower()
    # 2. Punctuation removal (keeping only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Simple tokenization by splitting and joining (removes extra spaces)
    tokens = text.split()
    processed_text = ' '.join(tokens)
    return processed_text

def get_llm_answer(prompt: str) -> dict:
    """
    Sends the prompt to the Gemini API with Google Search Grounding.
    Implements exponential backoff for robustness.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Use Google Search for up-to-date and factual grounding
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": "You are a helpful and expert Question-Answering system. Provide a concise and accurate answer based on the query and use Google Search grounding when necessary."}]},
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            
            # Check for content and parts
            candidate = result.get('candidates', [None])[0]
            if not candidate or not candidate.get('content') or not candidate['content'].get('parts'):
                return {"text": "API response was empty or malformed.", "sources": []}

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

            return {"text": text, "sources": sources}

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"[{time.strftime('%H:%M:%S')}] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {"text": f"Error: Failed to connect to API after {MAX_RETRIES} attempts. {e}", "sources": []}
        except Exception as e:
             return {"text": f"An unexpected error occurred: {e}", "sources": []}

def main():
    print("-" * 50)
    print("Welcome to the LLM-Powered Q&A CLI System")
    print("Model: gemini-2.5-flash-preview-09-2025 (Google Search Grounding enabled)")
    print("-" * 50)

    while True:
        try:
            raw_question = input("\nEnter your question (or type 'quit' to exit): \n> ")

            if raw_question.lower() in ('quit', 'exit'):
                print("Exiting Q&A system. Goodbye!")
                break

            if not raw_question.strip():
                continue

            # 1. Preprocessing
            processed_question = preprocess_question(raw_question)
            print(f"\n[Processed Question]: {processed_question}")
            print("\n[Thinking...]")

            # 2. Get Answer from LLM
            response = get_llm_answer(raw_question) # Send the raw question to the LLM for better context

            # 3. Display Final Answer
            print("\n" + "=" * 50)
            print("[LLM Answer]")
            print(response['text'])
            print("=" * 50)

            if response['sources']:
                print("\n[Sources Used]:")
                for i, source in enumerate(response['sources']):
                    print(f"  {i+1}. {source['title']} ({source['uri']})")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting Q&A system. Goodbye!")
            break
        except Exception as e:
            print(f"An application error occurred: {e}")

if __name__ == "__main__":
    main()