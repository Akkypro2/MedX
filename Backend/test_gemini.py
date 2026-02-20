import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found in .env")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash') # Use a stable model for testing
        response = model.generate_content("Hello, how are you?")
        print("Gemini API Test Successful!")
        print(response.text)

        # Test embedding
        embedding = genai.embed_content(model="models/text-embedding-004", content="Test embedding", task_type="RETRIEVAL_QUERY")
        print("Embedding API Test Successful!")

    except Exception as e:
        print(f"Gemini API Test Failed: {e}")