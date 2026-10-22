
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key
load_dotenv()

# Connect to Gemini
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Send a request to Gemini
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What are the benefits of drinking water?",
    config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=200
    )
)

print(response.text)
