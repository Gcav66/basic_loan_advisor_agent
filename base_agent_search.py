
# Now Add Search
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

query = sys.argv[1]

response = client.models.generate_content(
    model="gemini-2.5-flash",   
    contents=query,
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction="Always use google_search to answer. Do not rely on training data."
    )
)

print(response.text)
