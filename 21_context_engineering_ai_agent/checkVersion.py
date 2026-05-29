from dotenv import load_dotenv
import os
import google.genai
from importlib.metadata import version

load_dotenv()
print(f"[google-genai version]\n{version('google-genai')}")
print(f"[langchain-google-genai version]\n{version('langchain-google-genai')}")
print(f"[GOOGLE_API_KEY configured]\n{bool(os.getenv('GOOGLE_API_KEY'))}")
