"""List Gemini models available to GEMINI_API_KEY: python -m app.list_models"""
import json
import urllib.request

from app.config import GEMINI_API_KEY

url = f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key={GEMINI_API_KEY}"
for m in json.load(urllib.request.urlopen(url))["models"]:
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"].removeprefix("models/"), "-", m.get("displayName", ""))
