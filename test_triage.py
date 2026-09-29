import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google import genai
import os
from google.genai import types

# Allows running tests against custom endpoints or default lite model
TEST_MODEL_NAME = os.getenv("GEMINI_TEST_MODEL", "gemini-3.5-flash-lite")

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")
client = genai.Client(api_key=API_KEY)


# Citizen complaint in Tamil specifying night dumping and pedestrian littering
sample_citizen_complaint = (
    "மடிவாளா சந்தை அருகே இரவு 11 மணிக்கு மேல் மக்கள் நடந்து வந்து குப்பையை வீசிவிட்டுப் போகிறார்கள். "
    "ரோட்டில் ஒரே நாற்றம், தெரு நாய்கள் தொல்லை. காலையில் வண்டி வரும் போது யாரும் வீட்டில் இருப்பதில்லை. "
    "இங்கே கேமரா வைத்து அபராதம் போட வேண்டும், அல்லது இரவில் குப்பை போட தொட்டி வைக்க வேண்டும்."
)

system_instruction = """
You are CivicPulse AI, a municipal triage and urban enforcement engine for BBMP (Bengaluru).
Your role:
1. Ingest citizen feedback in regional Indian languages (like Tamil) and translate to English.
2. Identify the temporal pattern:
   - 'Daytime Routine': Normal collection issues.
   - 'Night Dump Window': Post-21:00 dumping by night commuters or pedestrians.
3. Classify intervention_type:
   - 'OpEx': Routine litter pickup or marshal dispatch.
   - 'CapEx': Structural need (semi-underground drop-box, motion-sensor lighting, twin bins).
4. Recommend an enforcement anchor (e.g., CCTV to Property Tax/PID surcharge, or e-Challan).
"""

prompt = f"""
Analyze this citizen complaint:
"{sample_citizen_complaint}"

Respond ONLY with valid JSON matching this schema:
{{
  "original_language": "string",
  "translated_summary": "string",
  "locality_or_landmark": "string",
  "temporal_window": "Daytime Routine" | "Night Dump Window",
  "root_cause_diagnosis": "string",
  "intervention_type": "OpEx" | "CapEx",
  "infrastructure_recommendation": "string",
  "enforcement_mechanism": "string"
}}
"""

response = client.models.generate_content(
    model=TEST_MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    ),
)

print("--- Updated CivicPulse Triage ---")
result = json.loads(response.text)
print(json.dumps(result, indent=2, ensure_ascii=False))