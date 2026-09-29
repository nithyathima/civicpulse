import json
import os
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from google.genai import types

# Allows running tests against custom endpoints or default lite model
TEST_MODEL_NAME = os.getenv("GEMINI_TEST_MODEL", "gemini-3.5-flash-lite")


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


image_path = "sample_trash.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Missing {image_path}. Please ensure sample_trash.jpg is present.")

uploaded_image = Image.open(image_path)
citizen_caption = "பஸ் ஸ்டாப் பக்கத்தில் நடைபாதையில் குப்பை கொட்டியுள்ளனர், நடக்கவே வழியில்லை."

system_instruction = """
You are CivicPulse AI's multimodal computer vision engine.
Tasks:
1. Verify if the image genuinely depicts street municipal solid waste.
2. Determine if pedestrian sidewalk/transit movement is blocked.
3. Propose a long-term placemaking deterrent (e.g., Cast-in-situ concrete planters, street vendor stewardship, or painted god-tile mural) to prevent blackspot relapse.
"""

prompt = f"""
Citizen note: "{citizen_caption}"

Analyze the image and citizen text. Respond ONLY with valid JSON:
{{
  "is_valid_civic_issue": bool,
  "detected_waste_types": [string],
  "visual_severity": "Low" | "Medium" | "Severe",
  "transit_pedestrian_obstruction": bool,
  "placemaking_deterrent_plan": "string",
  "infrastructure_hardware_needed": "string"
}}
"""



response = client.models.generate_content(
    model=TEST_MODEL_NAME,
    contents=[uploaded_image, prompt],
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    ),
)

print("--- Updated Multimodal Verification ---")
result = json.loads(response.text)
print(json.dumps(result, indent=2, ensure_ascii=False))