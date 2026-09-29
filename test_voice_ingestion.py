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


# Use your audio file (or sample_grievance.mp3)
audio_file_path = "sample_grievance.m4a"

if not os.path.exists(audio_file_path):
    raise FileNotFoundError(f"Missing {audio_file_path}. Please place an audio file in this directory.")

print(f"Uploading audio file '{audio_file_path}' to Google GenAI storage...")
# Upload the audio using the Files API
audio_file = client.files.upload(file=audio_file_path)
print(f"File uploaded successfully. URI: {audio_file.uri}")

system_instruction = """
You are CivicPulse AI's voice-first municipal intake agent.
Listen carefully to the citizen's audio recording (which may be in Tamil, Kannada, Hindi, or Indian English).

Tasks:
1. Transcribe the spoken audio into its native script.
2. Translate the transcript into standard municipal English.
3. Extract:
   - Specific landmark or street mentioned.
   - Time of dumping mentioned (e.g. night, morning, 11 PM).
   - Core problem (overflowing bin, street dumping, blocked sidewalk).
   - Recommended enforcement or infrastructure (e.g. night marshal patrol, twin bins, semi-underground bin).
"""

prompt = """
Listen to this citizen grievance audio.
Respond ONLY with a valid JSON object matching this schema:
{
  "detected_language": "string",
  "native_transcription": "string",
  "english_translation": "string",
  "extracted_landmark": "string",
  "temporal_window": "Daytime Routine" | "Night Dump Window",
  "urgency_score": float (0.0 to 1.0),
  "prescribed_intervention": "string"
}
"""

print("Processing audio with Gemini...")
response = client.models.generate_content(
    model=TEST_MODEL_NAME,
    contents=[audio_file, prompt],
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    ),
)

print("\n--- Audio Triage Output ---")
result = json.loads(response.text)
print(json.dumps(result, indent=2, ensure_ascii=False))