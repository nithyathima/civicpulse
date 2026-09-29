import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google import genai

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")
client = genai.Client(api_key=API_KEY)


# 2. Synthesize Realistic Geotagged Complaints across Bengaluru
np.random.seed(42)

# Anchor locations: [Latitude, Longitude, Locality Name, Ward Number]
anchors = [
    (12.9226, 77.6174, "Madiwala Market", 172),     # Chronic hotspot 1
    (12.9352, 77.6245, "Koramangala 4th Block", 151),# Mild reports
    (12.9260, 77.6762, "Bellandur Outer Ring", 150), # Chronic hotspot 2
    (13.0033, 77.5647, "Malleswaram 8th Cross", 45), # Isolated complaints
]

records = []
for lat, lon, locality, ward in anchors:
    # Chronic spots get 35-50 recurring complaints; mild spots get 4-6
    n_samples = 45 if "Madiwala" in locality or "Bellandur" in locality else 5
    
    # Add slight spatial variance (within ~300 meters)
    lats = lat + np.random.normal(0, 0.0018, n_samples)
    lons = lon + np.random.normal(0, 0.0018, n_samples)
    
    for i in range(n_samples):
        records.append({
            "complaint_id": f"BLR-{ward}-{i+1000}",
            "ward_number": ward,
            "locality": locality,
            "latitude": round(lats[i], 5),
            "longitude": round(lons[i], 5),
            "category": "Solid Waste",
            "recurrence_days": np.random.randint(3, 20)
        })

df = pd.DataFrame(records)
print(f"Total citizen complaints ingested: {len(df)}")

# 3. Spatial Hotspot Clustering with DBSCAN
# Coordinates to radians for haversine distance metric
coords = np.radians(df[['latitude', 'longitude']])
# 6371000 meters earth radius; 350 meters neighborhood radius
kms_per_radian = 6371.0088
epsilon = 0.35 / kms_per_radian  # 350 meters

db = DBSCAN(eps=epsilon, min_samples=15, metric='haversine')
df['cluster_id'] = db.fit_predict(coords)

# Filter out noise (-1) and isolate systemic infrastructure blackspots
hotspots = df[df['cluster_id'] != -1]
summary = hotspots.groupby(['cluster_id', 'ward_number', 'locality']).agg(
    total_complaints=('complaint_id', 'count'),
    avg_lat=('latitude', 'mean'),
    avg_lon=('longitude', 'mean'),
    avg_recurrence_days=('recurrence_days', 'mean')
).reset_index()

print("\n--- Detected Systemic Infrastructure Deficit Clusters ---")
print(summary.to_string(index=False))

# 4. Synthesize Top Deficit Zone for Gemini Policymaker Agent
top_hotspot = summary.sort_values(by='total_complaints', ascending=False).iloc[0]

# Synthetic infrastructure metadata for that ward (BBMP norms)
ward_meta = {
    "ward_name": top_hotspot['locality'],
    "ward_number": int(top_hotspot['ward_number']),
    "total_monthly_complaints": int(top_hotspot['total_complaints']),
    "avg_unresolved_days": round(top_hotspot['avg_recurrence_days'], 1),
    "estimated_population": 48000,
    "current_dwcc_distance_km": 4.6, # Too far for daily collection
    "nearby_commercial_activity": "High (Vegetable market + eateries)",
    "centroid_coordinates": [round(top_hotspot['avg_lat'], 4), round(top_hotspot['avg_lon'], 4)]
}

# 5. Gemini Policy Agent: Generates Detailed CapEx Project Proposal
system_instruction = """
You are the BRICS Digital Public Good Municipal Policy Synthesis Agent.
You analyze spatial complaint clusters combined with demographic and infrastructure indices.
Your output must be a formal, executive-ready Municipal Capital Expenditure (CapEx) Proposal
adhering to Swachh Bharat Mission 2.0 / Urban Local Body funding guidelines.
"""

prompt = f"""
Analyze this chronic municipal infrastructure deficit cluster:
{json.dumps(ward_meta, indent=2)}

Generate a formal CapEx Infrastructure Proposal in JSON matching this schema:
{{
  "project_title": string,
  "targeted_ward": string,
  "deficiency_diagnosis": string,
  "proposed_infrastructure": string,
  "estimated_capex_inr_lakhs": float,
  "implementation_timeline_months": int,
  "expected_impact": {{
    "complaint_reduction_percent": int,
    "beneficiary_population": int,
    "diversion_from_landfill_tpd": float
  }},
  "brics_replication_note": string
}}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    ),
)

print("\n--- Policymaker CapEx Recommendation (Automated SBM 2.0 DPR) ---")
policy_proposal = json.loads(response.text)
print(json.dumps(policy_proposal, indent=2, ensure_ascii=False))