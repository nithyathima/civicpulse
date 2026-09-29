import json
import os
import folium
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from google import genai
from google.genai import types
from bigquery_client import fetch_spatial_clusters
from dotenv import load_dotenv
from google import genai

# -------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="CivicPulse | Waste Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #f8fafc; color: #0f172a; }
    
    .demo-strip {
        background: #fef3c7;
        border: 1px solid #fde68a;
        color: #92400e;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        margin-bottom: 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .brand-tagline {
        font-size: 0.85rem;
        color: #64748b;
        margin: 2px 0 0 0;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .kpi-lbl {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    .score-breakdown-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CLIENT INITIALIZATION & DETERMINISTIC FORMULA ENGINE
# -------------------------------------------------------------
# Load key from .env file locally
load_dotenv()

# Safely read from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Missing GEMINI_API_KEY. Please set it in your environment or .env file.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def calculate_infrastructure_priority(report_density, recurrence_days, pop_exposure, env_risk_flag, service_gap_flag):
    """
    Deterministic Priority Scoring Engine (0 - 100)
    Density (30) + Recurrence (25) + Population (20) + Environment (15) + Service Gap (10)
    """
    score_density = min(30, int((report_density / 50.0) * 30))
    score_recurrence = min(25, int((recurrence_days / 30.0) * 25))
    score_population = min(20, int((pop_exposure / 50000.0) * 20))
    score_env = 15 if env_risk_flag else 4
    score_gap = 10 if service_gap_flag else 2
    
    total = score_density + score_recurrence + score_population + score_env + score_gap
    return {
        "total": total,
        "density": score_density,
        "recurrence": score_recurrence,
        "exposure": score_population,
        "env_risk": score_env,
        "service_gap": score_gap
    }

# -------------------------------------------------------------
# HEADER WITH DEMO DATA FLAG
# -------------------------------------------------------------
st.markdown('<span class="demo-strip">Demo Data • Simulated Bengaluru Pilot</span>', unsafe_allow_html=True)
st.markdown("""
<div class="brand-container">
    <div>
        <h1 class="brand-title">🌱 CivicPulse</h1>
        <p class="brand-tagline">Responsible AI for Municipal Waste Intelligence & Infrastructure Allocation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIMPLIFIED ROLE SELECTOR
# -------------------------------------------------------------
selected_role = st.radio(
    "Active Workspace:",
    ["👤 Citizen", "🛡️ Ward Officer", "🏛️ Commissioner"],
    horizontal=True
)

st.write("")

# -------------------------------------------------------------
# 👤 CITIZEN VIEW
# -------------------------------------------------------------
if selected_role == "👤 Citizen":
    c_tabs = st.tabs(["📝 Report Waste", "🗺️ My Area Waste Map", "🌿 Community Clean Status"])

    # 1. Report Waste
    with c_tabs[0]:
        st.subheader("Report a Cleanliness Issue")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            uploaded_audio = st.file_uploader("Upload Audio Grievance (Tamil, Kannada, Hindi, English)", type=["m4a", "mp3", "wav"])
            use_sample = False
            if not uploaded_audio and os.path.exists("sample_grievance.m4a"):
                use_sample = st.checkbox("Use local sample audio: sample_grievance.m4a (Tamil)", value=True)

            user_text = st.text_area("Or type description:", value="மடிப்பாக்கம் சந்தை கிட்ட குப்பை கொட்டி வச்சிருக்காங்க.", height=75)
            
            p1, p2 = st.columns(2)
            with p1:
                near_bus = st.checkbox("Near Bus Stop / Transit", value=True)
            with p2:
                near_lake = st.checkbox("Near Lake / Storm Drain", value=False)

            uploaded_img = st.file_uploader("Upload on-ground photo", type=["jpg", "jpeg", "png"])
            if not uploaded_img and os.path.exists("sample_trash.jpg"):
                preview_img = Image.open("sample_trash.jpg")
                st.image(preview_img, caption="Default Verification Photo", width=220)
            elif uploaded_img:
                preview_img = Image.open(uploaded_img)
                st.image(preview_img, caption="Attached photo", width=220)
            else:
                preview_img = None

            submit_citizen = st.button("Submit Report", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### AI Verification & Immediate Routing")
            if submit_citizen:
                with st.spinner("AI checking evidence and translating..."):
                    payload = []
                    if uploaded_audio:
                        temp_p = "temp_user_audio.m4a"
                        with open(temp_p, "wb") as f:
                            f.write(uploaded_audio.getbuffer())
                        payload.append(client.files.upload(file=temp_p))
                    elif use_sample and os.path.exists("sample_grievance.m4a"):
                        payload.append(client.files.upload(file="sample_grievance.m4a"))

                    if preview_img:
                        payload.append(preview_img)

                    sys_inst = """
                    You are CivicPulse AI assistant.
                    Translate input to clear English. Verify if photo shows trash.
                    Extract landmark and timing (Daytime Routine or Night Dump).
                    """
                    prompt = f"""
                    Citizen text: "{user_text}". Near bus: {near_bus}, Near water: {near_lake}
                    Output JSON:
                    {{
                      "detected_language": string,
                      "english_summary": string,
                      "landmark": string,
                      "is_valid_issue": bool,
                      "timing_pattern": "Daytime Routine" | "Night Dump Window"
                    }}
                    """
                    payload.append(prompt)

                    try:
                        resp = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=payload,
                            config=types.GenerateContentConfig(system_instruction=sys_inst, response_mime_type="application/json")
                        )
                        res = json.loads(resp.text)
                        st.success(f"Report Registered for {res.get('landmark', 'Identified Area')}")
                        st.markdown(f"**Translated:** {res.get('english_summary')}")
                        st.markdown(f"**Detected Language:** `{res.get('detected_language')}`")
                        st.markdown(f"**Temporal Flag:** `{res.get('timing_pattern')}`")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Submit on the left to see your report processed.")

    # 2. My Area Waste Map
    with c_tabs[1]:
        st.subheader("Local Ward Waste Map")
        st.caption("Verified public collection points and active cleanup requests in your neighborhood.")
        blr_map = folium.Map(location=[12.925, 77.62], zoom_start=13, tiles="OpenStreetMap")
        folium.CircleMarker([12.9226, 77.6174], radius=6, color="#ef4444", fill=True, tooltip="Active Cleanup Request: Madiwala").add_to(blr_map)
        folium.CircleMarker([12.9352, 77.6245], radius=5, color="#10b981", fill=True, tooltip="Cleaned & Monitored: Koramangala").add_to(blr_map)
        st_folium(blr_map, width=1000, height=400)

    # 3. Community Clean Status
    with c_tabs[2]:
        st.subheader("Community Cleanliness Streaks")
        st.caption("Streets maintaining source segregation and clean footpaths.")
        st.dataframe(pd.DataFrame([
            {"Street / Area": "14th Main, HSR Layout", "Compliance": "94%", "Streak": "42 Days", "Status": "🏆 Emerald"},
            {"Street / Area": "8th Cross, Malleswaram", "Compliance": "91%", "Streak": "33 Days", "Status": "🏆 Emerald"},
            {"Street / Area": "Market Road, Madiwala", "Compliance": "64%", "Streak": "3 Days", "Status": "🟡 Improving"}
        ]), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# 🛡️ WARD OFFICER VIEW (OPERATIONAL INTELLIGENCE & PATROLS)
# -------------------------------------------------------------
elif selected_role == "🛡️ Ward Officer":
    o_tabs = st.tabs([
        "📥 Incoming AI Reports", 
        "📍 Waste Hotspots", 
        "🌙 Collection Demand Mismatch", 
        "🪴 Interventions"
    ])

    # ---------------------------------------------------------
    # TAB 1: INCOMING AI REPORTS (TRIAGE QUEUE)
    # ---------------------------------------------------------
    with o_tabs[0]:
        st.subheader("Live Grievance Triage Queue")
        st.caption("Citizen voice notes, images, and text automatically categorized and translated via Gemini.")
        
        queue_data = pd.DataFrame([
            {
                "Ticket ID": "TCK-1092",
                "Locality": "Madipakkam Market",
                "Language": "Tamil",
                "Translated Summary": "Night dumping near bus bay after vegetable stalls close.",
                "Temporal Flag": "Night Dump Window",
                "Status": "Action Required"
            },
            {
                "Ticket ID": "TCK-1088",
                "Locality": "Bellandur Canal Bridge",
                "Language": "Kannada",
                "Translated Summary": "Mixed plastic and debris dumped along canal bridge railing.",
                "Temporal Flag": "High Ecological Hazard",
                "Status": "Priority Escalated"
            },
            {
                "Ticket ID": "TCK-1075",
                "Locality": "Koramangala 4th Block",
                "Language": "English",
                "Translated Summary": "Commercial cardboard cartons left on sidewalk overnight.",
                "Temporal Flag": "Daytime Routine",
                "Status": "Assigned to Sweeper"
            }
        ])
        st.dataframe(queue_data, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 2: WASTE HOTSPOTS (DBSCAN GEOSPATIAL CLUSTERING)
    # ---------------------------------------------------------
    with o_tabs[1]:
        st.subheader("Spatial Hotspot Clustering")
        st.caption("DBSCAN clusters isolating recurring blackspots from one-off transient litter.")
        
        # Base interactive map (OpenStreetMap tiles, no API key needed)
        blr_map = folium.Map(location=[12.93, 77.63], zoom_start=12, tiles="OpenStreetMap")
        
        # Red: High-density chronic commercial cluster
        folium.CircleMarker(
            location=[12.9226, 77.6174],
            radius=9,
            color="#ef4444",
            fill=True,
            fill_opacity=0.8,
            tooltip="Cluster #1: Madiwala Market (45 recurring incidents)"
        ).add_to(blr_map)
        
        # Purple: 100m Waterbody Buffer / Canal Corridor
        folium.CircleMarker(
            location=[12.9260, 77.6762],
            radius=9,
            color="#7c3aed",
            fill=True,
            fill_opacity=0.8,
            tooltip="Cluster #2: Bellandur Canal Buffer (48 recurring incidents - Eco Risk)"
        ).add_to(blr_map)

        # Blue: Low-density transient reports
        folium.CircleMarker(
            location=[12.9352, 77.6245],
            radius=4,
            color="#3b82f6",
            fill=True,
            fill_opacity=0.6,
            tooltip="Transient Litter: Koramangala 4th Block"
        ).add_to(blr_map)
        
        st_folium(blr_map, width=1000, height=400)

        # Legend
        l1, l2, l3 = st.columns(3)
        l1.markdown("🔴 **Chronic Deficit ($\ge$ 15 incidents):** Requires infrastructure/timing fix.")
        l2.markdown("🟣 **Waterbody Buffer (100m):** High ecological hazard; priority canal fencing.")
        l3.markdown("🔵 **Isolated Transient Litter:** Handled by standard street sweepers.")

   # ---------------------------------------------------------
    # TAB 3: COLLECTION DEMAND MISMATCH (COMPACT & SLEEK)
    # ---------------------------------------------------------
    with o_tabs[2]:
        st.markdown("##### Temporal Demand vs. Collection Window")
        st.caption("Operational bottleneck: Morning routes miss late-night vendor and commuter activity.")

        # Compact Side-by-Side Comparison
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #b45309; text-transform: uppercase;">Peak Street Dumping</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; margin-top: 2px;">10:30 PM – 1:30 AM</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">68% of Total Daily Discards</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #047857; text-transform: uppercase;">Scheduled Tipper Sweeps</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; margin-top: 2px;">7:30 AM – 10:00 AM</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Only 14% Daily Volume Present</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**Disposal Volume by Time Window**")

        # Compact horizontal progress breakdown (No giant charts)
        time_windows = [
            ("Night Dumping Peak (10:00 PM – 2:00 AM)", 68, "#d97706"),
            ("Morning Route [Scheduled Tipper] (6:00 AM – 10:00 AM)", 14, "#059669"),
            ("Evening Transit & Markets (2:00 PM – 7:00 PM)", 10, "#64748b"),
            ("Mid-Day Activity (10:00 AM – 2:00 PM)", 8, "#94a3b8"),
        ]

        for label, val, color in time_windows:
            c_lbl, c_bar, c_val = st.columns([2.5, 4, 0.8])
            with c_lbl:
                st.markdown(f"<span style='font-size:0.85rem; font-weight:600; color:#334155;'>{label}</span>", unsafe_allow_html=True)
            with c_bar:
                st.progress(val / 100)
            with c_val:
                st.markdown(f"<span style='font-size:0.85rem; font-weight:700; color:#0f172a;'>{val}%</span>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #fffbeb; border-left: 3px solid #f59e0b; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-top: 14px; font-size: 0.85rem; color: #92400e;">
            <b>Operational Insight:</b> A 14-hour gap exists between night discard and the morning sweep. Transitioning one tipper auto to an 11:00 PM secondary sweep eliminates this backlog.
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("🚨 Dispatch Targeted Night Patrol Manifest", type="primary", use_container_width=True):
            st.success("Patrol Manifest #NP-2026-992 routed to Ward Health Squad 3.")

    # ---------------------------------------------------------
    # TAB 4: INTERVENTIONS (ANTI-RELAPSE PLACEMAKING)
    # ---------------------------------------------------------
    with o_tabs[3]:
        st.subheader("Physical Placemaking & Anti-Relapse Tracking")
        st.caption("Preventing cleaned blackspots from relapsing within 72 hours via physical and community anchoring.")

        i_col1, i_col2 = st.columns([1.1, 1], gap="large")
        
        with i_col1:
            st.markdown("##### The 4-Stage De-Blackspotting Protocol")
            st.markdown("""
            * **Stage 1 (Day 0 - Deep Cleanse):** Mechanical waste clearing, deep wash, and odor neutralizers.
            * **Stage 2 (Day 3 - Placemaking):**
              * Installation of **150 kg cast-in-situ concrete planters** (cannot be moved or stolen).
              * Traditional wall art or geometric patterns painted on adjacent boundary walls.
            * **Stage 3 (Days 7–30 - Merchant Stewardship):**
              * Adjacent tea stalls or bakeries adopt the spot in exchange for clean outdoor seating legitimacy.
            * **Stage 4 (Day 60 - Delisting):** Spot permanently removed from the chronic deficit registry.
            """)

        with i_col2:
            st.markdown("##### Active Spot Status")
            interventions_df = pd.DataFrame([
                {
                    "Spot Location": "Madiwala Junction",
                    "Hardware Deployed": "150 kg concrete planters",
                    "Anchor Partner": "Venkateshwara Tea Stall",
                    "Current Stage": "Day 14 (Clean)"
                },
                {
                    "Spot Location": "Bellandur Canal Bridge",
                    "Hardware Deployed": "Canal barrier netting + twin bins",
                    "Anchor Partner": "Ward Marshals",
                    "Current Stage": "Day 4 (Monitoring)"
                },
                {
                    "Spot Location": "Malleswaram 8th Cross",
                    "Hardware Deployed": "Planters + Community Mural",
                    "Anchor Partner": "Resident Welfare Association",
                    "Current Stage": "Day 60 (Certified Clean)"
                }
            ])
            st.dataframe(interventions_df, use_container_width=True, hide_index=True)
            st.success("🏆 18 chronic blackspots across the sector have achieved Stage 4 de-listing.")
# -------------------------------------------------------------
# 🏛️ COMMISSIONER VIEW
# -------------------------------------------------------------
else:
    c_tabs = st.tabs(["📊 City Intelligence", "🎯 Infrastructure Priorities", "💡 AI Recommendations", "📈 Impact"])

    # 1. City Intelligence
    with c_tabs[0]:
        st.subheader("City-Wide SWM Intelligence")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Wards Monitored", "198", "BBMP Complete")
        m2.metric("Active Chronic Blackspots", "24", "-18 Past Quarter")
        m3.metric("Landfill Diversion Rate", "62.4%", "+8.1% YoY")
        
        st.write("")
        st.markdown("**City Macro Geospatial Distribution**")
        blr_map = folium.Map(location=[12.94, 77.62], zoom_start=11, tiles="OpenStreetMap")
        folium.CircleMarker([12.9226, 77.6174], radius=7, color="#ef4444", fill=True, tooltip="Ward 172 Deficit").add_to(blr_map)
        folium.CircleMarker([12.9260, 77.6762], radius=7, color="#7c3aed", fill=True, tooltip="Ward 150 Buffer Deficit").add_to(blr_map)
        st_folium(blr_map, width=1000, height=350)

    # 2. Infrastructure Priorities (Deterministic Priority Formula)
    with c_tabs[1]:
        st.subheader("Deterministic Infrastructure Priority Engine")
        st.caption("Citizen reports feed evidence into a deterministic formula. Gemini explains the reasoning.")

        loc_choice = st.selectbox(
            "Select Evaluated Hotspot:",
            ["Madiwala Market (Ward 172)", "Bellandur Canal Corridor (Ward 150)"]
        )

        if "Madiwala" in loc_choice:
            density_in, rec_in, pop_in, env_in, gap_in = 45, 26, 42000, False, True
        else:
            density_in, rec_in, pop_in, env_in, gap_in = 48, 28, 55000, True, True

        # Calculate deterministically in Python
        score_res = calculate_infrastructure_priority(density_in, rec_in, pop_in, env_in, gap_in)

        st.markdown(f"""
        <div class="score-breakdown-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#64748b; font-size:0.9rem;">INFRASTRUCTURE PRIORITY INDEX</span>
                <span style="font-size:2rem; font-weight:800; color:#059669;">{score_res['total']} / 100</span>
            </div>
            <hr style="border:0; border-top:1px solid #e2e8f0; margin:12px 0;">
            <div style="font-size:0.88rem; line-height:1.8; color:#334155;">
                • Report Density (Max 30): <b>{score_res['density']}</b><br>
                • Recurrence Frequency (Max 25): <b>{score_res['recurrence']}</b><br>
                • Population Exposure (Max 20): <b>{score_res['exposure']}</b><br>
                • Environmental / Lake Risk (Max 15): <b>{score_res['env_risk']}</b><br>
                • Collection Service Gap (Max 10): <b>{score_res['service_gap']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. AI Recommendations (Gemini explains the deterministic score)
    with c_tabs[2]:
        st.subheader("AI Reasoning & Infrastructure Solution")
        st.caption("Gemini interprets the deterministic priority data to explain why capital should be allocated.")

        if st.button("Synthesize Executive Brief", type="primary"):
            with st.spinner("Gemini explaining the infrastructure priority breakdown..."):
                prompt = f"""
                You are CivicPulse AI explaining a deterministic score to a City Commissioner.
                Location: {loc_choice}
                Calculated Score: {score_res['total']}/100
                Breakdown: Density={score_res['density']}, Recurrence={score_res['recurrence']}, Population Exposure={score_res['exposure']}, Eco Risk={score_res['env_risk']}, Service Gap={score_res['service_gap']}
                
                Provide:
                1. Executive Explanation (why this score was achieved).
                2. Recommended Infrastructure Fix (e.g., Micro-collection point, twin bins, barrier net).
                3. Operational Policy Shift (timing adjustments).
                Keep it concise and professional.
                """
                try:
                    explanation = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt
                    )
                    st.markdown(explanation.text)
                except Exception as e:
                    st.error(f"Error: {e}")

    # 4. Impact
    with c_tabs[3]:
        st.subheader("Projected Environmental & Fiscal Impact")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Estimated Landfill Diversion", "8.4 TPD", "Decentralized sorting")
            st.metric("Citizen Grievance Reduction", "74%", "Within 60 days of hardware install")
        with col_b:
            st.markdown("""
            **BRICS DPG Transferability:**
            - **São Paulo:** Favela alleyway micro-drop points.
            - **Johannesburg:** Taxi rank twin-bin nodes.
            - **Cairo:** Canal-bank protective barriers.
            """)