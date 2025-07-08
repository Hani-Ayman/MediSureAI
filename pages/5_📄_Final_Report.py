import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import os
import json

# --- Function to Load Lottie Animations ---
@st.cache_resource
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# --- Load Required Animations ---
lottie_report = load_lottiefile("lottie/lottie_report.json")
lottie_success = load_lottiefile("lottie/lottie_suc.json")

# --- Constants ---
LOG_FILE = "medicine_predictions.csv"  # Fixed filename to match prediction logging

# --- REPORT SECTION ---
st_lottie(lottie_report, speed=1, height=160, key="report-anim", loop=True)
st.markdown("<div class='title section-anim'>Prediction Report</div>", unsafe_allow_html=True)
st.markdown('<div class="subheader">See all your predictions and download the CSV report.</div>', unsafe_allow_html=True)

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns([1, 8])
    with col1:
        st_lottie(lottie_success, height=60, key="report-download", loop=True)
    with col2:
        st.download_button(
            label="📥 Download CSV Report",
            data=df.to_csv(index=False),
            file_name="medicine_quality_report.csv"
        )
else:
    st.warning("No prediction report found.")
