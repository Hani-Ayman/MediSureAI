import streamlit as st
from streamlit_lottie import st_lottie
import json
#set page title
st.set_page_config(page_title="MediSureAI Dashboard", page_icon="💊", layout="wide")

# Load welcome animation (main page)
with open("lottie/lottie_ai.json") as f:
    lottie_welcome = json.load(f)

# Load sidebar logo animation
with open("lottie/lottie_logo.json") as f:
    lottie_logo = json.load(f)

# Sidebar animation at the top-left
with st.sidebar:
    st_lottie(lottie_logo, height=120, key="sidebar-logo")

# Main welcome animation (center)
st_lottie(lottie_welcome, height=150, key="welcome-anim")

st.title("💊 MediSureAI: Online Testing & Monitoring System")

st.markdown("""
Welcome to **MediSureAI**, your intelligent assistant for ensuring the **quality and safety of medical items** used in hospitals and clinics.
""")

st.markdown("---")

### 🚨 Problem Statement and Challenges
st.markdown("""
### 🚨 The Problem Hospitals Face

Hospitals often deal with **large volumes of medical inventory**, including medicines and consumables that:
- May be **expired** or near expiration
- Arrive with **damaged packaging**
- Are **contaminated** or **tampered during transit**
- Lack proper tracking once inside storage

These issues pose serious **health risks to patients** and **liability risks to healthcare providers**. Manual inspection is time-consuming, inconsistent, and prone to error — especially under high-pressure scenarios like emergencies or pandemics.

---

### 🧠 How MediSureAI Helps

**MediSureAI** was designed to solve these problems with a **hands-free, AI-driven solution**:
- 🔍 **Unsupervised Clustering & Anomaly Detection** flags questionable items using K-Means and DBSCAN — no manual labels required.
- 🧪 **Augmentation Playground** helps users visually explore, test, and understand how the AI system responds to common image variations.
- 📊 **Model Quality Check** offers internal evaluation of clustering consistency and classification.
- 📄 **Final Report Page** provides an automated summary of insights, ready for review or compliance documentation.

This system is designed to be **scalable**, **automated**, and **deployable** in real-world hospital settings with minimal setup.

---

👈 **Use the sidebar to explore MediSureAI in action.**
""")
