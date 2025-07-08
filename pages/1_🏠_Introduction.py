# pages/1_🏠_Introduction.py
import streamlit as st
import streamlit_lottie as st_lottie
import json

st.set_page_config(page_title="Introduction", page_icon="🏠")

st.title("🏠 Welcome to MediSureAI")

# Load Lottie animation
with open("lottie/lottie_ai.json", "r") as f:
    lottie_json = json.load(f)

st_lottie.st_lottie(lottie_json, speed=1, reverse=False, loop=True, quality="high", height=300)

st.markdown("""
## 🧠 Smarter Medicine Safety Starts Here

**MediSureAI** is your AI-powered ally in ensuring that hospital medicines and consumables remain **safe**, **undamaged**, and **usable**. In critical healthcare environments, even small defects can risk lives — that’s why MediSureAI brings automation, anomaly detection, and visual intelligence to the frontline of quality control.

---

### 🔬 What Can MediSureAI Do?
- 📷 **Image-based Defect Detection**: Identify **damaged**, **contaminated**, or **compromised** medical items using AI.
- 🧪 **Augmentation Playground**: Simulate real-world variations and test model robustness using image augmentation.
- 🔍 **Clustering & Anomaly Detection**: Apply unsupervised learning (K-Means, DBSCAN) to flag suspicious items without prior labels.
- 📊 **Model Quality Check**: Evaluate clustering outputs and segmentation results with clear feedback.
- 📄 **Final Report**: Get a concise, auto-generated summary based on quality check outcomes.

---

### 🛠 Built With:
- **Streamlit**: For an intuitive and interactive UI
- **OpenCV**: For image preprocessing and analysis
- **Scikit-learn & DBSCAN/K-Means**: For unsupervised anomaly detection
- **TensorFlow/Keras**: For CNN-based visual classification
- **Matplotlib/Plotly**: For static result visualizations
- **Deployment**: Streamlit Community Cloud / Localhost

---

### 💡 Why It Matters:
By combining machine learning and real-world simulation, **MediSureAI** empowers healthcare providers to automatically detect potential risks **before** they reach the patient.

👉 Start exploring from the sidebar and see AI in action!
""")
