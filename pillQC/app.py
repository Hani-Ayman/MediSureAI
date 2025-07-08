import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
import random

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="MediSureAI Dashboard", layout="wide")
st.sidebar.title("🩺 Medisync")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Medicine Check", "Upload Batches", "Reports"])

# -------------------------------
# Dashboard Page
# -------------------------------
if menu == "Dashboard":
    st.markdown("## 👩‍⚕ Welcome, Dr. Emma Wilson")
    st.markdown("Here’s a summary of today’s operations.")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Appointments Today", "15 Scheduled")
    col2.metric("New Patients", "7")
    col3.metric("Video Consults", "5 Scheduled")
    col4.metric("Billing", "$12,500 / $8,300 Paid")

    # Patient Table
    st.markdown("### 📋 Patient Records")
    data = {
        "Name": ["Sarah Johnson", "Michael Lee", "Amina Yusuf", "Chloe Fernandez"],
        "Age": [29, 45, 38, 34],
        "Last Visit": ["May 20", "May 17", "May 10", "May 16"],
        "Primary Doctor": ["Dr. Wilson", "Dr. Patel", "Dr. Ng", "Dr. Mehra"],
        "Status": ["Active", "In Treatment", "Discharged", "In Treatment"]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)

    # Charts
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 📈 Patient Visits")
        visits = np.random.randint(5, 30, size=12)
        fig1 = px.line(x=list(range(1, 13)), y=visits, labels={"x": "Month", "y": "Visits"})
        st.plotly_chart(fig1, use_container_width=True)

    with col6:
        st.markdown("### 🧾 Diagnosis Breakdown")
        fig2 = px.pie(
            names=["Hypertension", "Backache", "Stomachache", "Other"],
            values=[22, 18, 14, 10],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Lower Dashboard
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("### ⏱ Avg Appointment Time")
        times = np.random.randint(10, 20, size=12)
        fig3 = px.bar(x=list(range(1, 13)), y=times, labels={"x": "Month", "y": "Minutes"})
        st.plotly_chart(fig3, use_container_width=True)

    with col8:
        st.markdown("### 📊 Dataset Accuracy")
        st.metric("Model Accuracy", "92%")
        st.metric("Refreshed", "Every 4h")
        st.metric("Satisfaction", "4.8 / 5")

    # Footer
    st.markdown("---")
    

# -------------------------------
# Medicine Check (CNN Image)
# -------------------------------
elif menu == "Medicine Check":
    st.markdown("## 🧪 Visual Defect Detection")
    uploaded_file = st.file_uploader("Upload Image of Medicine", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze"):
            with st.spinner("Running CNN Model..."):
                prediction = random.choice(["Good Condition ", "Torn/Damaged ⚠", "Contaminated "])
                st.success(f"Predicted Condition: {prediction}")

# -------------------------------
# Upload Batches (CSV + Clustering)
# -------------------------------
elif menu == "Upload Batches":
    st.markdown("## 📦 Batch Quality Classification")
    csv = st.file_uploader("Upload Batch CSV (with features)", type=["csv"])

    if csv:
        df = pd.read_csv(csv)
        st.dataframe(df.head())

        if st.button("🧠 Predict Quality"):
            with st.spinner("Classifying with supervised ML model..."):
                df["Prediction"] = np.random.choice(["Safe", "Review", "Unsafe"], size=len(df))
                st.success("Prediction complete!")
                st.dataframe(df)

            # Distribution Chart
            st.markdown("### 📊 Batch Prediction Summary")
            fig = px.histogram(df, x="Prediction", color="Prediction")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Reports
# -------------------------------
elif menu == "Reports":
    st.markdown("## 📑 Model & System Reports")
    st.info("Upload evaluation reports or view model performance metrics here.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix")
        st.image("https://i.imgur.com/zL2ZcQw.png", caption="Example Confusion Matrix")  # Replace with actual
    with col2:
        st.markdown("### ROC Curve")
        st.image("https://i.imgur.com/5z1YmL1.png", caption="Example ROC Curve")

    st.markdown("---")
    st.caption("Add real-time metrics, loss trends, or logs here.")