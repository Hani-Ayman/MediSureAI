import streamlit as st
from streamlit_lottie import st_lottie
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
from io import BytesIO
import base64

# Constants
LOG_FILE = "medicine_predictions.csv"
class_labels = ["chip", "dirt", "normal"]

# Load animations
def load_lottiefile(path: str):
    with open(path, "r") as f:
        return json.load(f)

lottie_predict = load_lottiefile("lottie/lottie_quality.json")
lottie_success = load_lottiefile("lottie/lottie_suc.json")

# Page title & animation
st_lottie(lottie_predict, speed=1, height=180, key="predict-anim", loop=True)
st.markdown("<div class='title section-anim'>Medicine Image Quality Check</div>", unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a medicine image to analyze its quality using AI.</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("DENSE.h5")

model = load_cnn_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload Medicine Image", type=["jpg", "jpeg", "png"])

# Predict logic
if uploaded_file:
    # Open original image and prepare for display and prediction
    original_img = Image.open(uploaded_file).convert("RGB")

    # Center and display image using base64 HTML
    buffered = BytesIO()
    original_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 10px;margin-bottom:50px'>
            <img src="data:image/png;base64,{img_base64}" width="300" style="border-radius: 10px;" />
            <div style='font-size: 16px; color: gray; margin-top: 5px;'>Uploaded Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Resize for model prediction only
    img = original_img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Analyze Image", type="primary",use_container_width=True):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            raw_result = class_labels[predicted_class]

            # Map output
            if raw_result == "chip":
                result = "Damaged"
            elif raw_result == "dirt":
                result = "Dirty"
            else:
                result = "Normal"

        # Display result
        with st.container():
            st_lottie(
                lottie_success,
                height=250,
                key="result-anim",
                speed=1
            )
            st.markdown(
                f"""
                <style>
                .result-card {{
                    background-color: #f0f8ff;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-top: 20px;
                }}
                .result {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #2e7d32;
                }}
                .confidence {{
                    font-size: 24px;
                    color: #555;
                }}
                </style>
                <div class='result-card'>
                    <span class='result'>🩺 Prediction: {result}</span><br>
                    <span class='confidence'>Confidence: {confidence * 100:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Show toast
        st.toast(
            f"Prediction: {result} ({confidence * 100:.2f}%)",
            icon="🤖" if result == "Not Damaged" and confidence > 0.8 else "⚠️"
        )

        # Log result
        new_entry = pd.DataFrame([[uploaded_file.name, raw_result, result, f"{confidence * 100:.2f}%"]],
                                 columns=["Image", "Original Label", "Binary Prediction", "Confidence"])
        if os.path.exists(LOG_FILE):
            old_data = pd.read_csv(LOG_FILE)
            updated_data = pd.concat([old_data, new_entry], ignore_index=True)
        else:
            updated_data = new_entry
        updated_data.to_csv(LOG_FILE, index=False)
