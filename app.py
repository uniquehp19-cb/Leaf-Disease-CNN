# app.py
import streamlit as st
from PIL import Image
import numpy as np
import random

# ----------------------
# App title
# ----------------------
st.set_page_config(page_title="Leaf Disease Detection Demo", page_icon="🌿", layout="centered")
st.title("🌿 Leaf Disease Detection (Demo)")
st.write("Upload a leaf image or take a picture with your webcam to see a demo prediction.")

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# ----------------------
# Webcam input
# ----------------------
cam_file = st.camera_input("Or take a picture of the leaf")

# ----------------------
# Select which input to use
# ----------------------
input_image = uploaded_file or cam_file

if input_image is not None:
    # Open image
    img = Image.open(input_image)
    st.image(img, caption="Input Image", use_column_width=True)
    
    # ----------------------
    # Placeholder prediction
    # ----------------------
    classes = ["Apple Scab", "Healthy", "Leaf Blight"]
    predicted_class = random.choice(classes)
    confidence = random.uniform(80, 99)  # Random confidence for demo
    
    st.markdown(f"### Prediction: **{predicted_class}** ({confidence:.2f}%)")
    
    # ----------------------
    # Disease info
    # ----------------------
    if predicted_class == "Apple Scab":
        st.info("Symptoms: Brown or black spots on leaves. Reduce moisture and remove infected leaves.")
    elif predicted_class == "Leaf Blight":
        st.warning("Symptoms: Yellowing and browning of leaf edges. Remove infected leaves and apply fungicide.")
    elif predicted_class == "Healthy":
        st.success("Leaf looks healthy! ✅")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Demo version: Predictions are randomly generated. Replace with a trained CNN model for real results.")