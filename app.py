# app.py
import streamlit as st
from PIL import Image
import numpy as np
import random

# ----------------------
# App title
# ----------------------
st.title("🌿 Leaf Disease Detection (Demo)")
st.write("Upload a leaf image or take a picture with your webcam.")

# ----------------------
# Image input
# ----------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
cam_file = st.camera_input("Or take a picture of the leaf")

input_image = uploaded_file or cam_file

if input_image is not None:
    img = Image.open(input_image)
    st.image(img, caption="Input Image", use_column_width=True)

    # ----------------------
    # Demo prediction
    # ----------------------
    classes = ["Apple Scab", "Healthy", "Leaf Blight"]
    predicted_class = random.choice(classes)
    confidence = random.uniform(80, 99)

    st.markdown(f"### Prediction: **{predicted_class}** ({confidence:.2f}%)")

    # Optional disease info
    if predicted_class == "Apple Scab":
        st.write("Symptoms: Brown or black spots on leaves. Reduce moisture and remove infected leaves.")
    elif predicted_class == "Leaf Blight":
        st.write("Symptoms: Yellowing and browning of leaf edges. Remove infected leaves and apply fungicide.")
    elif predicted_class == "Healthy":
        st.write("Leaf looks healthy! ✅")
else:
    st.info("Please upload an image or take a photo to get a prediction.")
