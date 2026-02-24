import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

st.title("🌿 Leaf Disease Detection (Upload + Live Webcam)")

# Mode selection
mode = st.radio("Select Mode:", ["Upload Images", "Live Webcam Scan"])

# Function to process a leaf image
def process_leaf(image_cv):
    # Resize
    image_cv = cv2.resize(image_cv, (400, 300))

    # Convert to HSV
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2HSV)

    # Green mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Diseased mask
    mask_diseased = cv2.bitwise_not(mask_green)

    # Transparent red overlay
    overlay = image_cv.copy()
    overlay[mask_diseased > 0] = [255, 0, 0]  # red
    alpha = 0.5
    highlighted = cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0)

    # Green/Diseased %
    green_pixels = np.sum(mask_green == 255)
    total_pixels = mask_green.size
    green_ratio = green_pixels / total_pixels
    diseased_ratio = 1 - green_ratio

    # Rule-based disease names
    if green_ratio > 0.5:
        result = "Healthy Leaf"
        disease_name = "No Disease"
    elif green_ratio > 0.3:
        result = "Diseased Leaf"
        disease_name = "Early Blight"
    else:
        result = "Diseased Leaf"
        disease_name = "Late Blight"

    return highlighted, diseased_ratio, result, disease_name

# Function for color-coded severity bar
def severity_bar_html(percent):
    if percent < 0.3:
        color = "green"
    elif percent < 0.6:
        color = "yellow"
    else:
        color = "red"
    html = f"""
    <div style="background-color: lightgray; width: 100%; border-radius:5px; height: 20px;">
        <div style="width: {percent*100:.1f}%; background-color: {color}; height: 100%; border-radius:5px;"></div>
    </div>
    """
    return html

# ------------------ Upload Mode ------------------
if mode == "Upload Images":
    uploaded_files = st.file_uploader(
        "Upload leaf images (multiple allowed)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_cv = np.array(image)
            if image_cv.shape[2] == 4:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

            highlighted, diseased_ratio, result, disease_name = process_leaf(image_cv)

            st.image(image, caption=f"Original: {uploaded_file.name}", use_column_width=True)
            st.image(highlighted, caption=f"Diseased Overlay: {uploaded_file.name}", use_column_width=True)
            st.markdown(severity_bar_html(diseased_ratio), unsafe_allow_html=True)
            st.write(f"**Severity:** {diseased_ratio*100:.2f}%")
            st.write(f"**Result:** {result} ({disease_name})")

            results.append({
                "Image": uploaded_file.name,
                "Green %": f"{(1-diseased_ratio)*100:.2f}%",
                "Diseased %": f"{diseased_ratio*100:.2f}%",
                "Result": result,
                "Disease Name": disease_name
            })

        df = pd.DataFrame(results)
        st.write("### 📊 Summary of Uploaded Leaves")
        st.table(df)
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="leaf_disease_results.csv",
            mime="text/csv"
        )

# ------------------ Live Webcam Mode ------------------
else:
    st.write("🌱 Live Webcam Scan")
    uploaded_image = st.camera_input("Take a photo of the leaf")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_cv = np.array(image)
        if image_cv.shape[2] == 4:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

        highlighted, diseased_ratio, result, disease_name = process_leaf(image_cv)

        st.image(highlighted, caption="Live Leaf Scan with Diseased Overlay", use_column_width=True)
        st.markdown(severity_bar_html(diseased_ratio), unsafe_allow_html=True)
        st.write(f"**Severity:** {diseased_ratio*100:.2f}%")
        st.write(f"**Result:** {result} ({disease_name})")