# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ----------------------
# Load your trained model
# ----------------------
# Make sure 'model.h5' is in your project folder
model = load_model("model.h5")

# ----------------------
# App title
# ----------------------
st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image or take a picture with your webcam.")

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# ----------------------
# Webcam input
# ----------------------
cam_file = st.camera_input("Or take a picture of the leaf")

# Use whichever input is available
input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif cam_file is not None:
    input_image = Image.open(cam_file)

# ----------------------
# Make prediction
# ----------------------
if input_image is not None:
    st.image(input_image, caption="Input Image", use_column_width=True)
    
    # Preprocess image
    img = input_image.resize((224, 224))  # replace 224,224 with your model's input size
    img_array = np.array(img)/255.0       # normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    
    # Define your classes
    classes = ["Apple Scab", "Healthy", "Leaf Blight"]  # replace with your classes
    predicted_class = classes[np.argmax(preds)]
    
    st.markdown(f"### Prediction: **{predicted_class}**")
    
    # Optional: add disease info
    if predicted_class == "Apple Scab":
        st.write("Symptoms: Brown or black spots on leaves. Reduce moisture and remove infected leaves.")
    elif predicted_class == "Leaf Blight":
        st.write("Symptoms: Yellowing and browning of leaf edges. Remove infected leaves and apply fungicide.")
    elif predicted_class == "Healthy":
        st.write("Leaf looks healthy! ✅")
# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ----------------------
# Load your trained model
# ----------------------
# Make sure 'model.h5' is in your project folder
model = load_model("model.h5")

# ----------------------
# App title
# ----------------------
st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image or take a picture with your webcam.")

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# ----------------------
# Webcam input
# ----------------------
cam_file = st.camera_input("Or take a picture of the leaf")

# Use whichever input is available
input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif cam_file is not None:
    input_image = Image.open(cam_file)

# ----------------------
# Make prediction
# ----------------------
if input_image is not None:
    st.image(input_image, caption="Input Image", use_column_width=True)
    
    # Preprocess image
    img = input_image.resize((224, 224))  # replace 224,224 with your model's input size
    img_array = np.array(img)/255.0       # normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    
    # Define your classes
    classes = ["Apple Scab", "Healthy", "Leaf Blight"]  # replace with your classes
    predicted_class = classes[np.argmax(preds)]
    
    st.markdown(f"### Prediction: **{predicted_class}**")
    
    # Optional: add disease info
    if predicted_class == "Apple Scab":
        st.write("Symptoms: Brown or black spots on leaves. Reduce moisture and remove infected leaves.")
    elif predicted_class == "Leaf Blight":
        st.write("Symptoms: Yellowing and browning of leaf edges. Remove infected leaves and apply fungicide.")
    elif predicted_class == "Healthy":
        st.write("Leaf looks healthy! ✅")
