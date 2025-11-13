import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import os

# ---------------------------------------------------------
# 🌎 PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="♻️ SmartSort - Smart Waste Sorter",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 🎨 CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    font-family: 'Arial', sans-serif;
    color: #1E90FF;
}
div[data-baseweb="radio"] label {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px 20px;
    margin: 5px;
    border: 2px solid #1E90FF;
    display: inline-block;
    cursor: pointer;
    transition: all 0.2s ease;
}
div[data-baseweb="radio"] input:checked + label {
    background-color: #1E90FF;
    color: white;
    font-weight: bold;
}
.prediction-card {
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
    margin-top: 15px;
    color: white;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 LOAD MODEL (from ZIP if needed)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    zip_path = "ecobot_updated.zip"
    model_path = "ecobot_updated.keras"

    # ✅ Step 1: Extract if not already extracted
    if not os.path.exists(model_path):
        if os.path.exists(zip_path):
            with st.spinner("📦 Extracting model files... please wait..."):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(".")
            st.success("✅ Model extracted successfully!")
        else:
            st.error("❌ Model file not found. Please ensure 'ecobot_updated.zip' is in your project folder.")
            st.stop()

    # ✅ Step 2: Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


# Load model once (cached)
model = load_model()

# ---------------------------------------------------------
# 🏷️ CLASS DEFINITIONS
# ---------------------------------------------------------
CLASSES = ['Dry', 'E-waste', 'Manual', 'Wet']
COLOR_MAP = {
    'Dry': '#1E90FF',     # Blue
    'Wet': '#2E8B57',     # Green
    'E-waste': '#FFA500', # Orange
    'Manual': '#A9A9A9'   # Grey
}

# ---------------------------------------------------------
# 🏷️ TITLE & DESCRIPTION
# ---------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>♻️ SmartSort - Smart Waste Sorter</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Real-time waste classification into <b>Dry</b>, <b>Wet</b>, <b>E-waste</b>, or <b>Manual</b> categories.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 📸 IMAGE INPUT OPTIONS
# ---------------------------------------------------------
st.markdown("### Choose how to provide a waste image:")
choice = st.radio("Select an option:", ["Upload Image", "Capture Photo"], index=0, horizontal=True)

uploaded_file = None
if choice == "Capture Photo":
    uploaded_file = st.camera_input(
        "Take Photo",
        key="camera",
        help="Use rear camera on mobile for better results."
    )
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------------
# 🔍 PREDICTION & OUTPUT
# ---------------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    image.thumbnail((400, 400))
    st.image(image, caption="Selected Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    predicted_class = CLASSES[class_index]
    confidence = float(np.max(preds) * 100)

    # Display prediction card
    st.markdown(
        f"<div class='prediction-card' style='background-color:{COLOR_MAP[predicted_class]};'>Prediction: {predicted_class}</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    # Disposal guidance
    disposal_tips = {
        "Dry": "♻️ Dispose in the **blue bin** (paper, cardboard, plastics, metals).",
        "Wet": "🪴 Dispose in the **green bin** (food scraps, leaves, organic waste).",
        "E-waste": "⚡ Take to an **authorized e-waste collection center**.",
        "Manual": "🧤 Please inspect manually — unclear or mixed waste category."
    }
    st.info(disposal_tips[predicted_class])

# ---------------------------------------------------------
# 📘 SIDEBAR INFO
# ---------------------------------------------------------
st.sidebar.title("ℹ️ About SmartSort")
st.sidebar.write("""
**SmartSort** is a real-time waste classification system powered by **MobileNetV2** and **Streamlit**.  
It helps promote sustainable waste management by identifying waste categories instantly.

### Features:
- Real-time image-based classification  
- Works on mobile (camera input)  
- Provides disposal guidance  

### Waste Categories:
1. Dry  
2. Wet  
3. E-waste  
4. Manual  
""")

# ---------------------------------------------------------
# 🪪 FOOTER
# ---------------------------------------------------------
st.markdown("<div class='footer'>Made with ❤️ by SmartSort | Smart Waste Classification © 2025</div>", unsafe_allow_html=True)
