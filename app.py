import streamlit as st
import joblib
import tempfile
from glcm_features import extract_glcm_features
from PIL import Image

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("🧵 Texture Classifier using GLCM")

# File uploader
uploaded_file = st.file_uploader("Upload a texture image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save image temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Feature extraction and prediction
    features = extract_glcm_features(tmp_path)
    prediction = model.predict([features])[0]

    # Output results
    st.subheader("🔎 Predicted Class:")
    st.success(prediction)

    st.subheader("📊 GLCM Features:")
    feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    for name, val in zip(feature_names, features):
        st.write(f"{name}: {val:.4f}")
