import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fashion Product Identifier",
    page_icon="👗",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model_cached():
    model_path = os.path.join(os.path.dirname(__file__), "fashion_model.h5")
    return load_model(model_path)

model = load_model_cached()

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# -----------------------------
# Title Section
# -----------------------------
st.title("👗 Fashion Product Identifier using ANN")
st.markdown(
    """
    Upload a fashion product image and the AI model will predict its category.
    """
)

st.divider()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:
    try:
        # Open image
        img = Image.open(uploaded_file).convert('L')

        # Resize image
        img = img.resize((28, 28))

        # Convert to array
        img_array = np.array(img) / 255.0

        # Auto invert if background is white
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array

        # Flatten image
        img_array = img_array.reshape(1, 784)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        pred_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        # Layout
        col1, col2 = st.columns([1, 1])

        # -----------------------------
        # Image Display
        # -----------------------------
        with col1:
            st.subheader("🖼 Uploaded Image")
            st.image(
                img,
                caption="Processed Image",
                use_container_width=True
            )

        # -----------------------------
        # Prediction Display
        # -----------------------------
        with col2:
            st.subheader("🎯 Prediction Result")

            st.success(
                f"Predicted Product: {CLASS_NAMES[pred_idx]}"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("📊 Probability Distribution")

            for i, name in enumerate(CLASS_NAMES):
                prob = predictions[0][i] * 100

                st.write(f"**{name}** — {prob:.2f}%")
                st.progress(float(prob / 100))

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "AI-powered Fashion Product Classification Web App using Artificial Neural Networks (ANN)"
)
