import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Fashion Product Identifier",
    page_icon="👗",
    layout="wide"
)

# -----------------------------------
# Load Trained ANN Model
# -----------------------------------
@st.cache_resource
def load_model_cached():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "fashion_model.h5"
    )

    # IMPORTANT FIX
    # compile=False avoids compatibility issues
    # with older .h5 models
    model = load_model(
        model_path,
        compile=False
    )

    return model

model = load_model_cached()

# -----------------------------------
# Fashion Categories
# -----------------------------------
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# -----------------------------------
# App Header
# -----------------------------------
st.title("👗 Fashion Product Identifier using ANN")

st.markdown("""
Upload a fashion product image and the AI model
will predict its clothing category with confidence score.
""")

st.divider()

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Clothing Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction Process
# -----------------------------------
if uploaded_file is not None:

    try:

        # -----------------------------
        # Image Processing
        # -----------------------------
        img = Image.open(uploaded_file).convert("L")

        # Resize image
        img = img.resize((28, 28))

        # Convert image to numpy array
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Auto invert for white backgrounds
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array

        # Flatten image
        img_array = img_array.reshape(1, 784)

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(
            img_array,
            verbose=0
        )

        pred_idx = np.argmax(predictions[0])

        confidence = float(
            np.max(predictions[0]) * 100
        )

        predicted_label = CLASS_NAMES[pred_idx]

        # -----------------------------------
        # Layout
        # -----------------------------------
        col1, col2 = st.columns([1, 1])

        # -----------------------------------
        # Uploaded Image
        # -----------------------------------
        with col1:

            st.subheader("🖼 Uploaded Image")

            st.image(
                img,
                caption="Processed Image",
                use_container_width=True
            )

        # -----------------------------------
        # Prediction Results
        # -----------------------------------
        with col2:

            st.subheader("🎯 Prediction Result")

            st.success(
                f"Predicted Product: {predicted_label}"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("📊 Probability Distribution")

            # Probability bars
            for i, label in enumerate(CLASS_NAMES):

                prob = float(
                    predictions[0][i] * 100
                )

                st.write(
                    f"**{label}** — {prob:.2f}%"
                )

                st.progress(prob / 100)

    except Exception as e:

        st.error(
            f"❌ Error loading model or predicting image: {e}"
        )

# -----------------------------------
# Footer
# -----------------------------------
st.divider()

st.caption(
    "AI-powered Fashion Product Classification Web Application using Artificial Neural Networks (ANN)"
)
