import streamlit as st
import numpy as np
from keras.models import load_model
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

    # Load legacy .h5 model safely
    model = load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    return model

# Load model
model = load_model_cached()

# -----------------------------------
# Fashion Product Categories
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
Upload a fashion product image and the AI model will predict the clothing category with confidence score.
""")

st.divider()

# -----------------------------------
# Upload Section
# -----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Clothing Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction Logic
# -----------------------------------
if uploaded_file is not None:

    try:

        # -----------------------------
        # Open and preprocess image
        # -----------------------------
        img = Image.open(uploaded_file).convert("L")

        # Resize image to model input size
        img = img.resize((28, 28))

        # Convert to array
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Auto invert image if needed
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array

        # Flatten image
        img_array = img_array.reshape(1, 784)

        # -----------------------------
        # Make Prediction
        # -----------------------------
        predictions = model.predict(
            img_array,
            verbose=0
        )

        # Get highest probability index
        pred_idx = np.argmax(predictions[0])

        # Confidence score
        confidence = float(
            np.max(predictions[0]) * 100
        )

        # Predicted label
        predicted_label = CLASS_NAMES[pred_idx]

        # -----------------------------------
        # Layout Columns
        # -----------------------------------
        col1, col2 = st.columns([1, 1])

        # -----------------------------------
        # Display Uploaded Image
        # -----------------------------------
        with col1:

            st.subheader("🖼 Uploaded Image")

            st.image(
                img,
                caption="Processed Image",
                use_container_width=True
            )

        # -----------------------------------
        # Display Prediction Result
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
            f"❌ Error: {e}"
        )

# -----------------------------------
# Footer
# -----------------------------------
st.divider()

st.caption(
    "AI-powered Fashion Product Classification Web Application using Artificial Neural Networks (ANN)"
)
