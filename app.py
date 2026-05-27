import streamlit as st
import numpy as np

from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from PIL import Image, ImageOps, ImageFilter

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Fashion Product Identifier",
    page_icon="👗",
    layout="wide"
)

# -----------------------------------
# Load Improved ANN Model
# -----------------------------------
@st.cache_resource
def load_model_cached():

    # Improved ANN Architecture
    model = Sequential([

        Dense(
            512,
            activation='relu',
            input_shape=(784,)
        ),

        Dropout(0.3),

        BatchNormalization(),

        Dense(
            256,
            activation='relu'
        ),

        Dropout(0.3),

        BatchNormalization(),

        Dense(
            128,
            activation='relu'
        ),

        Dropout(0.3),

        BatchNormalization(),

        Dense(
            64,
            activation='relu'
        ),

        Dense(
            10,
            activation='softmax'
        )

    ])

    # Build Model
    model.build((None, 784))

    # Load Trained Weights
    model.load_weights(
        "fashion_model_fixed.h5"
    )

    return model

# -----------------------------------
# Initialize Model
# -----------------------------------
model = load_model_cached()

# -----------------------------------
# Fashion Labels
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
# Header
# -----------------------------------
st.title(
    "👗 Fashion Product Identifier using ANN"
)

st.markdown("""
Upload a fashion product image and the AI model
will predict the clothing category with confidence score.
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
# Prediction Section
# -----------------------------------
if uploaded_file is not None:

    try:

        # -----------------------------------
        # Open Image
        # -----------------------------------
        img = Image.open(
            uploaded_file
        )

        # Convert to grayscale
        img = img.convert("L")

        # -----------------------------------
        # Better Image Processing
        # -----------------------------------

        # Resize image
        img = img.resize((28, 28))

        # Sharpen image slightly
        img = img.filter(
            ImageFilter.SHARPEN
        )

        # Auto contrast
        img = ImageOps.autocontrast(img)

        # Convert to numpy array
        img_array = np.array(img)

        # -----------------------------------
        # Normalize Images
        # -----------------------------------
        img_array = img_array.astype(
            "float32"
        ) / 255.0

        # -----------------------------------
        # Background Removal Logic
        # -----------------------------------
        # Fashion-MNIST uses dark background
        # so invert if background is white
        if np.mean(img_array) > 0.5:

            img_array = 1.0 - img_array

        # -----------------------------------
        # Flatten Image
        # -----------------------------------
        img_array = img_array.reshape(
            1,
            784
        )

        # -----------------------------------
        # Predict
        # -----------------------------------
        predictions = model.predict(
            img_array,
            verbose=0
        )

        pred_idx = np.argmax(
            predictions[0]
        )

        confidence = float(
            np.max(predictions[0]) * 100
        )

        predicted_label = CLASS_NAMES[
            pred_idx
        ]

        # -----------------------------------
        # Layout
        # -----------------------------------
        col1, col2 = st.columns([1, 1])

        # -----------------------------------
        # Image Display
        # -----------------------------------
        with col1:

            st.subheader(
                "🖼 Uploaded Image"
            )

            st.image(
                img,
                caption="Processed Image",
                use_column_width=True
            )

        # -----------------------------------
        # Prediction Results
        # -----------------------------------
        with col2:

            st.subheader(
                "🎯 Prediction Result"
            )

            st.success(
                f"Predicted Product: {predicted_label}"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader(
                "📊 Probability Distribution"
            )

            # Probability Bars
            for i, label in enumerate(CLASS_NAMES):

                prob = float(
                    predictions[0][i] * 100
                )

                st.write(
                    f"**{label}** — {prob:.2f}%"
                )

                st.progress(
                    prob / 100
                )

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
