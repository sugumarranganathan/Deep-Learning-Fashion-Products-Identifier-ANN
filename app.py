import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model_cached():
    model_path = os.path.join(os.path.dirname(__file__), "fashion_model.h5")
    return load_model(model_path)

model = load_model_cached()

# Class labels
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title("👗 Fashion Product Identifier")
st.markdown("Upload a clothing image to predict its category.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocessing
        img = Image.open(uploaded_file).convert('L')  # Grayscale
        img = img.resize((28, 28))                    # Resize to 28x28
        
        img_array = np.array(img) / 255.0             # Normalize
        
        # Auto-invert if needed
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array
            
        img_array = img_array.reshape(1, 784)         # Flatten
        
        # Prediction
        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Display
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Processed Image", use_column_width=True)
        with col2:
            st.success(f"🎯 Predicted: **{CLASS_NAMES[pred_idx]}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.subheader("📊 Probability Distribution")
            for i, name in enumerate(CLASS_NAMES):
                prob = predictions[0][i] * 100
                st.progress(prob / 100)
                st.caption(f"{name}: {prob:.1f}%")
                
    except Exception as e:
        st.error(f"❌ Error: {e}")