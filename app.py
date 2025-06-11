import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tablet_defect_model.h5")

model = load_model()

# App title and description
st.set_page_config(page_title="Tablet Defect Detection", page_icon="💊")
st.title("💊 Tablet Defect Detection App")
st.write("Upload an image of a tablet to predict whether it's **Normal** or **Defective** using a trained AI model.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a tablet image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Tablet Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)

    # Predict using the model
    prediction = model.predict(img_array)[0][0]
    result = "🟢 **Normal Tablet**" if prediction < 0.5 else "🔴 **Defective Tablet**"
    confidence = 100 * prediction if prediction > 0.5 else 100 * (1 - prediction)

    # Show the prediction
    st.markdown(f"### Prediction: {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")
