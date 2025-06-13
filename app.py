import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Tablet Defect Detection Demo")

st.write("TensorFlow version:", tf.__version__)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = tf.io.decode_image(uploaded_file.read(), channels=3)
    image = tf.image.resize(image, [224, 224])
    st.image(image.numpy().astype("uint8"), caption="Uploaded Image")
