import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('mnist_model.keras')

st.title('MNIST Digit Classifier')

uploaded_file = st.file_uploader("Upload image")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28)

    pred = model.predict(image)
    st.write("Prediction:", np.argmax(pred))