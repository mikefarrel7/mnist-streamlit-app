import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

st.title("🧠 MNIST Digit Classifier")
st.write("Gambarlah angka (0–9) dan model akan mencoba mengenalinya.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
    img = ImageOps.invert(img).resize((28, 28)).convert("L")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button("Prediksi"):
        pred = model.predict(img_array)
        st.write(f"Model memprediksi: **{np.argmax(pred)}**")
        st.bar_chart(pred[0])
