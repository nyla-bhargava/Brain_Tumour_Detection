import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_cnn():
    model = load_model("model.h5")   # your file
    return model

model = load_cnn()

#Preprocess function
IMG_SIZE = (224, 224)   

def preprocess(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    if arr.ndim == 2:              
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]   

#Streamlit UI
st.title("CNN Demo with model.h5")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        x = preprocess(img)
        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        st.write(f"Predicted class: {CLASS_NAMES[idx]}")
        st.write(f"Raw prediction: {preds}")
