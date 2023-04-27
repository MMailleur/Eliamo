import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test

from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Drawable Canvas")

image_canv = st_canvas(
     height=280,
     width=280,
     background_color  ="#000000",
     stroke_color = "#FFFFFF"
)

img_resized = Image.fromarray(image_canv.image_data.astype('uint8')).resize((28, 28))
# Convert the image to grayscale
img_gray = img_resized.convert('L')
# Convertir l'image en array numpy
img_array = np.array(img_gray)
# Traiter l'image comme nécessaire (ex: la normaliser)
processed_img_array = img_array / 255.0
st.image(processed_img_array)
# Stocker l'image dans une variable
image = np.expand_dims(processed_img_array, axis=0)

def pred():
    return model_1.predict(processed_img_array.reshape(1, 28, 28, 1)).argmax()

if st.button("Pred"):
    st.header(f'La prédiction est {pred()}')
