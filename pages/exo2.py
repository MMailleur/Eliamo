import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from PIL import Image

col1, col2,col3 = st.columns(3)

if 'preds' not in st.session_state:
	st.session_state.preds = []

if 'pred_imgs' not in st.session_state:
    st.session_state.pred_imgs = []

if 'affichage' not in st.session_state:
    st.session_state.affichage = False

if 'validation' not in st.session_state:
    st.session_state.validation = []

if 'bgcolor' not in st.session_state:
    st.session_state.bgcolor = "#000000"

with col1:
    st.title("Drawable Canvas")

    image_canv = st_canvas(
        height=280,
        width=280,
        background_color = st.session_state.bgcolor,
        stroke_color = "#FFFFFF",
        initial_drawing= None
    )


    img_resized = Image.fromarray(image_canv.image_data.astype('uint8')).resize((28, 28))
    # Convert the image to grayscale
    img_gray = img_resized.convert('L')
    # Convertir l'image en array numpy
    img_array = np.array(img_gray)
    # Traiter l'image comme nécessaire (ex: la normaliser)
    processed_img_array = img_array / 255.0
    # Stocker l'image dans une variable
    image = np.expand_dims(processed_img_array, axis=0)


def pred():
    return model_1.predict(processed_img_array.reshape(1, 28, 28, 1)).argmax()

if len(st.session_state.preds) != 10:
    with col3:
        if st.button("Pred"):
            st.session_state.bgcolor = "#000000"
            prediction = pred()
            st.header(f'La prédiction est {prediction}')
            fig, ax = plt.subplots()
            ax.imshow(processed_img_array)
            st.pyplot(fig = fig)
            st.session_state.preds.append(prediction)
            st.session_state.pred_imgs.append(processed_img_array)
            st.session_state.affichage = True
            if st.button("✅"):
                st.session_state.validation.append(True)
                st.session_state.bgcolor = "#000001"

            if st.button("❌"):
                st.session_state.validation.append(False)
                st.session_state.bgcolor = "#000001"



with col1:
    if st.button("Restart"):
        st.session_state.affichage = False
        st.session_state.preds = []
        st.session_state.pred_imgs = []
        st.session_state.validation = []


    if st.session_state.affichage:
        st.image(st.session_state.pred_imgs, caption=st.session_state.preds)

if len(st.session_state.validation) == 10:
    st.header(f"{sum(st.session_state.validation) * 10}% de bonnes réponses de l'IA")
