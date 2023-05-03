import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from PIL import Image

col1, col2,col3 = st.columns([1,0.4,0.8])

if 'preds' not in st.session_state:
	st.session_state.preds = []

if 'pred_imgs' not in st.session_state:
    st.session_state.pred_imgs = []

if 'affichage' not in st.session_state:
    st.session_state.affichage = False

if 'validation' not in st.session_state:
    st.session_state.validation = []



def reset_canvas():
        return st_canvas(
        height=280,
        width=280,
        background_color = "#000000",
        stroke_color = "#FFFFFF",
        initial_drawing= None
    )

with col1:
    st.title("Drawable Canvas")

    img_resized = Image.fromarray(reset_canvas().image_data.astype('uint8')).resize((28, 28))
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

if len(st.session_state.preds) < 11:
    with col3:
        if st.button("Pred"):
            prediction = pred()
            st.header(f'La prédiction est {prediction}')
            fig, ax = plt.subplots()
            ax.imshow(processed_img_array)
            st.pyplot(fig = fig)
            st.session_state.preds.append(prediction)
            st.session_state.pred_imgs.append(processed_img_array)
            st.session_state.affichage = True

        if st.session_state.affichage == True:
            if st.button("✅"):
                st.session_state.validation.append(1)
                st.header(f"affichage={st.session_state.affichage}")
                st.session_state.affichage = False
                st.header(f"affichage={st.session_state.affichage}")

            if st.button("❌"):
                st.session_state.validation.append(0)
                st.session_state.affichage = False



else: st.header(f"{sum(st.session_state.validation) * 100/ len(st.session_state.validation)}% de bonnes réponses de l'IA")
st.text(f"{st.session_state.validation}")
with col1:
    if st.button("Restart"):
        st.session_state.affichage = False
        st.session_state.preds = []
        st.session_state.pred_imgs = []
        st.session_state.validation = []


    if st.session_state.affichage:
        st.image(st.session_state.pred_imgs, caption=st.session_state.preds)
