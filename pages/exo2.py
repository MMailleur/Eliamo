import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from PIL import Image

col1,col2,col3 = st.columns([1,0.4,0.8])

if 'preds' not in st.session_state:
	st.session_state.preds = []

if 'pred_imgs' not in st.session_state:
    st.session_state.pred_imgs = []

if 'affichage' not in st.session_state:
    st.session_state.affichage = False

if 'validation' not in st.session_state:
    st.session_state.validation = []


if 'pred' not in st.session_state:
    st.session_state.pred = 0
    
def reset_canvas():
        return st_canvas(
        height=280,
        width=280,
        background_color = "#000000",
        stroke_color = "#FFFFFF",
        initial_drawing= None
    )

with col1:
    st.title("Canvas")

    img_resized = Image.fromarray(reset_canvas().image_data.astype('uint8')).resize((28, 28))
    # Convert the image to grayscale
    img_gray = img_resized.convert('L')
    # Convertir l'image en array numpy
    img_array = np.array(img_gray)
    # Traiter l'image comme nécessaire (ex: la normaliser)
    processed_img_array = img_array / 255.0
    # Stocker l'image dans une variable
    image = np.expand_dims(processed_img_array, axis=0)

def restart():

    st.session_state.pred_imgs = []
    st.session_state.preds = []
    st.session_state.validation = []
    st.session_state.affichage = False
    st.experimental_rerun()

def pred():
    return model_1.predict(processed_img_array.reshape(1, 28, 28, 1)).argmax()

def append_true() :
    st.session_state.validation.append(1)
    st.session_state.affichage = False
    st.experimental_rerun()

def append_false() :
    st.session_state.validation.append(0)
    st.session_state.affichage = False
    st.experimental_rerun()

if len(st.session_state.preds) < 11:
    with col3:
        st.title("Prediction")
        if st.session_state.affichage == False :
            st.markdown(f"![Alt Text](https://media.tenor.com/PtDbPUn0AhgAAAAM/crystal-ball-fortune-teller.gif)")
            if st.button("Make prediction"):
                st.session_state.pred = pred()
                st.session_state.preds.append(st.session_state.pred)
                st.session_state.pred_imgs.append(processed_img_array)
                st.session_state.affichage = True
                st.experimental_rerun()

        if st.session_state.affichage == True:

            fig_1, ax = plt.subplots()
            ax.imshow(processed_img_array)
            st.pyplot(fig = fig_1)
            st.subheader(f'La prédiction est {st.session_state.pred}')
            if st.button("✅") :
                append_true()

            if st.button("❌") :
                append_false()

else: st.header(f"{sum(st.session_state.validation) * 100/ len(st.session_state.validation)}% de bonnes réponses de l'IA")
if  st.session_state.validation:
    st.text("Bonne predict 0 ou 1 :")
    st.text(st.session_state.validation)

with col1:
    if st.button("Restart"):
        restart()
        # st.session_state.affichage = False
        # st.session_state.preds = []
        # st.session_state.pred_imgs = []
        # st.session_state.validation = []

    if  st.session_state.pred_imgs:
        st.text("Image et valeurs prédite :")
        st.image(st.session_state.pred_imgs, caption=st.session_state.preds)
