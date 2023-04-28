import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
import streamlit_nested_layout
from threading import Timer
import time
from utils import yes_gif,sad_gif


if 'good' not in st.session_state:
    st.session_state.good = False
if 'bad' not in st.session_state:
    st.session_state.bad = False
col1, col2,col3 = st.columns([0.5,0.2,1])

def make_pred():
    row_test = df_test.sample(1)
    arr = row_test.values.reshape(28, 28)
    pred = model_1.predict(row_test.values.reshape(1, 28, 28, 1)).argmax()

    return row_test,arr,pred

row_test,arr,pred= make_pred()

with col1:
    st.title('L image')


    st.image(arr,use_column_width = "always")



with col3:
    st.title(f'La prédiction est {pred}')
    col22,col23=st.columns(2)

    with col22 :
        if st.button('Predict juste ?'):
            st.session_state.good = True
        else :
            st.session_state.good = False




    with col23 :
        if  st.button('Predict fausse ?'):
            st.session_state.bad = True
        else :
            st.session_state.bad = False


    if  st.session_state.bad :
        st.markdown(f"![Alt Text]({sad_gif[np.random.randint(0,6)]})")
    if  st.session_state.good :
        st.markdown(f"![Alt Text]({yes_gif[np.random.randint(0,6)]})")
        st.balloons()
