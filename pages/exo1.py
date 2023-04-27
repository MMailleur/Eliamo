import streamlit as st
import pandas as pd
import numpy as np
import pickle


col1, col2,col3 = st.columns(3)
@st.cache_data

def import_data(filename):
    df = pd.read_csv(filename)
    return df

df_test =import_data("test.csv")

@st.cache_resource

def import_model(filenamemodel):
    loaded_model = pickle.load(open(filenamemodel, 'rb'))
    return loaded_model

model_1 = import_model("model.pickle")

def make_pred():
    row_test = df_test.sample(1)
    arr = row_test.values.reshape(28, 28)
    pred = model_1.predict(row_test.values.reshape(1, 28, 28, 1)).argmax()

    return row_test,arr,pred
good = False
bad = False
row_test,arr,pred= make_pred()
with col1:
    st.title('L image')


    st.image(arr,use_column_width = "always")
    st.header(f'La prédiction est {pred}')
    if st.button('Predict good ?'):
        good = True
    if st.button('Predict bad ?'):
        bad = True

with col3:

    st.title('Pov moi')
    st.markdown("![Alt Text](https://media.tenor.com/3HZaZH1XxRsAAAAC/le-coeur-a-ses-raisons-le-coeur-a-ses-raison.gif)")
