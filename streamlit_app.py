import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Eliamo",
    page_icon="👋",
)

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



st.title("suuu")
