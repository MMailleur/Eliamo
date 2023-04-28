import streamlit as st
import pandas as pd
import pickle
from keras.utils.vis_utils import model_to_dot

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

model_graph = model_to_dot(model_1,
                           show_layer_names=True, show_layer_activations= True)

model_graph = str(model_graph)

st.graphviz_chart(model_graph, use_container_width=True)
