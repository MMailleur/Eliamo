import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
from exo1 import make_pred
from streamlit_drawable_canvas import st_canvas

st.title("Drawable Canvas")

st.sidebar.header("Configuration")

# Specify brush parameters and drawing mode
b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

image_data = st_canvas(
    b_width, height=150, drawing_mode=drawing_mode, key="canvas"
)
