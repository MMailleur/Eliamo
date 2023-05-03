import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_app import model_1, df_test
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from PIL import Image


import tensorflow as tf

st. set_page_config(layout="wide")

def reset_canvas():
        return st_canvas(
        height=280,
        width=280,
        background_color = "#000000",
        stroke_color = "#FFFFFF",
        initial_drawing= None
    )


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



if st.button("Show layers:"):
    prediction = pred()
    st.header(f'La prédiction est {prediction}')
    successive_outputs = [layer.output for layer in model_1.layers[1:]]
    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model_1.input, outputs = successive_outputs)

    # Let's run input image through our vislauization network
    # to obtain all intermediate representations for the image.
    successive_feature_maps = visualization_model.predict(processed_img_array.reshape(1, 28, 28, 1))
    # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model_1.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)

        if len(feature_map.shape) == 4:

            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers

            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
            # Display the grid
                scale = 20. / n_features
            fig = plt.figure( figsize=(scale * n_features, scale) )
            plt.title ( layer_name )
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' )
            st.pyplot(fig)
