import tensorflow as tf
import tensorflow_hub as tf_hub
import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Style Transfer",layout="wide")


def load_image(image_path, image_size=(512, 256)):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

st.title("Style Transfer on your images")
st.write(
    """
    :cat: Try uploading an image to stylize and edit. 
    Full quality images can be downloaded from the sidebar. 
    This code is open source and available 
    [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. 
    """
)
st.write("Select one of the style below.")
st.sidebar.write("## Upload and download :gear:")
file_uploader =  st.sidebar.file_uploader("Upload you image here:",type=["png","jpg","jpeg"])
style_images = os.listdir("styled_images/")
Image.open(file_uploader)
if file_uploader is not None:
    st.sidebar.image(file_uploader)
    image = Image.open(file_uploader)
    with open("files/original.jpg",mode = "wb") as f: 
        f.write(file_uploader.getbuffer()) 
    original_image = load_image("files/original.jpg")
    col1 = st.columns((1,1,1,1,1))
    with col1[0]:
        imag0 = Image.open("styled_images/"+style_images[0])
        st.image(imag0,width=100)
        st.write(style_images[0].split(".")[0])

    with col1[1]:
        imag1 = Image.open("styled_images/"+style_images[1])
        x = st.image(imag1,width=100)
        st.write(style_images[1].split(".")[0])

    with col1[2]:
        imag1 = Image.open("styled_images/"+style_images[2])
        x = st.image(imag1,width=100)
        st.write(style_images[2].split(".")[0])

    with col1[3]:
        imag1 = Image.open("styled_images/"+style_images[3])
        x = st.image(imag1,width=100)
        st.write(style_images[3].split(".")[0])

    with col1[4]:
        imag1 = Image.open("styled_images/"+style_images[4])
        x = st.image(imag1,width=100)
        st.write(style_images[4].split(".")[0])

    style_image_item = st.selectbox("Select the style image correspond to it name above",
                               [i for i in style_images ])
    style_image = load_image("styled_images/"+style_image_item)
    style_image = tf.nn.avg_pool(style_image, 
                                        ksize=[3,3], 
                                        strides=[1,1], 
                                        padding='VALID')
    stylize_model = tf_hub.load('tf_model')
    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
    stylized_photo = results[0]
    st.image(np.array(stylized_photo))