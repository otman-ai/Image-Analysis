#import all the libraries
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
from io import BytesIO
import base64

st.set_page_config(page_title="Image Editing",layout="wide")

def denoise_image(noisy,h=10,hColor=10,templateWindowSize=7,searchWindowSize=21):
    cleaned = cv2.fastNlMeansDenoisingColored(noisy, None,h, 
                                    hColor,#The strength of the filter
                                    templateWindowSize, #the size of the search window
                                    searchWindowSize#  the size of the template window 
)
    return cleaned

def color_quantisizing(k,img):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 7, 1.0)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    # implement kmeans algorithm with the image
    # Extarct label(1,2...) and center :which is the correspond coordinate of a certains pixel
    _, label, center = cv2.kmeans(Z, k, None, criteria, 7,
    cv2.KMEANS_RANDOM_CENTERS)
    # transform the center to integer 
    center = np.uint8(center)
    # dubplicate the center pixel on the image bised on it label
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def blur(img,window_size):
    av3 = cv2.blur(img,(window_size,window_size))
    return av3

def median(img,ksize):
    return cv2.medianBlur(img,ksize)
st.title("Edite your images in few clicks")
st.write(
    """
    :dog: Try uploading an image to edit and filter it on your own way. 
    Full quality images can be downloaded from the sidebar. 
    This code is open source and available 
    [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. 
    """
)
st.sidebar.write("## Upload and download :gear:")
file_uploader =  st.sidebar.file_uploader("Upload you image here:",type=["png","jpg","jpeg"])
col1,col2, col3,col4= st.columns((1,1,1,1))

if file_uploader is not None:
     image = Image.open(file_uploader)

     with open("files/image.jpg",mode = "wb") as f: 
         f.write(file_uploader.getbuffer())  
     image = cv2.imread("files/image.jpg")
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     with col4:
         if st.checkbox("Remove noise"):
                h = st.slider("h:",1,100,20)
                hColor = st.slider("templateWindowSize:",1,100,10)
                templateWindowSize = st.slider("templateWindowSize:",1,100,7)
                searchWindowSize = st.slider("searchWindowSize:",1,100,21)
                image = denoise_image(image,h,hColor,templateWindowSize,searchWindowSize)
     with col2:
        if st.checkbox("Blur"):
            window_size = st.slider("Blur window size:",1,100,20)
            image = blur(image,window_size)

     with col1:
         if st.checkbox("Color Quantisizing"):
            k = st.slider("Color quantisizing key:",1,100,3)
            image = color_quantisizing(k,image)

     with col3:
         if st.checkbox("Median"):
            ksize = st.slider("Median keys size:",1,99,3,2)
            image = median(image,ksize)
     st.sidebar.image(image)
