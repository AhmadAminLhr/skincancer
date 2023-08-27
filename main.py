import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

set_background('./bgs/Untitled design.png')

# set title
st.title('Skin Cancer Detection')

# set header
st.header('Please upload a sample image for detection')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/Skin_Cancer.h5')

# Initialize class_names as an empty list
class_names = []

# Load class names from updated labels.txt with error handling
try:
    with open('./model/labels.txt', 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_names.append(parts[1])
except FileNotFoundError:
    st.error("Error: 'labels.txt' file not found.")
except Exception as e:
    st.error(f"An error occurred while reading 'labels.txt': {str(e)}")

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
