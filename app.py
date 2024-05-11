import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

model = load_model()

# Streamlit UI design
st.sidebar.header('Group 3 - CPE 019-CPE32S6')
st.sidebar.markdown("Ejercito, Marlon Jason")
st.sidebar.markdown("Flores, Mico Joshua")
st.sidebar.markdown("Flores, Marc Oliver")
st.sidebar.markdown("Gabiano, Chris Leonard")
st.sidebar.markdown("Gomez, Joram")

st.sidebar.header('Github Link')
st.sidebar.markdown("[Click Here](https://github.com/qmjae/Brain-Tumor-MRI-Classification-using-Streamlit)")

st.sidebar.header('Google Drive Link')
st.sidebar.markdown("[Click Here](https://drive.google.com/drive/folders/1MExGDFt6MVJunB97RloUM7sNb3rudecz?usp=sharing)")


st.write("""
# Brain Tumor MRI Classification
""")

file = st.file_uploader("", type=["jpg", "png"], key="fileuploader")

def import_and_predict(image_data, model):
    size = (150, 150)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  # Use PIL.Image.LANCZOS for resizing
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True, output_format='JPEG')
    
    # Add a border to the image
    st.markdown(
        "<style> img { display: block; margin-left: auto; margin-right: auto; border: 2px solid #ccc; border-radius: 8px; } </style>",
        unsafe_allow_html=True
    )
    
    prediction = import_and_predict(image, model)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)

