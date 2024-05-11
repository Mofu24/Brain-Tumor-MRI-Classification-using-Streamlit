import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
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

st.markdown("""
<div style='border: 2px solid #f63366; background-color: #f9f9f9; padding: 10px;'>
<h1>Brain Tumor MRI Classification</h1>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Choose File", type=["jpg", "png"])


# Function to import and predict
def import_and_predict(image_data, model):
    size = (150, 150)  
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  
    img = np.asarray(image)
    img = img / 255.0  
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Display prediction
if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    result = "Prediction: " + class_names[np.argmax(prediction)]
    st.success(result)
