import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

# Load the model
model = load_model()

# Custom CSS for changing background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.write("""
# Brain Tumor MRI Classification
""")

# File uploader for the MRI image
file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (150, 150)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  # Use PIL.Image.LANCZOS for resizing
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Display the uploaded image and make predictions
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
