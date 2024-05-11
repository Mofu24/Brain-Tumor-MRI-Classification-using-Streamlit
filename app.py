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


st.markdown("""
<h1 style='text-align: center; color: #f5f5f5;'>Brain Tumor MRI Classification</h1>
""", unsafe_allow_html=True)

# UI design for file uploader and prediction
uploaded_file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Make a prediction
    if st.button('Predict'):
        # Display a loading message while the model is predicting
        with st.spinner('Predicting...'):
            prediction = import_and_predict(image, model)
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
        
        # Display the prediction result
        st.success(f'Prediction: {predicted_class} (Confidence: {confidence:.2f})')

else:
    st.text("Please upload an image file")


