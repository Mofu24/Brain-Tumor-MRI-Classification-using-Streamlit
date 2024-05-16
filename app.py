import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

@st.cache
def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image
    
model = load_model()

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Guide", "About", "Links"])

# Home Page
if page == "Home":
    st.title("Brain Tumor MRI Classification")
    st.markdown("---")

    # File Uploader
    file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

    # Function to make predictions
    def import_and_predict(image_data, model):
        size = (150, 150)  
        image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS) 
        img = np.asarray(image)
        img = img / 255.0  
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    # Display the results
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded MRI', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = import_and_predict(image, model)
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        result = class_names[np.argmax(prediction)]
        if result == 'No Tumor':
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

# Names Page
elif page == "Guide":
    st.title("Guide")
    st.markdown("---")
    st.write("This page displays the names of the classes that the model can classify:")

    st.write("- Glioma")
    glioma_image = load_image("https://drive.google.com/uc?export=view&id=1_dHlhzdvtZxzPKiby1w9N__R9uPrAXUP")
    st.image(glioma_image, caption="Glioma", use_column_width=True)

    st.write("- Meningioma")
    meningioma_image = load_image("https://drive.google.com/uc?export=view&id=1gCTR9Oe4zuE3SDojoqYPMPwOupfSA9Lf")
    st.image(meningioma_image, caption="Meningioma", use_column_width=True)

    st.write("- No Tumor")
    no_tumor_image = load_image("https://drive.google.com/uc?export=view&id=1JqI8bUEW6P3PyYfGsudr_0oMxekgYLDy")
    st.image(no_tumor_image, caption="No Tumor", use_column_width=True)

    st.write("- Pituitary")
    pituitary_image = load_image("https://drive.google.com/uc?export=view&id=1gLzYhPu_P-ZZybapSBEE_mzTymFCd7FP")
    st.image(pituitary_image, caption="Pituitary", use_column_width=True)
    
# About Page
elif page == "About":
    st.title("About")
    st.markdown("---")
    st.write("This is a simple web application that classifies Brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor.")
    st.write("It uses a deep learning model trained on MRI images to make predictions.")
    st.markdown("---")
    st.header("Group 3 - CPE 019-CPE32S6")
    st.markdown("---")
    st.write("Ejercito, Marlon Jason")
    st.write("Flores, Mico Joshua")
    st.write("Flores, Marc Oliver")
    st.write("Gabiano, Chris Leonard")
    st.write("Gomez, Joram")
    st.markdown("---")

elif page == "Links":
    st.title("Links")
    st.markdown("---")
    st.header("Github Link")
    st.write("[Click Here](https://github.com/qmjae/Brain-Tumor-MRI-Classification-using-Streamlit)")
    st.header("Google Drive Link")
    st.write("[Click Here](https://drive.google.com/drive/folders/1MExGDFt6MVJunB97RloUM7sNb3rudecz?usp=sharing)")
    st.header("Google Colaboratory Link")
    st.write("[Click Here](https://colab.research.google.com/drive/1voRF5tQ49C45BU7mJRV8wBKjji_YANz5?usp=sharing)")
