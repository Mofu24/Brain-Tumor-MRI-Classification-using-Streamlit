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

<html>
<head>
  <title>Brain Tumor MRI Classification</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="container">
    <h1>Brain Tumor MRI Classification</h1>
    <p>Please upload an image file</p>
    <input type="file" id="fileInput" accept=".jpg,.png">
    <div id="imageContainer" class="hidden">
      <img id="uploadedImage" src="#" alt="Uploaded Image" />
      <p id="resultText"></p>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
  <script src="script.js"></script>
</body>
</html>

st.write("""
# Brain Tumor MRI Classification
""")

file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

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
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
