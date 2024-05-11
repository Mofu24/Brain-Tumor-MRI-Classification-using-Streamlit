st.markdown("""
<style>
h1 {
    border: 2px solid black;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.write("""
# Brain Tumor MRI Classification
""")

file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png"])

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
