const fileInput = document.getElementById('fileInput');
const imageContainer = document.getElementById('imageContainer');
const uploadedImage = document.getElementById('uploadedImage');
const resultText = document.getElementById('resultText');

fileInput.addEventListener('change', async function() {
  const file = fileInput.files[0];
  if (file) {
    const imgURL = URL.createObjectURL(file);
    uploadedImage.src = imgURL;
    imageContainer.classList.remove('hidden');

    const model = await tf.loadLayersModel('models/model.json');
    const imageElement = document.createElement('img');
    imageElement.src = imgURL;
    imageElement.width = 224;
    imageElement.height = 224;
    imageElement.onload = async function() {
      const tensor = tf.browser.fromPixels(imageElement).toFloat().expandDims();
      const predictions = await model.predict(tensor).data();
      const classNames = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];
      const result = classNames[predictions.indexOf(Math.max(...predictions))];
      resultText.innerText = `Prediction: ${result}`;
    };
  }
});
