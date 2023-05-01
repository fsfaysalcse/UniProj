from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Initialize the Flask application
app = Flask(__name__)


# Define the predict function
def predict(image):
    # Create an array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize and normalize the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the class and confidence score
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score


# Define the route for the API
@app.route('/predict', methods=['POST'])
def predict_api():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    # Read the image file
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Make the prediction
    class_name, confidence_score = predict(image)

    # Return the prediction result
    return jsonify({'class': class_name, 'confidence': confidence_score})


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
