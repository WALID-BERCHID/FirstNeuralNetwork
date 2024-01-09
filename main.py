from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'drawn_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.prediction = softmax(self.output)
        return self.prediction

    def load_weights(self, file_path):
        weights = np.load(file_path)
        self.weights_input_hidden = weights['weights_input_hidden']
        self.bias_hidden = weights['bias_hidden']
        self.weights_hidden_output = weights['weights_hidden_output']
        self.bias_output = weights['bias_output']

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Load the trained model
model = NeuralNetwork(input_size=784, hidden_size=512, output_size=10)
model.load_weights('your_model_weights .npz')  # Replace with the path to your saved weights

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_json()['imageData']

    # Decode base64 and convert to image
    img = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match MNIST image size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, -1)

    # Make prediction
    raw_prediction = model.forward(img_array)
    argmax_prediction = model.predict(img_array)

    # Save the drawn image
    image_name = f"drawn_{argmax_prediction[0]}.png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    img.save(image_path)

    # Show the drawn image using Pillow
    img.show()

    return str(argmax_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
