from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from chatapp import ChatApp as cA  # Importing ChatApp class from chatapp module
import os  # Importing the os module for path operations
from keras.models import load_model #  A function from Keras to load a pre-trained neural network model.
import nltk
import utils as u
import json
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)










# Load the model
model = tf.keras.models.load_model("/Users/krishna/Desktop/gdscweb/gdscbinaryfinal")

# Define the class names
class_names = ['Burn','Cut']

# Function to load and preprocess an image
def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img

# Function to make a prediction and plot the result
def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    return pred_class
# Configure the upload folder
UPLOAD_FOLDER = '/Users/krishna/Desktop/gdscweb/gdscbinaryfinal/UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filename)

        # Make prediction
        result = pred_and_plot(model, filename, class_names)

        # Delete the uploaded file
        #os.remove(filename)

        return render_template('predict.html', prediction=result, uploaded_image=filename)

    return render_template('predict.html')






if __name__ == "__main__":
    app.run(port=5003)
