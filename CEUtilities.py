from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image
from report_funcs import *

# API Imports
import base64
import requests
import json

# load the saved model
model = tf.keras.models.load_model("model_19_03_MLP", compile=True)
# rewriting with an api to increase the ease of use

report_gen = [  report_cheddar,
                report_emmantal,
                report_leicester,
                ]

def ce_prediction(image, image_np):

    # Preprocessing the image 
    image = image.resize((32, 32)) # Resize the image to the same size as the model's input shape
    image_array = np.array(image)
    image_array = image_array.reshape(1, 32, 32, 3) # Add a batch dimension to the image

    # Make a prediction on the image
    prediction = model.predict(image_array)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # current_report is set to the returned string output
    current_report = report_gen[predicted_class](image_np)
    # Input above is the unadultered image to properly asses the cheese features

    # Confidence rate is returned as a probability score
    #probability_score = model.predict_proba(image_array)[0][predicted_class]

    return current_report

def api_prediction(image, image_np):
    # Preprocessing the image
    image = image.resize((32, 32))  # Resize the image to the same size as the model's input shape
    image_array = np.array(image)
    image_array = image_array.reshape(1, 32, 32, 3)  # Add a batch dimension to the image

    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Set the API endpoint
    url = "https://your-api-url.com/predict"

    # Create the payload for the POST request
    payload = {
        "image": img_str
    }

    # Send the POST request
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the predicted class from the API response
        predicted_class = json.loads(response.text)["predicted_class"]

        # current_report is set to the returned string output
        current_report = report_gen[predicted_class](image_np)
        # Input above is the unadultered image to properly asses the cheese features

        return current_report
    else:
        return "Error: Could not get prediction from the API"
