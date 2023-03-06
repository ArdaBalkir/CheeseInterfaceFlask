from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from CEUtilities import *
from datetime import datetime
import base64

app = Flask(__name__)

@app.route('/')
def home():

    # Log will be passed on regardless if a report is generated
    log = report_history
    log = log[-5:]
    log.reverse()

    return render_template('index.html', report_history=log)

report_history = []

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded image file from the form
    image_file = request.files['image']
    image_name = image_file.filename
    # Open the image (and convert it to a numpy array)
    image = Image.open(image_file)
    # image_array = np.array(image)
    
    # Do something with the image array (e.g. pass it to a prediction model)
    # prediction = my_prediction_model(image_array)
    prediction = ce_prediction(image)
    
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # prediction log to be taken with timestamps
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%H:%M:%S")
    report = str(image_name) + ' submitted at ' + currentTime

    # Report history is added to a data structure
    report_history.append(report)

    # Log is created and reversed for intuitive viewing
    log = report_history
    #log = log[-5:]
    log.reverse()

    # Render the result template with the prediction text
    return render_template('index.html', prediction=prediction, report_history=log, image=encoded_string)

if __name__ == '__main__':
    app.run(debug=True)