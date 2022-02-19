import base64
import json
from flask import Flask, render_template, request
from flask.wrappers import Response
from predictor import predict_digit_in_base64

app = Flask(__name__, static_folder='.', template_folder='.')

@app.route('/', methods=['GET'])
def hello():
    print('GET / received.')
    # Return index.html.
    return render_template('index.html')

@app.route('/predictions', methods=['POST'])
def predict():
    print('POST /prediction received.')

    # Retrieve the image data in base64 from the request body.
    data = request.get_json(force=True)
    print('Request body: \n', data)
    image_in_base64 = data['image']

    # Predict the digit in the image using predict_digit_in_base64().
    predicted_digit, prediction = predict_digit_in_base64(image_in_base64)

    # Create a response in JSON.
    response = {
        'predictedDigit': predicted_digit,
        'prediction': prediction
    }
    response_in_json_string = json.dumps(response)
    print('Response: \n', response_in_json_string)

    # Return the response.
    return response_in_json_string
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')