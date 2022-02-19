import io
import base64
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained MNIST model.
model = load_model('./model.h5')

def predict_digit(image_in_binary):
    # Create an Image object from image_in_binary.
    image = Image.open(io.BytesIO(image_in_binary))

    # If the image size is NOT 116x82, resize to 116x82.
    print('Original size: ', image.size)
    if image.size != (116,82):
        image = image.resize((116,82))
    print('Reshaped size: ',image.size)

    # If the image mode is NOT gray-scale, convert RGB to gray-scale.
    
    print('Original mode: ', image.mode)
    if image.mode != 'RGB' and image.mode != '1':
        image = image.convert('RGB')
    print('Converted mode: ', image.mode)
    
    # Create a numpy array in the shape of (1, 116, 82, 3).
    # 変更
    image_array = np.array(image).reshape(1, 116, 82, 3)
    image_array = image_array / 255.0
    
    
    predictions = model.predict(image_array)
    predictions = predictions*100
    if(predictions < 50):
        predicted_digits = 0
    else:
        predicted_digits = 1

    print('Predictions: ', predictions)

    # Return the most probable digit and the probability list.
    return predicted_digits, predictions[0].tolist()
    

def predict_digit_in_base64(image_in_base64):
    # Decode the image data in base64 to the original binary data.
    image_in_binary = base64.b64decode(image_in_base64)

    # Predict the digit using predict_digit().
    return predict_digit(image_in_binary)

if __name__ == '__main__':
    # Predict the handwritten digit in a PNG image file.
    with open('data/digit-2-in-100x100x3.png', 'rb') as f:
        image_in_binary = f.read()
        predicted_digit, prediction = predict_digit(image_in_binary)
        print('Predicted digit: ', predicted_digit)
        print('Prediction details: ', prediction)

    # Predict the handwritten digit in a base64 text file encoded from PNG data.
    with open('data/digit-2-in-100x100x3.txt', 'r') as f:
        image_in_base64 = f.read()
        predicted_digit, prediction = predict_digit_in_base64(image_in_base64)
        print('Predicted digit: ', predicted_digit)
        print('Prediction details: ', prediction)
