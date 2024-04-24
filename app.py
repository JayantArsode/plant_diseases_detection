from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from vit_keras import vit

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model(
    'plant_disease_detection_model_checkpoint_05.keras')

# Define your class label dictionary
# Define your class label dictionary
class_labels = {0: 'Apple__black_rot',
                1: 'Apple__healthy',
                2: 'Apple__rust',
                3: 'Apple__scab',
                4: 'Chili__healthy',
                5: 'Chili__leaf curl',
                6: 'Chili__leaf spot',
                7: 'Chili__whitefly',
                8: 'Chili__yellowish',
                9: 'Corn__common_rust',
                10: 'Corn__gray_leaf_spot',
                11: 'Corn__healthy',
                12: 'Corn__northern_leaf_blight',
                13: 'Cucumber__diseased',
                14: 'Cucumber__healthy',
                15: 'Gauva__diseased',
                16: 'Gauva__healthy',
                17: 'Grape__black_measles',
                18: 'Grape__black_rot',
                19: 'Grape__healthy',
                20: 'Grape__leaf_blight_(isariopsis_leaf_spot)',
                21: 'Lemon__diseased',
                22: 'Lemon__healthy',
                23: 'Mango__diseased',
                24: 'Mango__healthy',
                25: 'Pepper_bell__bacterial_spot',
                26: 'Pepper_bell__healthy',
                27: 'Potato__early_blight',
                28: 'Potato__healthy',
                29: 'Potato__late_blight',
                30: 'Rice__brown_spot',
                31: 'Rice__healthy',
                32: 'Rice__hispa',
                33: 'Rice__leaf_blast',
                34: 'Rice__neck_blast',
                35: 'Soybean__bacterial_blight',
                36: 'Soybean__caterpillar',
                37: 'Soybean__diabrotica_speciosa',
                38: 'Soybean__downy_mildew',
                39: 'Soybean__healthy',
                40: 'Soybean__mosaic_virus',
                41: 'Soybean__powdery_mildew',
                42: 'Soybean__rust',
                43: 'Soybean__southern_blight',
                44: 'Sugarcane__bacterial_blight',
                45: 'Sugarcane__healthy',
                46: 'Sugarcane__red_rot',
                47: 'Sugarcane__red_stripe',
                48: 'Sugarcane__rust',
                49: 'Tomato__bacterial_spot',
                50: 'Tomato__early_blight',
                51: 'Tomato__healthy',
                52: 'Tomato__late_blight',
                53: 'Tomato__leaf_mold',
                54: 'Tomato__mosaic_virus',
                55: 'Tomato__septoria_leaf_spot',
                56: 'Tomato__spider_mites_(two_spotted_spider_mite)',
                57: 'Tomato__target_spot',
                58: 'Tomato__yellow_leaf_curl_virus',
                59: 'Wheat__brown_rust',
                60: 'Wheat__healthy',
                61: 'Wheat__septoria',
                62: 'Wheat__yellow_rust'}


def predict(image):
    # Resize the image to 224x224
    image = image.resize((224, 224))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Preprocess the image
    image_array = image_array / 255.0  # Rescale to [0,1]

    # Add an extra dimension for the batch size
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)

    # Get the class label and probability
    class_index = np.argmax(prediction)
    class_label = class_labels.get(class_index, 'Unknown')
    probability = prediction[0][class_index]

    return class_label, probability


@app.route('/predict', methods=['POST'])
def predict_disease():
    # Get the image file from the request
    file = request.files['image']

    # Open the image file
    image = Image.open(file.stream)

    # Preprocess and predict
    class_label, probability = predict(image)

    # Return the prediction as a JSON response
    response = {
        'class_label': class_label,
        'probability': float(probability)
    }
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
