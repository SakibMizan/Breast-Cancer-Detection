from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import os
import keras
import numpy as np
from glob import glob
import tensorflow as tf
import tensorflow.image as tfi
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization

# Callbacks 
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Metrics
from keras.metrics import MeanIoU
import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
from PIL import Image

app = Flask(__name__)

class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
            'pooling':self.pooling
        }

class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
        }

class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f
        # return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "bn":self.bn
        }


# Load your pre-trained model (make sure to replace 'path_to_model' with your model's actual path)
model = load_model("LungCancerAttentionUNet.h5", custom_objects={
    'MeanIoU': tf.keras.metrics.MeanIoU,
    'EncoderBlock': EncoderBlock,
    'DecoderBlock': DecoderBlock,
    'AttentionGate': AttentionGate
})

def preprocess_image(image_file, target_size):
    image_stream = io.BytesIO(image_file.read())
    img = load_img(image_stream, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img / 255.0  # Normalize the image to [0, 1] range


def calculate_white_percentage(processed_mask):
    white_pixels = np.sum(processed_mask > 0.5)
    total_pixels = processed_mask.size
    white_percentage = (white_pixels / total_pixels) * 100
    return white_percentage

def classify_cancer(processed_mask):
    white_percentage = calculate_white_percentage(processed_mask)
    if white_percentage > 1:
        print("Percentage of white area:", white_percentage)
        return "Cancer"
    else:
        print("Percentage of white area:", white_percentage)
        return "Non-Cancer"
        
def encode_image(image_array):
    """Converts a NumPy array to a base64-encoded PNG."""
    img = Image.fromarray((image_array * 255).astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            file.seek(0)  # Reset file pointer to the beginning
            processed_image = preprocess_image(file, target_size=(256, 256))
            prediction = model.predict(processed_image)
            processed_mask = (prediction > 0.5).astype(np.float32)
            cancer_status = classify_cancer(processed_mask)
            if cancer_status == "Cancer":
                original_img_encoded = encode_image(np.squeeze(processed_image))
                mask_img_encoded = encode_image(np.squeeze(processed_mask))
                return jsonify({
                    "cancer_status": cancer_status,
                    "original_img": original_img_encoded,
                    "mask_img": mask_img_encoded
                })
            return jsonify({"cancer_status": cancer_status})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
