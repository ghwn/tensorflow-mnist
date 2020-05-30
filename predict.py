import argparse
import io

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_image_object(image_bytes):
    """Create a PIL.Image object that contains the image bytes."""
    f = io.BytesIO()
    f.write(image_bytes)
    image = Image.open(f)
    return image


def to_jpeg(image_object, mode='RGB'):
    """Converts the given PIL.Image object's format into `JPEG`."""
    f = io.BytesIO()
    image_object.convert(mode).save(f, format='JPEG')
    image = Image.open(f)
    return image


def preprocess(image_bytes):
    """Preprocesses image bytes so that they can be input into the model.
    
    Returns:
        - a numpy array of which shape is (1, 28, 28)
    """
    image = create_image_object(image_bytes)
    image = to_jpeg(image, mode='L')    # 'L' for grayscale
    image = image.resize(size=(28, 28))
    image = np.array(image, dtype=np.uint8)
    image = np.expand_dims(image, axis=0)
    return image


def predict(model_dir, image_bytes):
    """Predicts a digit the image represents."""
    image = preprocess(image_bytes)
    model = tf.keras.models.load_model(model_dir)
    output = model.predict(image).squeeze()
    prediction = np.argmax(output)
    print('prediction: %s' % prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="path to MNIST model directory")
    parser.add_argument('--image', type=str, required=True, help="path to an image")
    args = parser.parse_args()    
    predict(args.model_dir, open(args.image, 'rb').read())
