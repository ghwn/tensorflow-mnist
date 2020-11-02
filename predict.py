import argparse

import cv2
import numpy as np
import tensorflow as tf


def preprocess(image):
    """Preprocesses image bytes so that they can be input into the model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    image = np.reshape(image, (-1, 28, 28, 1))
    return image


def predict(model_dir, image_path):
    """Predicts a digit the image represents."""
    image = cv2.imread(image_path)
    image = preprocess(image)
    model = tf.keras.models.load_model(model_dir)
    output = model.predict(image)
    prediction = np.argmax(output)
    return prediction


def main(model_dir, image_path):
    prediction = predict(model_dir, image_path)
    print('prediction: %s' % prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="path to MNIST model directory")
    parser.add_argument('--image', type=str, required=True, help="path to an image")
    args = parser.parse_args()
    main(args.model_dir, args.image)
