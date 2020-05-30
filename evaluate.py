import argparse

import tensorflow as tf


def evaluate(model_dir):
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0

    model = tf.keras.models.load_model(model_dir)
    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help="path to MNIST model directory")
    args = vars(parser.parse_args())
    evaluate(**args)
