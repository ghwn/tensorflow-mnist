import argparse

import tensorflow as tf


def evaluate(model_dir, batch_size):
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0

    model = tf.keras.models.load_model(model_dir)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss: %.6f, Test accuracy: %.2f%%" % (loss, accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help="path to MNIST model directory")
    parser.add_argument('--batch_size', type=int, default=50)
    args = vars(parser.parse_args())
    evaluate(**args)
