import argparse

import tensorflow as tf
import numpy as np


def train(train_batch_size, test_batch_size, epochs, output_dir):
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=train_batch_size, epochs=epochs)
    model.evaluate(x_test, y_test, batch_size=test_batch_size)
    model.save(output_dir, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='models')
    args = vars(parser.parse_args())
    train(**args)
