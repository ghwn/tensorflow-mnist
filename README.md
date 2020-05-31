# TensorFlow MNIST example


## Prerequisites

- TensorFlow 2.2.0


## Training

    $ python train.py \
        --train_batch_size=64 \
        --test_batch_size=50 \
        --epochs=5 \
        --output_dir="./models"


## Evaluation

    $ python evaluate.py \
        --model_dir="./models" \
        --batch_size=50


## Prediction

    $ python predict.py \
        --model_dir="./models" \
        --image=<path/to/image>


## Running gRPC Server

1. Generate Python code from `digit_classification.proto`, which provide gRPC server and client interfaces.

        $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./digit_classification.proto

2. Run server example.

    To use this server, you have to also implement its client satisfying `digit_classification.proto`.

        $ python server.py --model_dir="./models"
