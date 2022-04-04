import os


class Config:
    # Data
    image_size = 224
    input_dim = 3
    num_classes = 10

    # Trainer
    train_batch_size = 32
    val_batch_size = 128
    num_workers = 8
    epochs = 40
    lr = 1e-5
    seed = 42
