import torch

DNN_NAME = "dnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 3e-4

BATCH_SIZE = 256
NUM_EPOCHS = 5
SPLIT_RATE = 0.8
