import torch

DNN_NAME = "dnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 3e-4
ACTIVATION_FUNCTION = torch.nn.ReLU()

BATCH_SIZE = 256
NUM_EPOCHS = 64
SPLIT_RATE = 0.8
