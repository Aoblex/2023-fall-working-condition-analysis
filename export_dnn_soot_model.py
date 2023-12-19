import torch
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config.model_config import *
from config.data_config import *
from config.dnn_soot_config import *
from utils import clean_data, check_file_exists
import pandas as pd
import shutil
import pickle


class MyDataset(Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = torch.from_numpy(X.values).to(torch.float32)
        self.y = torch.from_numpy(y.values).to(torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def main():
    """ Train DNN soot model """

    """ Set model name """
    dnn_name = f"{DNN_NAME}_soot"
    soot_filename = SOOT_FILENAME


    """ Remove previous dnn """
    dnn_folder = os.path.join(MODEL_FOLDER, dnn_name)
    if os.path.exists(dnn_folder):
       shutil.rmtree(dnn_folder)
    os.makedirs(dnn_folder, exist_ok=True)


    """ Read dataset """
    dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, soot_filename)
    check_file_exists(dataset_path)
    dataset = pd.read_csv(dataset_path)
    X: pd.DataFrame = dataset.iloc[:, :-1]
    y: pd.DataFrame = dataset.iloc[:, -1]


    """ Clean dataset """
    X, y = clean_data(X, y, threshold=3.0) # Clean outliers


    """ Split dataset """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT_RATE)


    """ Prepare dataloader """
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE, 
                               shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,drop_last=False)


    """ Initialize model """
    dnn_model = torch.nn.Sequential(
        torch.nn.Linear(len(X.columns), 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    dnn_model.to(device=DEVICE)


    """ Training preparation """
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr = LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    training_losses, test_losses = [], []


    """ Start training """
    for epoch in range(NUM_EPOCHS):
        for i, (X, y) in enumerate(train_loader):
            
            """ Initialize training """
            dnn_model.train(True)
            optimizer.zero_grad()

            """ Get data"""
            X: torch.Tensor = X.to(DEVICE)
            y: torch.Tensor = y.unsqueeze(1).to(DEVICE)

            """ Compute loss """
            y_pred = dnn_model(X).to(DEVICE)
            loss: torch.Tensor = loss_fn(y, y_pred)

            """ Update model """
            loss.backward()
            optimizer.step()

            """ Compute test mse """
            dnn_model.train(False)
            current_test_mse_list = []
            for X_test, y_test in test_loader:

                """ Get dataset """
                X_test: torch.Tensor = X_test.to(DEVICE)
                y_test: torch.Tensor = y_test.unsqueeze(1).to(DEVICE)

                """ Compute loss """
                y_test_pred = dnn_model(X_test).to(DEVICE)
                current_test_mse_list.append(loss_fn(y_test, y_test_pred).item())

            """ Record current loss """
            training_losses.append(loss.item())
            test_losses.append(sum(current_test_mse_list)/len(current_test_mse_list))
    

    """ Save model """
    model_filename = f"{DNN_NAME}_model.pkl"
    model_filefolder = os.path.join(MODEL_FOLDER, dnn_name, PARAMETERS_FOLDER)
    model_filepath = os.path.join(model_filefolder, model_filename)
    os.makedirs(model_filefolder, exist_ok=True)

    with open(model_filepath, "wb") as f:
        pickle.dump(dnn_model, f)


    """ Save predictions """
    prediction_filename = f"{dnn_name}_predictions.csv"
    prediction_filefolder = os.path.join(MODEL_FOLDER, dnn_name, PREDICTIONS_FOLDER)
    prediction_filepath = os.path.join(prediction_filefolder, prediction_filename)
    os.makedirs(prediction_filefolder, exist_ok=True)

    y_test: torch.Tensor = torch.cat([y for X, y in test_loader])
    X_test: torch.Tensor = torch.cat([X for X, y in test_loader])
    y_pred: torch.Tensor = dnn_model(X_test.to(DEVICE))
    y_true = y_test.detach().cpu().numpy()
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    predictions = pd.DataFrame(data={
        "y_true": y_true,
        "y_pred": y_pred,
    })
    predictions.to_csv(prediction_filepath, index=False)


    """ Save metrics """
    metrics_filename = f"{DNN_NAME}_metrics.csv"
    metrics_filefolder = os.path.join(MODEL_FOLDER, dnn_name, METRICS_FOLDER)
    metrics_filepath = os.path.join(metrics_filefolder, metrics_filename)
    os.makedirs(metrics_filefolder, exist_ok=True)

    metrics_df: pd.DataFrame = pd.DataFrame(columns=MODEL_METRICS.keys())
    model_metrics = [metric_function(y_true, y_pred) for metric_function in MODEL_METRICS.values()]
    metrics_df.loc[len(metrics_df)] = model_metrics
    metrics_df.to_csv(metrics_filepath, index=False)


    """ Save pictures """
    losses_filename = f"{DNN_NAME}_losses.png"
    losses_filefolder = os.path.join(MODEL_FOLDER, dnn_name, PICTURES_FOLDER)
    losses_filepath = os.path.join(losses_filefolder, losses_filename)
    os.makedirs(losses_filefolder, exist_ok=True)
    plt.title("Training and Testing Loss")
    plt.xlabel("Num Batches")
    plt.ylabel("MSE Loss")
    plt.plot(range(1, len(training_losses) + 1), training_losses, label = "Training Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label = "Test Loss")
    plt.legend()
    plt.savefig(losses_filepath)

if __name__ == "__main__":
    main()