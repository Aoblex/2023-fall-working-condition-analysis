import os
import pickle
import pandas as pd
import numpy as np
import shutil

import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from config.data_config import *
from config.model_config import *
from config.svr_soot_config import SVR_NAME 
from config.dnn_soot_config import DNN_NAME, DEVICE
from config.stacking_soot_config import *
from utils import get_best_svr_soot_index, check_file_exists, clean_data

class Stacking:


    def __init__(self, final_estimator=Ridge()):
        self.final_estimator = final_estimator
        best_svr_soot_index = get_best_svr_soot_index()
        self.model_filepaths = [
            os.path.join(
                MODEL_FOLDER, f"{SVR_NAME}_soot",
                PARAMETERS_FOLDER, f"{best_svr_soot_index}th_fold_model.pkl"
            ), # svr_soot 
            os.path.join(
                MODEL_FOLDER, f"{DNN_NAME}_soot",
                PARAMETERS_FOLDER, f"{DNN_NAME}_model.pkl"
            ), # dnn_soot
        ]
        self.models = []
        for filepath in self.model_filepaths:
            check_file_exists(filepath)
            with open(filepath, "rb") as f:
                self.models.append(pickle.load(f))


    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X, self.y = X, y
        model_outputs = []
        for model in self.models:
            if isinstance(model, torch.nn.Module):
                X_tensor = torch.tensor(X.values).to(torch.float32)
                X_dnn: torch.Tensor = model(X_tensor.to(DEVICE))
                X_dnn = X_dnn.squeeze().detach().cpu().numpy()
                model_outputs.append(X_dnn)
            elif isinstance(model, BaseEstimator):
                X_skl = model.predict(X)
                model_outputs.append(X_skl)
        X_meta = np.column_stack(model_outputs)
        self.final_estimator.fit(X_meta, y)


    def predict(self, X:pd.DataFrame):
        model_outputs = []
        
        for model in self.models:
            if isinstance(model, torch.nn.Module):
                X_tensor = torch.tensor(X.values).to(torch.float32)
                X_dnn: torch.Tensor = model(X_tensor.to(DEVICE))
                X_dnn = X_dnn.squeeze().detach().cpu().numpy()
                model_outputs.append(X_dnn)
            elif isinstance(model, BaseEstimator):
                X_skl = model.predict(X)
                model_outputs.append(X_skl) 
        X_meta = np.column_stack(model_outputs)

        return self.final_estimator.predict(X_meta) 


def main():
    """Train Stacking Soot Model"""

    
    """ Set model name """
    stacking_name = f"{STACKING_NAME}_soot"


    """ Remove previous stacking """
    stacking_folder = os.path.join(MODEL_FOLDER, stacking_name)
    if os.path.exists(stacking_folder):
        shutil.rmtree(stacking_folder)
    os.makedirs(stacking_folder, exist_ok=True)


    """ Read dataset and split into Xs and ys """
    dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, SOOT_FILENAME)
    check_file_exists(dataset_path)
    dataset = pd.read_csv(dataset_path)
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]


    """ Clean data """
    X, y = clean_data(X, y, threshold=3.0) # Clean outliers


    """ Kfold validation """
    stacking_list, mse_list, all_metrics = [], [], pd.DataFrame(columns=MODEL_METRICS.keys())
    kfold = KFold(n_splits=N_SPLITS, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y), start=1):


        """ Split dataset """
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        """ Train model """

        stacking_model = Stacking()
        stacking_model.fit(X_train, y_train)
        stacking_list.append(stacking_model)


        """ Save model """
        model_filename = f"{fold}th_fold_model.pkl"
        model_filefolder = os.path.join(MODEL_FOLDER, stacking_name, PARAMETERS_FOLDER)
        model_filepath = os.path.join(model_filefolder, model_filename)
        os.makedirs(model_filefolder, exist_ok=True)

        with open(model_filepath, "wb") as f:
            pickle.dump(stacking_model, f)


        """ Save predictions """
        y_true, y_pred = y_test, stacking_model.predict(X_test)
        prediction_filename = f"{fold}th_fold_predictions.csv"
        prediction_filefolder = os.path.join(MODEL_FOLDER, stacking_name, PREDICTIONS_FOLDER)
        prediction_filepath = os.path.join(prediction_filefolder, prediction_filename)
        os.makedirs(prediction_filefolder, exist_ok=True)
        predictions = pd.DataFrame(data={
            "y_true": y_true,
            "y_pred": y_pred,
        })
        predictions.to_csv(prediction_filepath, index=False)


        """ Save metrics """
        metrics_filename = f"{fold}th_fold_metrics.csv"
        metrics_filefolder = os.path.join(MODEL_FOLDER, stacking_name, METRICS_FOLDER)
        metrics_filepath = os.path.join(metrics_filefolder, metrics_filename)
        os.makedirs(metrics_filefolder, exist_ok=True)

        metrics_df: pd.DataFrame = pd.DataFrame(columns=MODEL_METRICS.keys())
        model_metrics = [metric_function(y_true, y_pred) for metric_function in MODEL_METRICS.values()]
        metrics_df.loc[len(metrics_df)] = model_metrics
        all_metrics.loc[fold] = model_metrics
        mse_list.append(mean_squared_error(y_true, y_pred))
        
        metrics_df.to_csv(metrics_filepath, index=False)


    """ Find best model """
    best_model_index = mse_list.index(min(mse_list))
    best_filename = "best_model.txt"
    best_filefolder = os.path.join(MODEL_FOLDER, stacking_name)
    best_filepath = os.path.join(best_filefolder, best_filename)
    with open(best_filepath, 'w') as f:
        f.write(f"{best_model_index}")


    """ Write all metrics """ 
    all_metrics_filename = "all_metrics.csv"
    all_metrics_filefolder = os.path.join(MODEL_FOLDER, stacking_name, METRICS_FOLDER)
    all_metrics_filepath = os.path.join(all_metrics_filefolder, all_metrics_filename)
    all_metrics.to_csv(all_metrics_filepath, index=True)




if __name__ == "__main__":
    main()


