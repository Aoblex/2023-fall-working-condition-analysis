import os
import shutil
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor

from config.svr_config import *
from config.model_config import *
from config.data_config import *
from utils import check_file_exists, clean_data

def main():
    """Train SVR Soot Model"""

    
    """ Set model name """
    svr_name = f"{SVR_NAME}_soot"
    soot_filename = SOOT_FILENAME

    """ Remove previous SVR """
    svr_folder = os.path.join(MODEL_FOLDER, svr_name)
    if os.path.exists(svr_folder):
        shutil.rmtree(svr_folder)
    os.makedirs(svr_folder, exist_ok=True)


    """ Read dataset and split into Xs and ys """
    dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, soot_filename)
    check_file_exists(dataset_path)
    dataset = pd.read_csv(dataset_path)
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]


    """ Clean data """
    X, y = clean_data(X, y, threshold=3.0) # Clean outliers


    """ Kfold validation """
    svr_list, mse_list, all_metrics = [], [], pd.DataFrame(columns=MODEL_METRICS.keys())
    kfold = KFold(n_splits=N_SPLITS, shuffle=True) 
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y), start=1):


        """ Split dataset """
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        """ Train model """
        log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)

        svr_pipeline = make_pipeline(
            StandardScaler(),
            SVR(kernel=KERNEL, C=C, tol=TOL),)
        
        svr_model = TransformedTargetRegressor(
            regressor=svr_pipeline,
            transformer=log_transformer,)
        
        svr_model.fit(X_train, y_train)
        svr_list.append(svr_model)


        """ Save model """
        model_filename = f"{fold}th_fold_model.pkl"
        model_filefolder = os.path.join(MODEL_FOLDER, svr_name, PARAMETERS_FOLDER)
        model_filepath = os.path.join(model_filefolder, model_filename)
        os.makedirs(model_filefolder, exist_ok=True)

        with open(model_filepath, "wb") as f:
            pickle.dump(svr_model, f)


        """ Make predictions """
        metrics_filename = f"{fold}th_fold_metrics.csv"
        metrics_filefolder = os.path.join(MODEL_FOLDER, svr_name, METRICS_FOLDER)
        metrics_filepath = os.path.join(metrics_filefolder, metrics_filename)
        os.makedirs(metrics_filefolder, exist_ok=True)

        metrics_df: pd.DataFrame = pd.DataFrame(columns=MODEL_METRICS.keys())
        y_true, y_pred = y_test, svr_model.predict(X_test)
        model_metrics = [metric_function(y_true, y_pred) for metric_function in MODEL_METRICS.values()]
        metrics_df.loc[len(metrics_df)] = model_metrics
        all_metrics.loc[fold] = model_metrics
        mse_list.append(mean_squared_error(y_true, y_pred))
        
        metrics_df.to_csv(metrics_filepath, index=False)


        """ Save predictions """
        prediction_filename = f"{fold}th_fold_predictions.csv"
        prediction_filefolder = os.path.join(MODEL_FOLDER, svr_name, PREDICTIONS_FOLDER)
        prediction_filepath = os.path.join(prediction_filefolder, prediction_filename)
        os.makedirs(prediction_filefolder, exist_ok=True)

        predictions = pd.DataFrame(data={
            "y_true": y_true,
            "y_pred": y_pred,
        })
        predictions.to_csv(prediction_filepath, index=False)


    """ Find best model """
    best_model_index = mse_list.index(min(mse_list))
    best_filename = "best_model.txt"
    best_filefolder = os.path.join(MODEL_FOLDER, svr_name)
    best_filepath = os.path.join(best_filefolder, best_filename)
    with open(best_filepath, 'w') as f:
        f.write(f"{best_model_index}th fold")


    """ Write all metrics """ 
    all_metrics_filename = "all_metrics.csv"
    all_metrics_filefolder = os.path.join(MODEL_FOLDER, svr_name, METRICS_FOLDER)
    all_metrics_filepath = os.path.join(all_metrics_filefolder, all_metrics_filename)
    all_metrics.to_csv(all_metrics_filepath, index=True)

if __name__ == "__main__":
    main()