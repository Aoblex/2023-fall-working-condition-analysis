import os
import pickle
import shutil
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from config.model_config import *
from config.data_config import *
from config.simulation_config import *

def main():
    """"""

    """ Set model name """
    poly_name = f"{POLY_NAME}_regeneration_speed"


    """ Remove previous poly """
    poly_folder = os.path.join(MODEL_FOLDER, poly_name)
    if os.path.exists(poly_folder):
       shutil.rmtree(poly_folder)
    os.makedirs(poly_folder, exist_ok=True)


    """ Read dataset """
    excel_dataset = pd.read_excel(os.path.join(RAW_DATASET_FOLDER, RAW_REGENERATION_FILENAME))


    """ Select features """
    X: pd.DataFrame = excel_dataset.iloc[:, :-1]
    y: pd.DataFrame = excel_dataset.iloc[:, -1]


    """ Train model """
    poly_model = make_pipeline(
        PolynomialFeatures(degree=3),
        LinearRegression()
    )
    poly_model.fit(X, y)


    """ Save model """
    model_filename = f"{POLY_NAME}_model.pkl"
    model_filefolder = os.path.join(MODEL_FOLDER, poly_name, PARAMETERS_FOLDER)
    model_filepath = os.path.join(model_filefolder, model_filename)
    os.makedirs(model_filefolder, exist_ok=True)

    with open(model_filepath, "wb") as f:
        pickle.dump(poly_model, f)
    
        
if __name__ == "__main__":
    """"""
    main()
