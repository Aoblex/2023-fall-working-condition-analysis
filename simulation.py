import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import check_file_exists
from config.simulation_config import *
from config.data_config import *
from config.model_config import *
from config.svr_soot_config import *
from config.svr_GPF_config import *
from config.dnn_soot_config import *
from config.dnn_GPF_config import *
from config.stacking_soot_config import *
from config.stacking_GPF_config import *

def is_regenerating(row: pd.Series) -> bool:
    return row[GPF_NAME] > 600

def main():
    """"""

    """ Read dataset """
    dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, VALIDATION_FILENAME)
    check_file_exists(dataset_path)
    X = pd.read_csv(dataset_path)


    """ Load model """
    GPF_model_filepath = os.path.join(
        MODEL_FOLDER, f"{STACKING_NAME}_GPF",
        PARAMETERS_FOLDER, "1th_fold_model.pkl"
    )
    with open(GPF_model_filepath, "rb") as f:
        GPF_model = pickle.load(f)

    soot_model_filepath = os.path.join(
        MODEL_FOLDER, f"{STACKING_NAME}_soot",
        PARAMETERS_FOLDER, "1th_fold_model.pkl" 
    ) 
    with open(soot_model_filepath, "rb") as f:
        soot_model = pickle.load(f)
    
    regeneration_model_filepath = os.path.join(
        MODEL_FOLDER, f"{POLY_NAME}_regeneration_speed",
        PARAMETERS_FOLDER, f"{POLY_NAME}_model.pkl"
    )
    with open(regeneration_model_filepath, "rb") as f:
        regeneration_model = pickle.load(f)

    """ Start simulation"""
    X[GPF_NAME] = [GPF_model.predict(X.iloc[[i]].astype(float)).squeeze() for i in range(len(X))]
    X[SOOT_NAME] = [soot_model.predict(X.iloc[[i]].astype(float)).squeeze() for i in range(len(X))]
    
    carbon_loads = []
    current_carbon_load = INITIAL_CARBON_LOAD
    for i in range(len(X)):

        carbon_loads.append(current_carbon_load)
        regeneration_speed = regeneration_model.predict(
            pd.DataFrame({
                CARBON_LOAD_NAME: [current_carbon_load],
                GPF_NAME: [X.loc[i, GPF_NAME]],
            })
        ).squeeze()
        accumulation_speed = X.at[i, SOOT_NAME]

        current_carbon_load += accumulation_speed
        #if is_regenerating(X.iloc[i]):
        #    current_carbon_load -= regeneration_speed

    """ Save pictures """
    carbonloads_filename = "carbon_loads.png"
    plt.clf()
    plt.title("Carbon loads")
    plt.xlabel("Time")
    plt.ylabel("Carbon Load")
    plt.plot(range(len(carbon_loads)), carbon_loads)
    plt.savefig(carbonloads_filename)

if __name__ == "__main__":
    """"""
    main()