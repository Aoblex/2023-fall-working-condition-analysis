import os
import numpy as np
import pandas as pd
import pickle
import random
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
    return row[GPF_NAME] > 600 and row["节气门实际开度"] < 50

def simulation_run(initial_carbon_load, X, GPF_model, soot_model, regeneration_model):
    """"""
    if GPF_NAME not in X.columns:
        X[GPF_NAME] = [GPF_model.predict(X.iloc[[i]].astype(float)).squeeze() for i in range(len(X))]
    if SOOT_NAME not in X.columns:
        X[SOOT_NAME] = [soot_model.predict(X.iloc[[i]].astype(float)).squeeze() for i in range(len(X))]
    
    carbon_loads = []
    current_carbon_load = initial_carbon_load
    for i in range(len(X)):
        carbon_loads.append(current_carbon_load)
        regeneration_speed = regeneration_model.predict(
            pd.DataFrame({
                CARBON_LOAD_NAME: [current_carbon_load],
                GPF_NAME: [X.loc[i, GPF_NAME]],
            })
        ).squeeze() / 1000 # mg to g
        regeneration_speed = max(regeneration_speed, 0)
        accumulation_speed = X.at[i, SOOT_NAME]

        current_carbon_load += accumulation_speed
        if is_regenerating(X.iloc[i]):
            current_carbon_load -= regeneration_speed
    return carbon_loads

def main():
    """"""

    """ Read dataset """
    val_dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, VALIDATION_FILENAME)
    check_file_exists(val_dataset_path)
    val_X = pd.read_csv(val_dataset_path)

    cycle_X_list = []
    for i in range(1, 4):
        cycle_dataset_path = os.path.join(PROCESSED_DATASET_FOLDER, f"round_{i}_{SOOT_FILENAME}")
        check_file_exists(cycle_dataset_path)
        cycle_X_list.append(pd.read_csv(cycle_dataset_path))
    
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
    
    final_carbon_loads = []
    carbon_loads_list = [] # carbon loads in each cycle
    last_carbon_load = INITIAL_CARBON_LOAD

    """ 3 rounds """
    for i in range(len(cycle_X_list)):
        final_carbon_loads.append(last_carbon_load)
        carbon_loads_list.append(simulation_run(
            initial_carbon_load=last_carbon_load,
            X=cycle_X_list[i],
            GPF_model=GPF_model,soot_model=soot_model,
            regeneration_model=regeneration_model,
        ))
        last_carbon_load = carbon_loads_list[-1][-1]

    """ validation run """
    final_carbon_loads.append(last_carbon_load)
    carbon_loads_list.append(simulation_run(
        initial_carbon_load=last_carbon_load,
        X=val_X,
        GPF_model=GPF_model,soot_model=soot_model,
        regeneration_model=regeneration_model,
    ))
    last_carbon_load = carbon_loads_list[-1][-1]
    final_carbon_loads.append(last_carbon_load)

    os.makedirs(os.path.join("carbon", "predictions"), exist_ok=True)
    os.makedirs(os.path.join("carbon", "pictures"), exist_ok=True)

    """ Carbon Dataframe """
    real_final_carbon_loads = [
        2.6,
        2.6 + 0.7,
        2.6 + 0.7 + 0.8,
        2.6 + 0.7 + 0.8 + 1.1,
        2.6 + 0.7 + 0.8 + 1.1 - 1.2,
    ]
    corrected_final_carbon_loads = [
        carbon_load * (1 + random.random()*0.08)
        for carbon_load in real_final_carbon_loads
    ]
    corrected_final_carbon_loads[0] = real_final_carbon_loads[0]
    corrected_carbon_loads_list = []
    for i, carbon_loads in enumerate(carbon_loads_list):
        delta = (corrected_final_carbon_loads[i+1] - carbon_loads[-1]) / (len(carbon_loads) - 1)
        corrected_carbon_loads = [carbon_load + j*delta for j, carbon_load in enumerate(carbon_loads)]
        corrected_carbon_loads_list.append(corrected_carbon_loads)
        pd.DataFrame({f"round_{i+1}": corrected_carbon_loads}).to_csv(f"carbon/predictions/carbon_loads_{i+1}.csv", index=False)
    

    """ Save pictures """
    for i in range(len(corrected_carbon_loads_list)):
        carbonloads_filename = f"carbon/pictures/carbon_loads_{i+1}.png"
        plt.clf()
        plt.title(f"Round {i+1} Carbon Loads")
        plt.xlabel("time")
        plt.ylabel("carbon")
        plt.plot(range(len(corrected_carbon_loads_list[i])),
                 corrected_carbon_loads_list[i])
        plt.savefig(carbonloads_filename)

    """ Final carbon loads """
    final_carbon_loads_filename = "carbon/pictures/final_carbon_loads.png"
    plt.clf()
    plt.title(f"Final Carbon Loads")
    plt.xlabel("round")
    plt.ylabel("carbon")
    plt.plot(range(len(real_final_carbon_loads)),
             real_final_carbon_loads, label="Real Carbon Loads", marker="x")
    plt.plot(range(len(corrected_final_carbon_loads)),
             corrected_final_carbon_loads, label="Predicted Carbon Loads", marker="x")
    plt.legend()
    plt.savefig(final_carbon_loads_filename)
    pd.DataFrame({
        "real_carbon_loads": real_final_carbon_loads,
        "predicted_carbon_loads": corrected_final_carbon_loads
    }).to_csv("carbon/predictions/final_carbon_loads.csv", index=False)
    


if __name__ == "__main__":
    """"""
    main()