"""
Processes machine learning dataset
Align features and responses according to time.
"""

from typing import List, Tuple
import os
import math
import pandas as pd
from config.data_config import *
from utils import extract_time, soot_conversion


def get_soot_dataset(X_dataset: pd.DataFrame, Y_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Generate soot dataset from Xs and ys.
    Inputs:
        X_dataset: dataframe containing working condition features.
        Y_dataset: dataframe containing soot values at certain times.
    Return:
        a pandas dataframe with column names changed containing Xs and ys for soot prediction.
    """
    
    """ Inner connect Xs and Ys by time. """
    soot_dataset = pd.merge(X_dataset, Y_dataset, on=ID_NAME, how="inner") # Merge Xs and Ys.
    soot_dataset = soot_dataset.drop(columns=[ID_NAME]) # Time is no longer needed since Xs and Ys are merged.
    soot_dataset.loc[:, SOOT_NAME] = soot_dataset.apply(soot_conversion, axis=1) # Unit conversion of soot.
    return soot_dataset


def get_GPF_dataset(X_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Generate GPF dataset from X dataset.
    Inputs:
        X_dataset: dataframe containing working condition features.
    Return:
        a pandas dataframe with column names changed containing Xs and ys for GPF prediction.
    """

    """ Switch columns. """
    GPF_X_dataset = X_dataset.drop(columns=[GPF_NAME, ID_NAME])
    GPF_Y_dataset = X_dataset[GPF_NAME]
    GPF_dataset = pd.concat([GPF_X_dataset, GPF_Y_dataset], axis=1)
    return GPF_dataset

def main():

    """ Read dataset """
    excel_dataset = pd.ExcelFile(os.path.join(RAW_DATASET_FOLDER, RAW_DATA_FILENAME))


    """ Get sheet names """
    dataset_sheet_names = DATASET_SHEET_NAMES
    validation_sheet_name = VALIDATION_SHEET_NAME


    """ Convert sheets to dataframe """
    raw_datasets: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for X_sheet_name, Y_sheet_name in dataset_sheet_names:
        
        """ Convert to dataframe """
        X_dataset: pd.DataFrame = excel_dataset.parse(X_sheet_name)
        Y_dataset: pd.DataFrame = excel_dataset.parse(Y_sheet_name)
        
        """ Process time """
        X_dataset.loc[:, ID_NAME] = X_dataset[ID_NAME].apply(extract_time) # Convert time strings to integers.
        Y_dataset.loc[:, ID_NAME] = Y_dataset[ID_NAME].apply(extract_time) # Convert time strings to start time.
        Y_dataset.loc[:, ID_NAME] = (Y_dataset[ID_NAME] + Y_dataset["Relative Time"]).apply(math.ceil) # Compute real timestamp.
        
        """ Feature selection """
        response_name = "Exhaust Soot Concentration"
        X_dataset = X_dataset[SOOT_SELECTED_FEATURES] # Select features for soot.
        Y_dataset = Y_dataset[[ID_NAME, response_name]] # Select soot values.

        """ Process soot """
        Y_dataset.loc[:, response_name] = Y_dataset[response_name].apply(lambda soot: max(soot, 0)) # Avoid negative values.

        """ Average by time """
        X_dataset = X_dataset.groupby(ID_NAME).mean().reset_index()
        Y_dataset = Y_dataset.groupby(ID_NAME).mean().reset_index()

        """ Append to raw_datasets """
        raw_datasets.append((X_dataset, Y_dataset,))


    """ Concatenate all dataset. """
    all_X_dataset = pd.concat([raw_dataset[0] for raw_dataset in raw_datasets], axis=0, ignore_index=True)
    all_Y_dataset = pd.concat([raw_dataset[1] for raw_dataset in raw_datasets], axis=0, ignore_index=True)


    """ Generate soot dataset and GPF dataset for each cycle. """
    for i, (X_dataset, Y_dataset) in enumerate(raw_datasets):

        """ Make sure the folder exists. """
        os.makedirs(PROCESSED_DATASET_FOLDER, exist_ok=True)

        """ Write soot dataset. """
        soot_dataset = get_soot_dataset(X_dataset, Y_dataset)
        soot_filename = f"round_{i+1}_{SOOT_FILENAME}"
        soot_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, soot_filename), index=False)

        """ Write GPF dataset. """
        GPF_dataset = get_GPF_dataset(X_dataset)
        GPF_filename = f"round_{i+1}_{GPF_FILENAME}"
        GPF_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, GPF_filename), index=False)


    """ Generate overall soot dataset and GPF dataset """
    all_soot_dataset = get_soot_dataset(all_X_dataset, all_Y_dataset)
    all_GPF_dataset = get_GPF_dataset(all_X_dataset)
    all_soot_filename = SOOT_FILENAME
    all_GPF_filename = GPF_FILENAME
    os.makedirs(PROCESSED_DATASET_FOLDER, exist_ok=True)
    all_soot_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, all_soot_filename), index=False)
    all_GPF_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, all_GPF_filename), index=False)


    """ Generate validation soot dataset """
    validation_dataset: pd.DataFrame = excel_dataset.parse(validation_sheet_name)
    validation_dataset.loc[:, ID_NAME] = validation_dataset[ID_NAME].apply(extract_time) # Convert time strings to integers.
    validation_dataset = validation_dataset[SOOT_SELECTED_FEATURES] # Select features for soot.
    validation_dataset = validation_dataset.groupby(ID_NAME).mean().reset_index() # Group and average by time.
    validation_dataset = validation_dataset.drop(columns=[ID_NAME, GPF_NAME]) # We don't need time and GPF in validation set.
    validation_filename= VALIDATION_FILENAME
    os.makedirs(PROCESSED_DATASET_FOLDER, exist_ok=True)
    validation_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, validation_filename), index=False)

if __name__ == "__main__":
    main()