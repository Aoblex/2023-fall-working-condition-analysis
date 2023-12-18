"""
Processes machine learning dataset
Align features and responses according to time.
"""

from typing import List, Tuple
import os
import math
import time
import pandas as pd
from data_config import *
from utils import extract_time, soot_conversion


""" Dataset parameters """
raw_dataset_folder = RAW_DATASET_FOLDER 
processed_dataset_folder = PROCESSED_DATASET_FOLDER
filename = "机器学习数据集.xlsx"
soot_selected_features = SOOT_SELECTED_FEATURES
GPF_selected_features = GPF_SELECTED_FEATURES


""" Read dataset """
excel_dataset = pd.ExcelFile(os.path.join(raw_dataset_folder, filename))
excel_dataset_sheet_names = excel_dataset.sheet_names


""" Get sheet names """
dataset_sheet_names = [
    ('原始数据X-1', '原始数据Y-1'),
    ('原始数据X-2', '原始数据Y-2'),
    ('原始数据X-3', '原始数据Y-3'),
]
validation_sheet_name = '验证原始数据X'


""" Convert sheets to dataframe """
raw_datasets: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
for X_sheet_name, Y_sheet_name in dataset_sheet_names:
    
    """ Convert to dataframe """
    X_dataset: pd.DataFrame = excel_dataset.parse(X_sheet_name)
    Y_dataset: pd.DataFrame = excel_dataset.parse(Y_sheet_name)
    
    """ Process time """
    X_dataset.loc[:, "时间"] = X_dataset["时间"].apply(extract_time) # Convert time strings to integers.
    Y_dataset.loc[:, "时间"] = Y_dataset["时间"].apply(extract_time) # Convert time strings to start time.
    Y_dataset.loc[:, "时间"] = (Y_dataset["时间"] + Y_dataset["Relative Time"]).apply(math.ceil) # Compute real timestamp.
    
    """ Feature selection """
    response_name = "Exhaust Soot Concentration"
    X_dataset = X_dataset[soot_selected_features] # Select features for soot.
    Y_dataset = Y_dataset[["时间", response_name]] # Select soot values.

    """ Process soot """
    Y_dataset.loc[:, response_name] = Y_dataset[response_name].apply(lambda soot: max(soot, 0)) # Avoid negative values.

    """ Average by time """
    X_dataset = X_dataset.groupby("时间").mean().reset_index()
    Y_dataset = Y_dataset.groupby("时间").mean().reset_index()

    """ Append to raw_datasets """
    raw_datasets.append((X_dataset, Y_dataset,))


""" Concatenate all dataset. """
all_X_dataset = pd.concat([raw_dataset[0] for raw_dataset in raw_datasets], axis=0, ignore_index=True)
all_Y_dataset = pd.concat([raw_dataset[1] for raw_dataset in raw_datasets], axis=0, ignore_index=True)


def get_soot_dataset(X_dataset: pd.DataFrame, Y_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Generate soot dataset from Xs and ys.
    Inputs:
        X_dataset: dataframe containing working condition features.
        Y_dataset: dataframe containing soot values at certain times.
    Return:
        a pandas dataframe with column names changed containing Xs and ys for soot prediction.
    """
    
    """ Inner connect Xs and Ys by time. """
    soot_dataset = pd.merge(X_dataset, Y_dataset, on="时间", how="inner") # Merge Xs and Ys.
    soot_dataset = soot_dataset.drop(columns=["时间"]) # Time is no longer needed since Xs and Ys are merged.
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
    GPF_X_dataset = X_dataset.drop(columns=[GPF_NAME, "时间"])
    GPF_Y_dataset = X_dataset[GPF_NAME]
    GPF_dataset = pd.concat([GPF_X_dataset, GPF_Y_dataset], axis=1)
    return GPF_dataset


""" Generate soot dataset and GPF dataset for each cycle. """
for i, (X_dataset, Y_dataset) in enumerate(raw_datasets):

    """ Write soot dataset. """
    soot_dataset = get_soot_dataset(X_dataset, Y_dataset)
    soot_filename = f"round_{i+1}_soot_dataset.csv"
    soot_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, soot_filename), index=False)

    """ Write GPF dataset. """
    GPF_dataset = get_GPF_dataset(X_dataset)
    GPF_filename = f"round_{i+1}_GPF_dataset.csv"
    GPF_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, GPF_filename), index=False)


""" Generate overall soot dataset and GPF dataset """
all_soot_dataset = get_soot_dataset(all_X_dataset, all_Y_dataset)
all_GPF_dataset = get_GPF_dataset(all_X_dataset)
all_soot_filename = f"all_soot_dataset.csv"
all_GPF_filename = f"all_GPF_dataset.csv"
all_soot_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, all_soot_filename), index=False)
all_GPF_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, all_GPF_filename), index=False)


""" Generate validation soot dataset """
validation_dataset: pd.DataFrame = excel_dataset.parse(validation_sheet_name)
validation_dataset.loc[:, "时间"] = validation_dataset["时间"].apply(extract_time) # Convert time strings to integers.
validation_dataset = validation_dataset[soot_selected_features] # Select features for soot.
validation_dataset = validation_dataset.groupby("时间").mean().reset_index() # Group and average by time.
validation_dataset = validation_dataset.drop(columns=["时间", GPF_NAME]) # We don't need time and GPF in validation set.
validation_filename= f"validation_dataset.csv"
validation_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, validation_filename), index=False)