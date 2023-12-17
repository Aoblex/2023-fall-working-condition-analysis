"""
Processes machine learning dataset

1. Change column names.
2. Align features and responses according to time.
3. Seperate datasets for soot and for GPF.
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
validation_sheet_name = [
    '验证原始数据',
]

""" Convert sheets to dataframe """
raw_datasets: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
for X_sheet_name, Y_sheet_name in dataset_sheet_names:
    X_dataset = excel_dataset.parse(X_sheet_name)
    Y_dataset = excel_dataset.parse(Y_sheet_name)
    raw_datasets.append((X_dataset, Y_dataset,))


def get_soot_dataset(X_dataset: pd.DataFrame, Y_dataset: pd.DataFrame) -> pd.DataFrame:
    """ Generate soot dataset from Xs and ys.
    Inputs:
        X_dataset: dataframe containing working condition features.
        Y_dataset: dataframe containing soot values at certain times.
    Return:
        a pandas dataframe with column names changed containing Xs and ys for soot prediction.
    """
    
    """ Process time """
    response_name = "Exhaust Soot Concentration"
    X_dataset.loc[:, "时间"] = X_dataset["时间"].apply(extract_time) # Convert time strings to integers.
    Y_dataset.loc[:, "时间"] = Y_dataset["时间"].apply(extract_time) # Convert time strings to start time.
    Y_dataset.loc[:, "时间"] = (Y_dataset["时间"] + Y_dataset["Relative Time"]).apply(math.ceil) # Compute real timestamp.
    X_dataset = X_dataset[soot_selected_features] # Select features for soot.
    Y_dataset = Y_dataset[["时间", response_name]] # Select soot values.
    Y_dataset.loc[:, response_name] = Y_dataset[response_name].apply(lambda soot: max(soot, 0)) # Avoid negative values.
    
    """ Average by time """
    X_dataset = X_dataset.groupby("时间").mean().reset_index()
    Y_dataset = Y_dataset.groupby("时间").mean().reset_index()

    """ Inner connect Xs and Ys by time."""
    soot_dataset = pd.merge(X_dataset, Y_dataset, on="时间", how="inner") # Merge Xs and Ys.
    soot_dataset = soot_dataset.drop(columns=["时间"]) # Time is no longer needed since Xs and Ys are merged.
    soot_dataset = soot_dataset.rename(columns=RENAME_DICT) # Rename columns
    soot_dataset.loc[:, "soot"] = soot_dataset.apply(soot_conversion, axis=1) # Unit conversion of soot.
    return soot_dataset
    
for i, (X_dataset, Y_dataset) in enumerate(raw_datasets):
    soot_dataset = get_soot_dataset(X_dataset, Y_dataset)
    soot_filename = f"round_{i+1}_soot.csv"
    soot_dataset.to_csv(os.path.join(PROCESSED_DATASET_FOLDER, soot_filename), index=False)