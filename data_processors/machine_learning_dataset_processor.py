"""
Processes machine learning dataset

1. Change column names.
2. Align features and responses according to time.
3. Seperate datasets for soot and for GPF.
"""

import os
import pandas as pd
from data_config import *

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
raw_datasets = []
for X_sheet_name, Y_sheet_name in dataset_sheet_names:
    X_dataset = excel_dataset.parse(X_sheet_name)
    Y_dataset = excel_dataset.parse(Y_sheet_name)
    raw_datasets.append(X_dataset, Y_dataset)


def get_soot_dataset(X_dataset, Y_dataset) -> pd.DataFrame:
    """ Generate soot dataset from Xs and ys.
    Inputs:
        X_dataset: dataframe containing working condition features.
        Y_dataset: dataframe containing soot values at certain times.
    Return:
        a pandas dataframe with column names changed containing Xs and ys for soot prediction.
    """

