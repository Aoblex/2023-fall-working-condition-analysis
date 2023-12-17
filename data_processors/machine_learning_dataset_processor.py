""" Processes machine learning dataset """

import os
import pandas as pd
import numpy as np
from data_config import *

""" Dataset parameters """
raw_dataset_folder = RAW_DATASET_FOLDER 
processed_dataset_folder = PROCESSED_DATASET_FOLDER
filename = "机器学习数据集.xlsx"

""" Read dataset """
dataset = pd.ExcelFile(os.path.join(raw_dataset_folder, filename))
