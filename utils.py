import os
import numpy as np
import pandas as pd
import math
import re
from typing import Tuple
from datetime import datetime
from config.data_config import *

def extract_time(time_string: str) -> int:
    """Convert time_strings to integer timestamps
    Input:
        time_string: a string like '[2023-10-13 21:16:08.727]'.
    Return:
        An integer representing number of seconds elapsed since 1970/1/1.
    >>> extract_time("[1970-01-03 00:00:00.000]")
    144000
    >>> extract_time("[2023-10-13 21:16:08.727]")
    1697202969
    """
    time_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]')
    time_match = time_pattern.search(time_string)
    if time_match:
        time_str = time_match.group(1)
        time_extracted = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        return math.ceil(time_extracted.timestamp())
    else:
        raise ValueError("Not a string of time.")

def soot_conversion(row: pd.DataFrame) -> float:
    return (row["空气流量"]/1.205 + row["油耗量<kg/h>"]/751.7)/3600*row[SOOT_NAME]

def check_file_exists(file_path):
    if os.path.exists(file_path):
        pass
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def clean_data(X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove outliers from the data based on z-scores.

    Parameters:
        X (pd.DataFrame): Features DataFrame.
        y (pd.DataFrame): Target DataFrame.
        threshold (float): Z-score threshold for outlier detection.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned features and target DataFrames.
    
    Examples:
        # Create a sample DataFrame
        >>> X = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5],
        ...                   'Feature2': [2, 3, 4, 5, 6]})
        >>> y = pd.DataFrame({'Target': [10, 11, 12, 30, 13]})
        >>> X_cleaned, y_cleaned = clean_data(X, y, threshold=1.0)
        >>> X_cleaned
           Feature1  Feature2
        0         1         2
        1         2         3
        2         3         4
        3         5         6
        >>> y_cleaned
           Target
        0      10
        1      11
        2      12
        3      13

    """
    # Compute z-scores
    z_scores = (y - y.mean()) / y.std()

    # Identify outlier indices
    outlier_indices = (z_scores.abs() < threshold).to_numpy()

    # Remove outliers from X and y
    X_cleaned = X.iloc[outlier_indices, :].reset_index(drop=True)
    y_cleaned = y.iloc[outlier_indices].reset_index(drop=True)

    return X_cleaned, y_cleaned


if __name__ == "__main__":
    import doctest
    doctest.testmod()