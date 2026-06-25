import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_data_to_df(path: Path, parameter_row_names: list, target_row_names: list, profile: list) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load CSV data and prepare it for machine learning.

    Args:
        path: Path to the CSV file
        parameter_row_names: List of column names to use as input features (X)
        target_row_names: List of column names to use as target values (y)
        profile: List of column names to drop from the DataFrame
    Returns:
        Tuple containing:
        - formatted_df: DataFrame with all processed data
        - train_X_raw: 2D numpy array of shape (n_samples, n_parameters) for training inputs
        - train_Y_raw: 2D numpy array of shape (n_samples, 1) for training targets
    """
    df: pd.DataFrame = pd.read_csv(path)
    df = df.drop(profile).reset_index(drop=True)
    missing_params = [p for p in parameter_row_names if p not in df.columns]
    missing_targets = [t for t in target_row_names if t not in df.columns]
    if missing_params or missing_targets:
        raise ValueError(f"Missing columns: {missing_params + missing_targets}")

    formatted_df: pd.DataFrame = _build_formatted_df(
        df=df,
        parameter_row_names=parameter_row_names,
        target_row_names=target_row_names
    )

    _calculate_target_var_std(formatted_df)

    train_X_raw, train_Y_raw = _prepare_train_data(formatted_df, parameter_row_names)

    return formatted_df, train_X_raw, train_Y_raw

def _build_formatted_df(df: pd.DataFrame, parameter_row_names: list, target_row_names: list) -> pd.DataFrame:
    """
    Build a formatted DataFrame with parameter columns and a combined target column.
    """
    formatted_df = pd.DataFrame()
    target_rows = []

    for param in parameter_row_names:
        formatted_df[param] = df[param]

    for target in target_row_names:
        target_rows.append(df[target])

    formatted_df["target"] = [list(l) for l in zip(*target_rows)]

    return formatted_df

def _calculate_target_var_std(df: pd.DataFrame) -> None:
    """
    Calculate variance and standard deviation for each target row.
    Modifies the DataFrame in-place.
    """
    target = df["target"]

    df["Y_vars"] = np.array([np.var(y) for y in target])

    df["Y_std"] = np.array([np.std(y) for y in target])

def _prepare_train_data(df: pd.DataFrame, parameter_row_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data as 2D numpy arrays for machine learning.
    Returns arrays, does NOT store them in the DataFrame.
    """
    train_X_raw = df[parameter_row_names].values

    target = df["target"]
    train_Y_raw = np.array([[np.mean(y)] for y in target])

    return train_X_raw, train_Y_raw

