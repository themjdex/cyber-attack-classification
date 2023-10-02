# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities.split_params import SplittingParams
from src.entities.feature_params import FeatureParams


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def strip_spaces(data: pd.DataFrame) -> pd.DataFrame:
    new_columns = {}
    for col in data.columns:
        new_columns[col] = col.strip()
    data = data.rename(columns=new_columns)
    return data


def fill_na(data: pd.DataFrame, col: str = 'Flow Bytes/s') -> pd.DataFrame:
    data[col] = data[col].fillna(data[col].median())
    return data


def split_features_target(data: pd.DataFrame, params: FeatureParams
                          ) -> Tuple[pd.DataFrame, pd.Series]:
    features = data.drop(params.target_col, axis=1)
    target = data[params.target_col]

    return features, target


def split_train_val_data(
        features: pd.DataFrame, target: pd.Series, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target,
        test_size=params.val_size, random_state=params.random_state, shuffle=params.shuffle
    )

    return features_train, features_valid, target_train, target_valid
