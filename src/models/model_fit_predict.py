import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    roc_auc_score
)
from typing import Dict, Union, Tuple
import joblib

from src.entities.train_params import TrainingParams
from src.entities.feature_params import FeatureParams

Classifier = Union[CatBoostClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams, feature_param: FeatureParams
) -> Classifier:
    """Train and save model from configs"""

    model = CatBoostClassifier(
        loss_function=train_params.loss_function,
        n_estimators=train_params.n_estimators,
        learning_rate=train_params.learning_rate,
        depth=train_params.depth,
        random_seed=train_params.random_state,
        bagging_temperature=train_params.bagging_temperature,
        verbose=True,
        thread_count=train_params.thread_count
    )
    model.fit(features, target, cat_features=feature_param.cat_features)
    return model


def predict_model(
    model: CatBoostClassifier, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict model from configs"""
    preds = model.predict(features, thread_count=-1)
    predicted_proba = model.predict_proba(features)
    return predicted_proba, preds


def evaluate_model(
    predicted_proba: np.ndarray, predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """Evaluate model from configs"""
    return {
        "f1_score": f1_score(target, predicts, average="weighted"),
        "precision": precision_score(target, predicts, average="weighted"),
        "recall": recall_score(target, predicts, average="weighted"),
        "accuracy": accuracy_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicted_proba, average='weighted', multi_class='ovo'),
    }


def serialize_model(model, output: str) -> str:
    """Serialize model from configs"""
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output
