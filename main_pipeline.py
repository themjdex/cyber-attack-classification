import json
import logging
import sys
import argparse


import pandas as pd

from src.data.make_dataset import read_data, strip_spaces, fill_na, split_train_val_data, split_features_target

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)


from src.models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )

    data: pd.DataFrame = read_data(training_pipeline_params.input_data_path)
    data: pd.DataFrame = strip_spaces(data)
    data: pd.DataFrame = fill_na(data)

    logger.debug(f"Start train pipeline with params {training_pipeline_params}")
    logger.debug(f"data:  {data.shape} \n {data.info()} \n {data.nunique()}")

    features, target = split_features_target(data, training_pipeline_params.feature_params)
    logger.debug(f"Split dataframe - features: {features.shape}, target: {target.shape}")

    features_train, features_valid, target_train, target_valid = split_train_val_data(
        features, target, training_pipeline_params.splitting_params
    )

    logger.debug("Split train and valid sample:")
    logger.debug(f"""features_train: {features_train.shape}, features_valid: {features_valid.shape},
                 target_train: {target_train.shape}, target_valid: {target_valid.shape}""")

    model = train_model(
            features_train, target_train, training_pipeline_params.train_params, training_pipeline_params.feature_params
        )

    predicted_proba, preds = predict_model(model, features_valid)
    metrics = evaluate_model(predicted_proba, preds, target_valid)
    logger.debug(f"preds/ targets shapes:  {(preds.shape, target_valid.shape)}")

    # dump metrics to json
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metric is {metrics}")

    # serialize model
    serialize_model(model, training_pipeline_params.output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
