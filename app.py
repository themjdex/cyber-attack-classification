import sys
import logging
import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from catboost import CatBoostClassifier

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from src.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class AddTraffic(BaseModel):
    data: list
    features: list


app = FastAPI()


@app.get('/')
def main():
    return 'Мы в точке входа для предсказания вредоносности трафика'


def load_models(training_pipeline_params: TrainingPipelineParams):
    model = joblib.load(training_pipeline_params.output_model_path)
    return model


@app.get('/health')
def check_models(training_pipeline_params: TrainingPipelineParams):
    model = load_models(training_pipeline_params)
    if model is None:
        logger.error('app/check_models models are None')
        raise HTTPException(status_code=500, detail='Models are unavailable')


@app.get('/check_schema')
def check_schema(features: list, training_pipeline_params: TrainingPipelineParams):
    if not set(training_pipeline_params.feature_params.features).issubset(
        set(features)
    ):
        logger.error('app/check_schema missing columns')
        raise HTTPException(
            status_code=400, detail=f'Missing features in schema {features}'
        )


def make_predict(
    data: list,
    features: list,
    model: CatBoostClassifier,
    training_pipeline_params: TrainingPipelineParams,
) -> list:
    check_schema(features, training_pipeline_params)

    df = pd.DataFrame(data, columns=features)

    _, predictions = predict_model(model, df)

    logger.debug(f'predictions: {predictions[0]}')

    return predictions.tolist()[0]


@app.post('/predict/')
def predict(request: AddTraffic):
    logger.debug('app/predict run')

    config_path = 'configs/config.yaml'
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )
    logger.debug(f'app/predict training_pipeline_params: {training_pipeline_params}')

    check_models(training_pipeline_params)
    logger.debug('app/predict check_models passed')

    model = load_models(training_pipeline_params)

    return make_predict(
        request.data, request.features, model, training_pipeline_params
    )


if __name__ == "__main__":
    uvicorn.run("app:app", port=os.getenv("PORT", 8000))
