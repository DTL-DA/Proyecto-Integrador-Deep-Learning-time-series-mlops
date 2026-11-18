"""
API FastAPI para predicción multi-step de velocidad del viento.
Acepta los últimos N valores observados (lags) y el horizonte deseado,
y devuelve H predicciones (m/s).

Uso:
    uvicorn app.api:app --reload

Ejemplo:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"lags": [2.1, 1.9, 2.3, 2.0, 1.8, 2.2, 2.4, 2.1, 2.0, 1.9, 2.3,
                       2.5, 2.2, 2.0, 1.8, 2.1, 2.3, 2.2, 2.0, 1.9, 2.1], "horizon": 14}'
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException

from app.schemas import TimeSeriesData, PredictionResponse

app = FastAPI(
    title="Wind Speed Forecast API",
    description=(
        "Predicción multi-step de velocidad del viento (m/s) "
        "con el mejor MLP encontrado por búsqueda de arquitecturas. "
        "Estación INMET A001 — Brasília, DF."
    ),
    version="1.0.0",
)

VALID_HORIZONS = (7, 14, 21, 28)
APP_DIR = os.path.dirname(__file__)

# Carga perezosa de modelos y escaladores
_models: dict = {}
_scalers: dict = {}


def _load_assets(horizon: int):
    if horizon not in _models:
        model_path = os.path.join(APP_DIR, f"model_h{horizon}.keras")
        scalers_path = os.path.join(APP_DIR, f"scalers_h{horizon}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}. "
                "Ejecute primero 3_model_training.ipynb para generarlo."
            )
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(
                f"Scalers no encontrados: {scalers_path}."
            )
        _models[horizon] = tf.keras.models.load_model(model_path)
        _scalers[horizon] = joblib.load(scalers_path)
    return _models[horizon], _scalers[horizon]


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Wind Speed Forecast API v1.0.0"}


@app.post("/predict", response_model=PredictionResponse, summary="Predicción multi-step")
def predict(data: TimeSeriesData):
    """
    Recibe los últimos N valores de velocidad del viento (m/s) como `lags`
    y el `horizon` deseado (7, 14, 21 o 28 días).
    Devuelve una lista de H predicciones.

    El número de lags enviados debe coincidir con el número de lags de entrada
    del modelo almacenado: 21 en los cuatro modelos actuales.
    """
    H = data.horizon
    try:
        model, scalers = _load_assets(H)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    sx = scalers["x"]
    sy = scalers["y"]
    n_expected = sx.n_features_in_

    if len(data.lags) != n_expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperaban {n_expected} lags "
                f"(el modelo para H={H} fue entrenado con lag={n_expected}), "
                f"pero se recibieron {len(data.lags)}."
            ),
        )

    features = np.array(data.lags, dtype=np.float32).reshape(1, -1)
    features_sc = sx.transform(features)
    pred_sc = model.predict(features_sc, verbose=0)
    prediction = sy.inverse_transform(pred_sc)[0].tolist()

    return PredictionResponse(
        horizon=H,
        n_lags_recibidos=len(data.lags),
        prediction=prediction,
    )
