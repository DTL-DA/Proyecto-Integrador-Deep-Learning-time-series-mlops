"""
Pruebas unitarias de los modelos Keras guardados en app/.
Verifican la forma de entrada/salida, el tipo de las predicciones y que el
número de lags del modelo coincida con el de su escalador.
Ejecutar con: pytest tests/test_model.py -v
"""
import os

import numpy as np
import pytest

HORIZONS = [7, 14, 21, 28]
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def model_path(H: int) -> str:
    return os.path.join(APP_DIR, f"model_h{H}.keras")


def scalers_path(H: int) -> str:
    return os.path.join(APP_DIR, f"scalers_h{H}.joblib")


def load_assets(H: int):
    pytest.importorskip("tensorflow")
    if not (os.path.exists(model_path(H)) and os.path.exists(scalers_path(H))):
        pytest.skip(f"model_h{H}.keras o scalers_h{H}.joblib no disponibles")
    import joblib
    import tensorflow as tf
    return tf.keras.models.load_model(model_path(H)), joblib.load(scalers_path(H))


@pytest.mark.parametrize("H", HORIZONS)
def test_model_output_shape(H):
    """El modelo debe producir salida de forma (n_samples, H)."""
    model, _ = load_assets(H)
    n_lags = model.input_shape[1]
    y_pred = model.predict(np.random.rand(5, n_lags).astype(np.float32), verbose=0)
    assert y_pred.shape == (5, H)


@pytest.mark.parametrize("H", HORIZONS)
def test_model_prediction_types(H):
    """Las predicciones deben ser float finitos."""
    model, _ = load_assets(H)
    n_lags = model.input_shape[1]
    pred_list = model.predict(np.random.rand(1, n_lags).astype(np.float32), verbose=0)[0].tolist()
    assert len(pred_list) == H
    assert all(isinstance(v, float) and np.isfinite(v) for v in pred_list)


@pytest.mark.parametrize("H", HORIZONS)
def test_model_input_matches_scaler(H):
    """El número de lags del modelo debe coincidir con el del escalador de entrada."""
    model, scalers = load_assets(H)
    assert model.input_shape[1] == scalers["x"].n_features_in_
    assert scalers["y"].n_features_in_ == H
