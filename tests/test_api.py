"""
Pruebas unitarias de la API FastAPI.
Ejecutar con: pytest tests/test_api.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock de _load_assets para no requerir modelos entrenados durante las pruebas
# ---------------------------------------------------------------------------

def _make_mock_assets(n_lags: int, horizon: int):
    """Crea mocks de modelo Keras y scalers de sklearn."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros((1, horizon), dtype=np.float32)

    mock_sx = MagicMock()
    mock_sx.n_features_in_ = n_lags
    mock_sx.transform.side_effect = lambda x: x

    mock_sy = MagicMock()
    mock_sy.inverse_transform.side_effect = lambda x: x

    return mock_model, {"x": mock_sx, "y": mock_sy}


@pytest.fixture
def client():
    """TestClient con modelos mockeados para H=7 (n_lags=14)."""
    with patch("app.api._load_assets", side_effect=lambda h: _make_mock_assets(14, h)):
        from app.api import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests de la API
# ---------------------------------------------------------------------------

VALID_LAGS_14 = [3.2, 2.8, 3.1, 3.4, 2.9, 3.0, 3.3, 3.5, 2.7, 3.0, 3.2, 3.1, 2.9, 3.3]


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize("horizon", [7, 14, 21, 28])
def test_valid_prediction(client, horizon):
    resp = client.post("/predict", json={"lags": VALID_LAGS_14, "horizon": horizon})
    assert resp.status_code == 200
    data = resp.json()
    assert data["horizon"] == horizon
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == horizon


def test_prediction_returns_floats(client):
    resp = client.post("/predict", json={"lags": VALID_LAGS_14, "horizon": 7})
    assert resp.status_code == 200
    for val in resp.json()["prediction"]:
        assert isinstance(val, float)


def test_invalid_horizon(client):
    resp = client.post("/predict", json={"lags": VALID_LAGS_14, "horizon": 10})
    assert resp.status_code == 422


def test_wrong_lag_count(client):
    """Enviar menos lags de los esperados debe retornar 422."""
    resp = client.post("/predict", json={"lags": [1.0] * 7, "horizon": 7})
    assert resp.status_code == 422


def test_empty_lags(client):
    resp = client.post("/predict", json={"lags": [], "horizon": 7})
    assert resp.status_code == 422


def test_negative_lags_rejected(client):
    lags_neg = [-1.0] * 14
    resp = client.post("/predict", json={"lags": lags_neg, "horizon": 7})
    assert resp.status_code == 422


def test_response_has_n_lags_field(client):
    resp = client.post("/predict", json={"lags": VALID_LAGS_14, "horizon": 7})
    assert resp.status_code == 200
    assert resp.json()["n_lags_recibidos"] == 14
