# time-series-mlops

Proyecto Integrador de Aprendizaje Profundo: predicción multipaso de la velocidad del viento y de su volatilidad con MLP, validación cruzada temporal y MLOps.

**Libro (Jupyter Book):** https://dtl-da.github.io/Proyecto-Integrador-Deep-Learning-time-series-mlops/

**Dataset:** INMET, estación A001 (Brasília, DF). Datos horarios agregados a media diaria; el EDA usa 2000–2026 y los modelos 2020–2026 (2 496 días).  
**Horizontes:** 7, 14, 21 y 28 días (multisalida), combinados con lags de 7, 14, 21 y 28 días.  
**Validación:** Group K-Fold, Forward Chaining y K-Fold de [timeseries-cv](https://pypi.org/project/timeseries-cv/), más un hold-out temporal del último 10 %.

## Estructura

```
time-series-mlops/
├── data/
│   ├── download_data.py           # Descarga y prepara los CSV del INMET
│   └── dataset.csv                # Serie diaria: date, wind_speed, log_return, volatility
├── notebooks/
│   ├── 1_eda.ipynb                # Exploración y análisis de la serie
│   ├── 2_architecture_search.ipynb# Búsqueda de la mejor arquitectura MLP
│   ├── 3_model_training.ipynb     # Tres estrategias de CV, modelo final y hold-out
│   ├── 4_residual_analysis.ipynb  # BDS, ACF, distribución y referencias ingenuas
│   ├── 5_volatility_model.ipynb   # Modelo de volatilidad frente a la persistencia
│   └── figs/                      # Figuras generadas
├── app/
│   ├── api.py                     # API FastAPI de inferencia
│   ├── schemas.py                 # Modelos Pydantic
│   ├── model_h{H}.keras           # Mejor MLP del nivel para H = 7, 14, 21, 28
│   ├── scalers_h{H}.joblib        # Escaladores (x, y) del modelo de nivel
│   ├── model_vol_h{H}.keras       # MLP de volatilidad para cada H
│   └── scalers_vol_h{H}.joblib
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── results/                       # CSV de métricas, búsqueda de arquitecturas y residuos
├── _config.yml, _toc.yml, intro.md# Jupyter Book
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Instalación

```bash
pip install -r requirements.txt
```

Probado con Python 3.12; las versiones están fijadas en `requirements.txt`. Keras 3.15, con el que se guardaron los modelos, exige Python 3.11 o superior, por eso el CI y el Dockerfile usan Python 3.12.

## Notebooks (orden de ejecución)

1. `1_eda.ipynb`: EDA y generación de `data/dataset.csv`.
2. `2_architecture_search.ipynb`: búsqueda exhaustiva en lag = 14, H = 7 y reevaluación del Top-10 en las 16 combinaciones; guarda `results/best_architecture_summary.csv`.
3. `3_model_training.ipynb`: tres estrategias de CV en las 16 combinaciones, modelo final por horizonte y evaluación en el hold-out; guarda `app/model_h{H}.keras`.
4. `4_residual_analysis.ipynb`: diagnóstico de residuos y comparación con referencias ingenuas.
5. `5_volatility_model.ipynb`: modelo de volatilidad; guarda `app/model_vol_h{H}.keras`.

En una CPU de 12 núcleos la ejecución completa tarda unas dos horas: cerca de una hora el cuaderno 2, 35 minutos el 3 y 13 minutos el 5.

## Resultados principales

| Modelo | Validación (Forward Chaining) | Hold-out |
|---|---|---|
| Nivel del viento, lag = 21 | RMSE 0,561 m/s, MAPE 23 % | RMSE 0,567–0,575 m/s, MAPE 24,5–25,0 % |
| Volatilidad | Mejora de 7 % a 17 % sobre la persistencia con H = 14 a 28 | Mejora de 5,8 %, 12,5 % y 16,9 % con H = 14, 21 y 28 |

El modelo de nivel supera a la persistencia (cerca de un 25 %) y al `MLPRegressor` de referencia (17 %), pero no a la media en el hold-out. Los detalles y limitaciones están en los cuadernos 3 y 4.

## API

```bash
uvicorn app.api:app --reload
```

Los cuatro modelos de nivel usan 21 lags, de modo que la petición debe enviar los últimos 21 valores diarios de velocidad del viento:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lags": [2.1, 1.9, 2.3, 2.0, 1.8, 2.2, 2.4, 2.1, 2.0, 1.9, 2.3, 2.5, 2.2, 2.0, 1.8, 2.1, 2.3, 2.2, 2.0, 1.9, 2.1], "horizon": 14}'
```

La respuesta trae `horizon`, `n_lags_recibidos` y `prediction`, una lista de H valores en m/s.

## Docker

```bash
docker build -t wind-mlops .
docker run -p 8000:8000 wind-mlops
```

## Pruebas

```bash
pytest tests/ -v
flake8 app/ --ignore=E501,W503
```

## Jupyter Book

```bash
jupyter-book build .
```

El HTML queda en `_build/html/`. Los cuadernos se publican con las salidas ya ejecutadas (`execute_notebooks: "off"`). Para publicarlo en GitHub Pages:

```bash
ghp-import -n -p -f _build/html
```
