# Predicción de Velocidad del Viento con MLP y MLOps

**Proyecto Integrador de Aprendizaje Profundo**  
Deep Learning · 3er Semestre · Universidad del Norte

---

## Descripción

Este libro documenta el desarrollo de un sistema de predicción multipaso de la **velocidad del viento** (m/s) y de su **volatilidad** para la estación meteorológica **A001 — Brasília, DF** del INMET (Instituto Nacional de Meteorología de Brasil), con redes neuronales **MLP multisalida**, validación cruzada temporal con `timeseries-cv` y un despliegue reproducible con FastAPI, Docker y CI.

## Objetivo

Predecir el nivel de la serie y modelar su variabilidad en horizontes de **7, 14, 21 y 28 días**, a partir únicamente de sus observaciones históricas, con:

- una matriz de experimentos lag × horizonte (4 × 4 = 16 configuraciones);
- búsqueda sistemática de la mejor arquitectura MLP (capas, neuronas y dropout por capa);
- comparación de tres estrategias de validación de `timeseries-cv` (Group K-Fold, Forward Chaining y K-Fold) y un hold-out temporal final;
- análisis riguroso de residuos (prueba BDS, ACF, distribución) y comparación con referencias ingenuas;
- despliegue mediante una API REST (FastAPI), Docker y CI.

## Dataset

| Fuente | INMET — Instituto Nacional de Meteorología |
|---|---|
| Estación | A001 — Brasília, DF |
| Período | 2000–2026 en el EDA; 2020–2026 en los modelos (2 496 días) |
| Frecuencia original | Horaria, agregada a media diaria |
| Variables objetivo | Velocidad del viento (m/s) y volatilidad móvil de 28 días de sus retornos logarítmicos |
| Hold-out | Último 10 % (249 días, del 25 de febrero al 31 de octubre de 2026) |

## Estructura del libro

| Capítulo | Contenido |
|---|---|
| 1. EDA | Estadísticas descriptivas, retornos logarítmicos, volatilidad, ACF y prueba ADF |
| 2. Búsqueda de arquitectura | Búsqueda exhaustiva de 584 candidatos y reevaluación del Top-10 en las 16 combinaciones |
| 3. Entrenamiento y evaluación | Tres estrategias × 16 combinaciones; MAPE, MAE, RMSE, MSE y BDS en validación; modelo final y hold-out |
| 4. Análisis de residuos | BDS por fold y en el hold-out, ACF, QQ-plot, residuos en el tiempo y referencias ingenuas |
| 5. Modelo de volatilidad | MLP para la volatilidad con el mismo protocolo, comparado con la persistencia |

## Resultados principales

- **Nivel del viento.** RMSE de validación de 0,561 m/s (Forward Chaining) y de 0,567 a 0,575 m/s en el hold-out. El MLP supera a la persistencia en cerca de un 25 % y al `MLPRegressor` de referencia en un 17 %, pero no mejora a la media en el hold-out: la serie diaria tiene poca predictibilidad a partir de su propio pasado.
- **Volatilidad.** En el hold-out el MLP reduce el RMSE de la persistencia en 5,8 %, 12,5 % y 16,9 % con H = 14, 21 y 28, porque aprende la reversión de la volatilidad a su nivel medio.

## Repositorio

El código fuente, la API, el Dockerfile y las pruebas unitarias están en  
[github.com/DTL-DA/Proyecto-Integrador-Deep-Learning-time-series-mlops](https://github.com/DTL-DA/Proyecto-Integrador-Deep-Learning-time-series-mlops).
