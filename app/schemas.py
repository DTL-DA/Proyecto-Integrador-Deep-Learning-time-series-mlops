from typing import List, Literal
from pydantic import BaseModel, field_validator


class TimeSeriesData(BaseModel):
    lags: List[float]
    horizon: Literal[7, 14, 21, 28] = 7

    @field_validator("lags")
    @classmethod
    def validate_lags(cls, v):
        if len(v) == 0:
            raise ValueError("lags debe contener al menos un valor.")
        if any(x < 0 for x in v):
            raise ValueError("Los valores de lags deben ser no negativos.")
        return v


class PredictionResponse(BaseModel):
    horizon: int
    n_lags_recibidos: int
    prediction: List[float]
