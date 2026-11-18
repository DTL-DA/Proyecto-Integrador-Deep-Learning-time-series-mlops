"""
Descarga y preprocesa los datos horarios de velocidad del viento del INMET
para la estación A001 (Brasília, DF) en los años 2000–2026.
Genera data/dataset.csv con columnas [date, wind_speed] a resolución diaria.
Los años sin archivo disponible en el portal se omiten con advertencia.
"""

import io
import os
import zipfile
import requests
import pandas as pd

STATION_CODE = "A001"
# YEARS = [2021, 2022, 2023, 2024]  # rango corto (uso rápido / pruebas)
YEARS = list(range(2000, 2027))     # todos los años disponibles en INMET
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

WIND_COL_CANDIDATES = [
    "VENTO, VELOCIDADE HORARIA (m/s)",
    "VENTO, VELOCIDADE HORARIA (M/S)",
    "Vento, Velocidade Horária (m/s)",
]


def _find_wind_col(df: pd.DataFrame) -> str:
    for c in WIND_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"Columna de viento no encontrada. Columnas disponibles: {df.columns.tolist()}"
    )


def _parse_inmet_csv(raw: bytes) -> pd.DataFrame:
    """Lee un CSV del INMET (separador ';', decimal ',', 8 filas de cabecera)."""
    text = raw.decode("latin-1")
    lines = text.splitlines()
    # Las primeras 8 líneas son metadatos; la fila 9 es el encabezado real
    header_line = 8
    data_text = "\n".join(lines[header_line:])
    df = pd.read_csv(
        io.StringIO(data_text),
        sep=";",
        decimal=",",
        encoding="latin-1",
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    # Construir timestamp
    date_col = [c for c in df.columns if c.lower().startswith("data")][0]
    hour_col = [c for c in df.columns if c.lower().startswith("hora")][0]
    # Formatos INMET: viejo "00:00" / nuevo "0000 UTC"
    hour_raw = df[hour_col].astype(str).str.strip()
    has_colon = hour_raw.str.contains(":", na=False)
    time_str = hour_raw.where(has_colon, other=hour_raw.str[:2] + ":" + hour_raw.str[2:4])
    date_raw = df[date_col].astype(str).str.strip()
    dt_combined = date_raw + " " + time_str
    df["datetime"] = pd.to_datetime(dt_combined, format="%Y-%m-%d %H:%M", errors="coerce")
    mask_nat = df["datetime"].isna()
    if mask_nat.any():
        df.loc[mask_nat, "datetime"] = pd.to_datetime(
            dt_combined[mask_nat], format="%Y/%m/%d %H:%M", errors="coerce"
        )
    wind_col = _find_wind_col(df)
    df["wind_speed"] = pd.to_numeric(df[wind_col], errors="coerce")
    df = df[["datetime", "wind_speed"]].dropna(subset=["datetime"])
    # -9999 es el código de faltante del INMET
    df.loc[df["wind_speed"] < 0, "wind_speed"] = float("nan")
    return df


def download_year(year: int) -> pd.DataFrame:
    url = f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
    print(f"  Descargando {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    frames = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if STATION_CODE in name and name.upper().endswith(".CSV"):
                raw = zf.read(name)
                frames.append(_parse_inmet_csv(raw))
    if not frames:
        raise FileNotFoundError(
            f"No se encontró el archivo de la estación {STATION_CODE} en el ZIP de {year}."
        )
    return pd.concat(frames, ignore_index=True)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega velocidad del viento horaria → media diaria."""
    df = df.set_index("datetime").sort_index()
    daily = df["wind_speed"].resample("D").mean().dropna()
    daily = daily.reset_index().rename(columns={"datetime": "date", "wind_speed": "wind_speed"})
    return daily


def main():
    frames = []
    for year in YEARS:
        try:
            df_year = download_year(year)
            frames.append(df_year)
            print(f"  -> {year}: {len(df_year)} registros horarios cargados.")
        except Exception as exc:
            print(f"  ADVERTENCIA - {year}: {exc}")

    if not frames:
        raise RuntimeError("No se pudo descargar ningún año de datos.")

    df_full = pd.concat(frames, ignore_index=True)
    daily = build_daily(df_full)
    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset diario guardado en: {OUTPUT_PATH}")
    print(f"Periodo: {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print(f"Observaciones: {len(daily)}")


if __name__ == "__main__":
    main()
