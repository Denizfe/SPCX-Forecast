"""Stock price forecasting with Prophet."""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

SUPPORTED_TICKERS = ["SPCX", "SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]

DEFAULT_PARAMS: dict[str, Any] = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",
    "use_regressors": True,
}

EXPERIMENTS_CSV = Path(__file__).parent / "experiments.csv"

# Backtest CI threshold: fail workflow if MAPE exceeds this (percent).
MAPE_THRESHOLD = 15.0


class DataFetchError(Exception):
    """Raised when market data cannot be retrieved."""


class ModelError(Exception):
    """Raised when model training or evaluation fails."""


def is_supported_ticker(ticker: str) -> bool:
    return ticker.upper().strip() in SUPPORTED_TICKERS


def validate_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not is_supported_ticker(ticker):
        supported = ", ".join(SUPPORTED_TICKERS)
        raise ValueError(f"'{ticker}' desteklenmiyor. Desteklenen semboller: {supported}")
    return ticker


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _prepare_prophet_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to Prophet format with optional regressors."""
    if raw.empty:
        raise DataFetchError("Veri bulunamadı.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        volume = raw["Volume"]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
    else:
        close = raw["Close"]
        volume = raw.get("Volume", pd.Series(index=raw.index, dtype=float))

    df = pd.DataFrame({"ds": close.index, "y": close.values.astype(float)})
    df["volume"] = volume.reindex(close.index).fillna(0).astype(float).values
    df["rolling_mean_7"] = df["y"].rolling(window=7, min_periods=1).mean()
    df["rsi"] = _compute_rsi(df["y"]).fillna(50.0)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


def fetch_data(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV history and return Prophet-ready dataframe."""
    ticker = validate_ticker(ticker)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    raw = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:
            last_exc = exc
        if attempt < 2:
            time.sleep(2 ** attempt)

    if raw is None or raw.empty:
        if last_exc is not None:
            raise DataFetchError(f"'{ticker}' için veri indirilemedi: {last_exc}") from last_exc
        raise DataFetchError(
            f"'{ticker}' için geçerli fiyat verisi bulunamadı. "
            "Yahoo Finance geçici olarak yanıt vermiyor olabilir; birkaç dakika sonra tekrar deneyin."
        )

    return _prepare_prophet_df(raw)


def train_model(df: pd.DataFrame, params: dict[str, Any] | None = None) -> Prophet:
    """Fit a Prophet model with optional regressors."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    use_regressors = params.pop("use_regressors", True)

    model = Prophet(
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        seasonality_mode=params.get("seasonality_mode", "multiplicative"),
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )

    if use_regressors:
        for reg in ("volume", "rolling_mean_7", "rsi"):
            if reg in df.columns:
                model.add_regressor(reg)

    try:
        model.fit(df)
    except Exception as exc:
        raise ModelError(f"Model eğitimi başarısız: {exc}") from exc

    return model


def forecast(model: Prophet, days: int, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate future predictions with confidence intervals."""
    if days < 1:
        raise ValueError("Tahmin süresi en az 1 gün olmalıdır.")

    future = model.make_future_dataframe(periods=days, freq="B")

    if history_df is not None and any(c in history_df.columns for c in ("volume", "rolling_mean_7", "rsi")):
        hist = history_df.set_index("ds")
        for reg in ("volume", "rolling_mean_7", "rsi"):
            if reg in hist.columns:
                last_val = hist[reg].iloc[-1]
                future[reg] = future["ds"].map(hist[reg]).fillna(last_val)

    try:
        return model.predict(future)
    except Exception as exc:
        raise ModelError(f"Tahmin üretilemedi: {exc}") from exc


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def naive_baseline_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Tomorrow = today's close baseline on held-out tail."""
    if len(df) < 2:
        return {"mape": float("nan"), "rmse": float("nan")}

    actual = df["y"].values[1:]
    predicted = df["y"].values[:-1]
    return {"mape": _mape(actual, predicted), "rmse": _rmse(actual, predicted)}


def evaluate(model: Prophet, df: pd.DataFrame) -> dict[str, float]:
    """Backtest via Prophet cross-validation."""
    n = len(df)
    if n < 60:
        raise ModelError("Backtest için en az 60 günlük veri gerekir.")

    initial = f"{max(30, int(n * 0.6))} days"
    period = f"{max(7, int(n * 0.05))} days"
    horizon = f"{max(5, int(n * 0.05))} days"

    try:
        cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="threads",
        )
        metrics = performance_metrics(cv, rolling_window=1)
        return {
            "mape": float(metrics["mape"].mean() * 100),
            "rmse": float(metrics["rmse"].mean()),
        }
    except Exception as exc:
        raise ModelError(f"Backtest başarısız: {exc}") from exc


def log_experiment(
    ticker: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    baseline: dict[str, float],
    notes: str = "",
) -> None:
    """Append one experiment row to experiments.csv."""
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "changepoint_prior_scale": params.get("changepoint_prior_scale"),
        "seasonality_prior_scale": params.get("seasonality_prior_scale"),
        "seasonality_mode": params.get("seasonality_mode"),
        "use_regressors": params.get("use_regressors"),
        "mape": round(metrics.get("mape", float("nan")), 4),
        "rmse": round(metrics.get("rmse", float("nan")), 4),
        "baseline_mape": round(baseline.get("mape", float("nan")), 4),
        "baseline_rmse": round(baseline.get("rmse", float("nan")), 4),
        "notes": notes,
    }
    write_header = not EXPERIMENTS_CSV.exists()
    with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tune_hyperparameters(
    df: pd.DataFrame,
    ticker: str = "SPCX",
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Simple grid search; returns best params and metrics."""
    grid = [
        {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "seasonality_mode": sm}
        for cps in [0.01, 0.05, 0.1, 0.5]
        for sps in [1.0, 10.0]
        for sm in ["additive", "multiplicative"]
    ]

    baseline = naive_baseline_metrics(df)
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    for params in grid:
        full_params = {**DEFAULT_PARAMS, **params}
        try:
            model = train_model(df, full_params)
            metrics = evaluate(model, df)
            log_experiment(ticker, full_params, metrics, baseline, notes="grid_search")
            if best_metrics is None or metrics["mape"] < best_metrics["mape"]:
                best_params = full_params
                best_metrics = metrics
        except ModelError as exc:
            logger.warning("Grid point failed: %s %s", params, exc)

    if best_params is None or best_metrics is None:
        raise ModelError("Hiçbir hiperparametre kombinasyonu backtest geçemedi.")

    return best_params, best_metrics, baseline


def run_backtest_check(ticker: str = "SPCX", threshold: float = MAPE_THRESHOLD) -> dict[str, Any]:
    """Run final model backtest; used by CI workflow."""
    df = fetch_data(ticker)
    params, metrics, baseline = tune_hyperparameters(df, ticker=ticker)

    passed = metrics["mape"] <= threshold
    return {
        "ticker": ticker,
        "params": params,
        "metrics": metrics,
        "baseline": baseline,
        "threshold": threshold,
        "passed": passed,
    }
