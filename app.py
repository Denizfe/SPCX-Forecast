"""SPCX Forecast App — Streamlit UI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    DataFetchError,
    ModelError,
    SUPPORTED_TICKERS,
    evaluate,
    fetch_data,
    forecast,
    naive_baseline_metrics,
    train_model,
    validate_ticker,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = 3600  # 1 hour


st.set_page_config(page_title="SPCX Forecast", page_icon="📈", layout="wide")

st.title("SPCX Forecast App")
st.caption("Prophet tabanlı günlük hisse fiyat tahmini — eğitim amaçlıdır.")

st.warning("Not financial advice, for educational purposes. / Yatırım tavsiyesi değildir, eğitim amaçlıdır.")


@st.cache_data(ttl=CACHE_TTL, show_spinner="Veri indiriliyor, model eğitiliyor…")
def cached_forecast_pipeline(ticker: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Fetch data, train, evaluate, and forecast in one cached call."""
    from forecasting import DEFAULT_PARAMS

    df = fetch_data(ticker)
    params = DEFAULT_PARAMS
    model = train_model(df, params)
    metrics = evaluate(model, df)
    baseline = naive_baseline_metrics(df)
    pred = forecast(model, horizon, history_df=df)
    return df, pred, params, metrics, baseline


def build_chart(history: pd.DataFrame, prediction: pd.DataFrame, ticker: str) -> go.Figure:
    hist = history.copy()
    hist["ds"] = pd.to_datetime(hist["ds"])
    pred = prediction.copy()
    pred["ds"] = pd.to_datetime(pred["ds"])

    last_hist = hist["ds"].max()
    future = pred[pred["ds"] > last_hist]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            name="Geçmiş",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future["ds"],
            y=future["yhat"],
            mode="lines",
            name="Tahmin",
            line=dict(color="#16a34a", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(22, 163, 74, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Güven aralığı",
            showlegend=True,
        )
    )
    fig.update_layout(
        title=f"{ticker} — Geçmiş & Tahmin",
        xaxis_title="Tarih",
        yaxis_title="Kapanış ($)",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


if "has_run" not in st.session_state:
    st.session_state.has_run = False

with st.sidebar:
    st.header("Ayarlar")
    ticker_input = st.text_input("Sembol", value="SPCX", help="Desteklenen sembollerden biri")
    horizon = st.slider("Tahmin ufku (iş günü)", min_value=1, max_value=30, value=5)
    st.markdown("**Desteklenen semboller:**")
    st.code(", ".join(SUPPORTED_TICKERS))

run = st.button("Tahmin Oluştur", type="primary")
should_run = run or not st.session_state.has_run

if should_run:
    try:
        ticker = validate_ticker(ticker_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    try:
        df, pred, params, metrics, baseline = cached_forecast_pipeline(ticker, horizon)
        st.session_state.has_run = True
    except DataFetchError as exc:
        st.error(f"Veri hatası: {exc}")
        st.stop()
    except ModelError as exc:
        st.error(f"Model hatası: {exc}")
        st.stop()
    except Exception as exc:
        logger.exception("Beklenmeyen hata")
        st.error(f"Beklenmeyen bir hata oluştu: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model MAPE", f"{metrics['mape']:.2f}%")
    col2.metric("Naif MAPE", f"{baseline['mape']:.2f}%")
    col3.metric("Model RMSE", f"${metrics['rmse']:.2f}")
    col4.metric("Naif RMSE", f"${baseline['rmse']:.2f}")

    beats_baseline = metrics["mape"] < baseline["mape"]
    if beats_baseline:
        st.info(
            f"Model, naif baz çizgisinden (dünkü kapanış = yarın) "
            f"**{baseline['mape'] - metrics['mape']:.2f} puan** daha iyi MAPE ile performans gösteriyor."
        )
    else:
        st.warning(
            f"Model naif baz çizgisinden daha kötü performans gösteriyor "
            f"(+{metrics['mape'] - baseline['mape']:.2f} puan MAPE). "
            "Günlük hisse tahmini doğası gereği zordur; sonuçları dikkatli yorumlayın."
        )

    st.plotly_chart(build_chart(df, pred, ticker), use_container_width=True)

    last_hist = pd.to_datetime(df["ds"]).max()
    future_rows = pred[pd.to_datetime(pred["ds"]) > last_hist][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    future_rows.columns = ["Tarih", "Tahmin", "Alt", "Üst"]
    future_rows["Tahmin"] = future_rows["Tahmin"].map(lambda x: f"${x:.2f}")
    future_rows["Alt"] = future_rows["Alt"].map(lambda x: f"${x:.2f}")
    future_rows["Üst"] = future_rows["Üst"].map(lambda x: f"${x:.2f}")
    st.subheader("Tahmin tablosu")
    st.dataframe(future_rows, hide_index=True, use_container_width=True)

    with st.expander("Model yapılandırması"):
        st.json(params)
else:
    st.info("Sembol ve ufku seçip **Tahmin Oluştur** butonuna basın.")
