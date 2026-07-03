# SPCX Forecast App

Prophet tabanlı günlük hisse fiyat tahmin uygulaması. Streamlit arayüzü ile geçmiş fiyatları ve güven aralıklı tahminleri görselleştirir.

> **Uyarı:** Yatırım tavsiyesi değildir; yalnızca eğitim amaçlıdır.

## Desteklenen semboller

`SPCX`, `SPY`, `QQQ`, `AAPL`, `MSFT`, `GOOGL`

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Yerel çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini açın.

## Backtest (CI)

```bash
python scripts/run_backtest.py SPCX
```

MAPE eşiğini (`forecasting.MAPE_THRESHOLD`, varsayılan %15) aşarsa script exit code 1 döner.

## Streamlit Community Cloud ile deploy

1. Bu repoyu GitHub'a push edin.
2. [share.streamlit.io](https://share.streamlit.io) üzerinden **New app** oluşturun.
3. Repository, branch ve main file path olarak `app.py` seçin.
4. **Deploy** — ekstra secret gerekmez.

## Proje yapısı

```
forecast/
├── app.py                 # Streamlit arayüzü
├── forecasting.py         # Veri, model, backtest mantığı
├── forecast.ipynb         # Orijinal keşif notebook'u (referans)
├── requirements.txt
├── experiments.csv        # Hiperparametre deneyleri (tuning sonrası)
└── scripts/run_backtest.py
```

## Model

- **Algoritma:** Facebook Prophet (`prophet` paketi)
- **Regresörler:** Volume, 7 günlük hareketli ortalama, RSI
- **Değerlendirme:** `cross_validation` + naif baz çizgisi (dünkü kapanış = yarın)
- **Varsayılan hiperparametreler:** `changepoint_prior_scale=0.05`, `seasonality_prior_scale=10.0`, `seasonality_mode=multiplicative`

Günlük hisse tahmini doğası gereği zordur; model çoğu zaman naif baz çizgisinden anlamlı şekilde daha iyi olmayabilir.

### Hiperparametre tuning

```bash
python scripts/run_backtest.py SPCX
```

Bu komut 16 noktalık grid search çalıştırır, sonuçları `experiments.csv` dosyasına yazar ve en iyi MAPE'yi raporlar.

## Notlar

- Yahoo Finance sık istek atıldığında geçici rate limit uygular. Uygulama 3 denemeli retry içerir; sorun devam ederse birkaç dakika bekleyin.
- Deploy için repoyu GitHub'a push edin, ardından [Streamlit Community Cloud](https://share.streamlit.io) üzerinden `app.py` dosyasını seçerek yayınlayın.
