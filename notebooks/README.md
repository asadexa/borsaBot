# BorsaBot Notebooks

Bu klasör Google Colab'da çalıştırılmak üzere hazırlanmış model eğitim scripti içerir.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `train_colab.py` | **Ana eğitim scripti** — bunu Colab'a yükle |
| `01_data_download.py` | Veri indirme (ayrı bölüm) |
| `02_feature_engineering.py` | Özellik mühendisliği |
| `03_labeling.py` | Triple Barrier etiketleme |
| `04_model_training.py` | Model eğitimi |
| `05_cpcv_backtest.py` | CPCV backtest & görselleştirme |

## Colab'da Nasıl Çalıştırılır?

1. [colab.research.google.com](https://colab.research.google.com) aç
2. **File → Upload notebook → Upload** → `train_colab.py` yükle  
   *(veya: File → New notebook → sol üst `+Code` ile kopyala/yapıştır)*  

3. **Runtime → Change runtime type → T4 GPU** seç (hız için)

4. Hücreleri sırayla çalıştır (`Ctrl+F9` = tümünü çalıştır)

## Veri Kaynağı

- **Binance Public REST API** — ücretsiz, kayıt gerekmez
- URL: `https://api.binance.com/api/v3/klines`
- Desteklenen semboller: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT` vs.
- Desteklenen interval: `15m`, `1h`, `4h`, `1d`
- Maksimum tarih aralığı: ~5 yıl

## Konfigürasyon (train_colab.py → Cell 2)

```python
CFG = dict(
    symbols    = ["BTCUSDT", "ETHUSDT"],  # Semboller
    interval   = "1h",                    # Zaman dilimi
    start_date = "2021-01-01",
    end_date   = "2024-12-31",
    vertical_bars = 48,   # Maks. tutma süresi (bar)
    pt_sl = [1.5, 1.0],   # Profit-take / Stop-loss
    n_groups = 6,          # CPCV grup sayısı
    n_test   = 2,          # CPCV test grup sayısı
)
```

## Pipeline Özeti

```
Binance API
    ↓
OHLCV (1h, 2021-2024)
    ↓
Feature Engineering
  - Fractional Differentiation (d=0.4)
  - RSI(14), ATR(14), OBV
  - Bollinger Band genişliği
  - VWAP sapması
  - Gecikmeli getiriler (1,2,3,6,12,24 bar)
    ↓
Triple Barrier Labeling
  PT=1.5σ  SL=1.0σ  Vertical=48bar
    ↓  
RegimeDetector (HMM, 3 durum)
    ↓
PrimaryModel (XGBoost, multi-class: -1/0/+1)
  + SMOTE sınıf dengeleme
    ↓
Meta Labeling + MetaModel (LightGBM, binary)
    ↓
CPCV Backtest (C(6,2)=15 path)
  + Sharpe dağılımı
  + PSR hesabı
    ↓
borsabot_models.zip (indir)
```

## Çıktı Dosyaları

```
/content/borsabot_data/
├── raw/
│   ├── BTCUSDT_1h.parquet
│   └── ETHUSDT_1h.parquet
├── processed/
│   ├── BTCUSDT_features.parquet
│   ├── BTCUSDT_labeled.parquet
│   └── BTCUSDT_cpcv.png
└── models/               ← borsabot_models.zip olarak indir
    ├── regime.pkl
    ├── BTCUSDT_primary.pkl
    ├── BTCUSDT_meta.pkl
    ├── ETHUSDT_primary.pkl
    └── ETHUSDT_meta.pkl
```

## Modelleri Bota Yüklemek

İndirilen ZIP'i aç, `models/` klasörüne kopyala:

```bash
# models/ klasörü oluştur
mkdir -p c:\borsaBot\models

# ZIP'i aç, içindekileri kopyala
# Ardından:
python scripts/main.py --broker mock --paper --model-dir models/
```
