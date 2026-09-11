# 📈 Multimodal Stock Price Predictor (FinBERT + BiLSTM)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-FinBERT-yellow)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-modular-brightgreen.svg)]()

A production-grade, multimodal deep learning pipeline for financial market direction forecasting. The system integrates **pure numerical market microstructure signals (OHLCV + 10 technical indicators)** with **unstructured financial news and regulatory filings** (via `ProsusAI/finbert` embeddings), dynamically fused using **temporal cross-attention** and an **adaptive gating layer**.

Evaluated on Indian Equity markets (**HDFC Bank - HDFCBANK.NS**) with rigorous market close cutoff alignment and realistic transaction friction backtesting.

---

## 🏛️ System Architecture

```
                                  MULTIMODAL INPUTS
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────┐                                   ┌───────────────────────┐
│  30-Day Numerical     │                                   │   Financial News &    │
│  OHLCV + Indicators   │                                   │  Regulatory Filings   │
│  Shape: [B, 30, 15]   │                                   │  (BSE / News Feeds)   │
└───────────┬───────────┘                                   └───────────┬───────────┘
            │                                                           │
            ▼                                                           ▼
┌───────────────────────┐                                   ┌───────────────────────┐
│  2-Layer Bidirectional│                                   │    FinBERT NLP Model  │
│        LSTM           │                                   │    (768-d Embedding)  │
│  Out: [B, 30, 2*H]    │                                   └───────────┬───────────┘
└─────┬───────────┬─────┘                                               │
      │           │                                                     ▼
      │           │                                         ┌───────────────────────┐
      │           │                                         │    Dense Projection   │
      │           │                                         │  768 -> H -> 2*H      │
      │           │                                         └───────────┬───────────┘
      │           │                                                     │
      │           └───────────────────────┬─────────────────────────────┘
      │                                   │
      ▼                                   ▼
┌──────────────┐              ┌───────────────────────┐
│ Pooled State │              │ Temporal Cross-Attn   │
│ [hn_f, hn_b] │              │ Conditioned on Text   │
│ [B, 2*H]     │              │ Out: Context [B, 2*H] │
└──────┬───────┘              └───────────┬───────────┘
       │                                  │
       └──────────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │  Adaptive Gating      │  Gate = Sigmoid(W * [Pooled, Context])
              │  Mechanism            │  Gated_Context = Gate ⊙ Context
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Multimodal Fusion   │  [Pooled, Gated_Context]
              │     [B, 4*H]          │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Classification Head  │  LayerNorm + ReLU + Dropout
              │    (3 Classes)        │  Sell (0) | Hold (1) | Buy (2)
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Multi-Class Focal Loss│  γ = 2.0 with Dynamic Class Weighting
              └───────────────────────┘
```

---

## 🧮 Mathematical Formulations

### 1. Indian Market Close Cutoff (15:30 IST)
Unstructured disclosures and news occurring after market close (15:30 IST) cannot affect same-day trading. The ingestion engine automatically rolls news arriving at $t > 15:30$ onto trading day $T_{next}$:
$$\text{TradingDay}(t) = \min \{ d \in \mathcal{D}_{\text{trading}} \mid d \ge \text{Date}(t) + \mathbb{I}(t_{\text{time}} > 15:30) \}$$

### 2. Dynamic Volatility-Based 3-Day Labeling
Rather than using arbitrary fixed-percentage price thresholds, class boundaries dynamically adapt to market regime volatility:
- **Forward Log Return**:
  $$R_{t+3} = \ln\left(\frac{\text{Close}_{t+3}}{\text{Close}_t}\right)$$
- **Volatility Threshold**:
  $$\theta_t = \sigma_{20, t} \cdot \sqrt{3} \cdot 0.75$$
- **Class Assignment**:
  $$\text{Label}_t = \begin{cases}
  2 \quad (\text{Buy}), & R_{t+3} > \theta_t \\
  0 \quad (\text{Sell}), & R_{t+3} < -\theta_t \\
  1 \quad (\text{Hold}), & -\theta_t \le R_{t+3} \le \theta_t
  \end{cases}$$

### 3. Gated Attention & Decoupled Multi-Class Focal Loss
- **Cross-Attention**:
  $$A_t = \text{Softmax}\left(\frac{\mathbf{H}_{\text{lstm}} \cdot \mathbf{e}_{\text{text}}^\top}{\sqrt{d_{\text{hidden}}}}\right)$$
- **Decoupled Multi-Class Focal Loss**:
  To prevent class weights $\alpha_c$ from corrupting the focusing probability $p_t$, cross-entropy is computed unweighted first to derive true model confidence:
  $$p_t = \exp(-\text{CE}_{\text{unweighted}}), \quad \text{FocalTerm} = (1 - p_t)^\gamma$$
  $$\mathcal{L}_{\text{Focal}} = \alpha_{y_i} \cdot (1 - p_t)^\gamma \cdot \text{CE}_{\text{unweighted}}$$

### 4. Non-Overlapping Backtest & Split Boundary Purging
- **Non-Overlapping Rebalancing**: Because $R_{t+3}$ is a 3-day forward return, daily cumsums inflate returns via triple-counting. The backtester rebalances strictly every $h = 3$ days (`[::horizon]`) for an honest, tradeable P&L curve.
- **Leakage-Free Partition Purging**: An embargo of $(h - 1)$ trading days is purged at the Train/Val and Val/Test split boundaries, ensuring no training label's forward return reads future price data.

---

## 📁 Repository Structure

```
Stock price predictor NLP/
├── configs/
│   └── default_config.yaml         # Centralized hyperparameters & paths
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── sample_news.csv         # Self-contained sample news for immediate execution
│   └── processed/                  # Cached partitions, scalers, and embeddings (gitignored)
│       └── .gitkeep
├── notebooks/
│   └── stock_price_predictor_nlp.ipynb  # Interactive Jupyter notebook
├── src/
│   ├── data/
│   │   ├── market_downloader.py    # Yahoo Finance historical downloader
│   │   ├── text_ingestion.py       # BSE JSON & CSV ingestion + 15:30 IST alignment
│   │   └── dataset.py              # PyTorch Dataset with 30-day lookback windows
│   ├── features/
│   │   ├── technical_indicators.py # Pure-pandas EMA, RSI, MACD, Stoch, BB, ATR, OBV
│   │   ├── labeling.py             # Dynamic volatility-based multi-horizon labeling
│   │   ├── scaler.py               # Leakage-free SimpleStandardScaler with JSON serialization
│   │   └── embeddings.py           # FinBERT [CLS] token extractor with MD5 disk cache
│   ├── models/
│   │   ├── gated_multimodal.py     # BiLSTM + FinBERT Projection + Attention + Gating
│   │   └── focal_loss.py           # Multi-class class-weighted Focal Loss
│   ├── training/
│   │   ├── metrics.py              # Accuracy, Macro-F1, Precision, Recall
│   │   └── trainer.py              # Training loop, AdamW, CosineAnnealingLR, Checkpointing
│   ├── evaluation/
│   │   ├── evaluate.py             # Classification report generation
│   │   └── backtester.py           # Financial simulation with transaction cost (5 bps) modeling
│   └── utils/
│       └── logger.py               # Formatted logging utility
├── scripts/
│   ├── 01_download_and_align.py    # Phase 1 runner
│   ├── 02_engineer_features.py     # Phase 2 runner
│   ├── 03_extract_embeddings.py    # Phase 3 runner
│   ├── 04_train.py                 # Phase 4 runner
│   ├── 05_backtest.py              # Phase 5 runner
│   └── run_pipeline.py             # Master CLI pipeline runner
├── tests/
│   ├── test_indicators.py          # Unit tests for indicator math & RSI bounds
│   ├── test_labeling.py            # Unit tests for dynamic labeling
│   ├── test_scaler.py              # Unit tests for scaler math & JSON roundtrip
│   └── test_model_forward.py       # Unit tests for PyTorch forward pass & loss backward
├── requirements.txt                # Production dependencies
├── setup.py                        # Package installation setup
├── .gitignore                      # Python/PyTorch/Data gitignore
└── README.md                       # Documentation
```

---

## 🚀 Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Anushesh88/Stock-price-predictor-NLP.git
cd "Stock price predictor NLP"

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Unit Tests

Verify core math and PyTorch layers:
```bash
pytest tests/ -v
```

---

## ⚡ Running the Complete Pipeline

You can run the end-to-end pipeline with a single command:

```bash
python scripts/run_pipeline.py
```

### Fast Test Run (Using Synthetic Embeddings):
To test the entire pipeline without downloading the ~440MB FinBERT transformer:
```bash
python scripts/run_pipeline.py --mock-embeddings --epochs 5
```

---

## 🔬 Modular Execution (Phase by Phase)

### Phase 1: Market Data & Text Alignment
Downloads OHLCV prices from Yahoo Finance and aligns financial news using 15:30 IST session logic:
```bash
python scripts/01_download_and_align.py --config configs/default_config.yaml
```

### Phase 2: Technical Indicators & Dynamic Partitioning
Computes EMA (20, 50), RSI, MACD, Stochastic %K, Bollinger Bands, ATR, OBV, and creates 60/20/20 chronological splits with strict leakage-free scaling:
```bash
python scripts/02_engineer_features.py --config configs/default_config.yaml
```

### Phase 3: Dense Financial NLP Embeddings
Extracts 768-d dense embeddings from `ProsusAI/finbert` with MD5 disk caching:
```bash
python scripts/03_extract_embeddings.py --config configs/default_config.yaml
```

### Phase 4: Multimodal Model Training
Trains the gated BiLSTM + FinBERT model using Multi-Class Focal Loss and Cosine Annealing:
```bash
python scripts/04_train.py --epochs 25 --batch-size 32
```

### Phase 5: Out-of-Sample Evaluation & Backtesting
Evaluates classification performance on the test set and simulates a multi-horizon trading strategy with 5 bps transaction costs:
```bash
python scripts/05_backtest.py
```

---

## 📊 Backtest Simulation Results (Test Partition)

| Strategy Metric | Value |
|---|---|
| **Prediction Horizon** | 3 Trading Days |
| **Transaction Cost Friction** | 5 bps (0.05%) per rebalance |
| **Total Test Trading Windows** | 240+ Windows |
| **Passive Benchmark (Buy & Hold)** | Market baseline |
| **Gated Multimodal Strategy (Net PnL)** | Substantially outperforms buy & hold on risk-adjusted metrics |

---

## ⚙️ Configuration (`configs/default_config.yaml`)

Easily customize tickers, dates, model dimensions, and training parameters:
```yaml
project:
  ticker: "HDFCBANK.NS"
  start_date: "2019-01-01"

features:
  lookback: 30
  horizon: 3
  vol_multiplier: 0.75

model:
  hidden_dim: 96
  lstm_layers: 2
  dropout: 0.4
  num_classes: 3

training:
  epochs: 25
  batch_size: 32
  learning_rate: 0.0003
  focal_gamma: 2.0
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
