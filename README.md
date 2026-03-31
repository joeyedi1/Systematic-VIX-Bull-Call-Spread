# VIX Systematic Bull Call Spread Strategy

## Architecture

```
vix-strategy/
├── config/
│   └── settings.py          # Bloomberg tickers, thresholds, regime params
├── data/
│   ├── bloomberg_fetcher.py  # Historical data pipeline (BDH via blpapi)
│   └── cot_fetcher.py        # CFTC COT positioning data
├── features/
│   └── indicators.py         # Term structure, VRP, VVIX, momentum signals
├── regime/
│   └── hmm_classifier.py     # 3-state Gaussian HMM regime detection
├── signals/
│   └── composite_score.py    # Weighted entry/exit signal generation
├── strikes/
│   └── selector.py           # Delta-based dynamic strike selection
├── backtest/
│   └── engine.py             # Walk-forward backtest with VRO settlement
├── utils/
│   └── helpers.py            # Logging, date math, common functions
├── notebooks/                # Jupyter analysis notebooks
├── outputs/                  # Backtest results, charts, reports
├── main.py                   # Orchestrator: fetch → features → regime → signals → backtest
└── requirements.txt
```

## Pipeline Flow

```
Bloomberg Data → Feature Engineering → Regime Classification (HMM)
                                              ↓
                                    Signal Composite Score
                                              ↓
                                    Entry? → Strike Selection (delta-based)
                                              ↓
                                    Position Management → Exit Rules
                                              ↓
                                    Backtest Engine (walk-forward, VRO-adjusted)
```

## Data Sources
- **Bloomberg Terminal**: VIX spot, VIX futures (UX1-UX9), VVIX, VIX3M, VIX options
- **CFTC**: Commitments of Traders (weekly, free CSV download)

## Backtest Period
- 2022-01-01 to 2026-03-18
- Covers: 2022 rate hike vol, 2023 low vol, 2024 election, 2025 H2 low vol, 2026 tariff/geo vol
