# Causal Inference and Predictive Modeling for Competitive Hotel Pricing

**CSE 847: Machine Learning - Course Project**

Nandan Keshav Hegde & Adithya Hassan Hemakantharaju  
Michigan State University

---

## Project Overview

This project implements two complementary approaches to hotel pricing analysis:

1. **Causal Inference Track**: Two-Stage Residual Inclusion (2SRI) methodology for understanding competitive pricing dynamics and identifying causal effects
2. **Predictive Modeling Track**: Linear regression models with adaptive log-detrending for out-of-sample forecasting

### Research Questions

1. **Causal**: If a competitor raises their price by $1, how much should our hotel adjust in response?
2. **Predictive**: Given competitor prices and market conditions, what will our hotel's price be tomorrow?

---

## Key Results

### Causal Analysis (Principal Focal Hotel)

| Metric | Value |
|--------|-------|
| Model R² | 0.512 |
| RMSE | $23.91 |
| Endogeneity Detected | 40% of competitors (2/5) |
| Total Competitive Effect | +1.32 (complementary pricing) |
| Price Leader | Competitor 4 (β = 1.758**) |

### Predictive Modeling (34-Hotel Portfolio)

| Metric | In-Sample | Out-of-Sample (Detrended) |
|--------|-----------|---------------------------|
| Mean Adj R² | 0.603 | — |
| Median CV R² | — | +0.25 |
| Improvement | — | 147% vs direct modeling |
| Strong Performers (R² > 0.4) | — | 7 hotels |

---

## Project Structure

```
CSE847_Machine_Learning_Capstone/
│
├── Causal-Inference/                    # 2SRI causal analysis
│   ├── scripts/
│   │   ├── 04_LinearModels-Stage1-causal-Iteration1.ipynb   # Stage 1: Instrument validation
│   │   └── 05_LinearModels_Stage2_causal-Iteration1.ipynb   # Stage 2: Causal estimation
│   └── results/                         # Stage 1 & Stage 2 outputs
│
├── Predictive-Models/                   # In-sample linear regression
│   ├── scripts/
│   │   ├── 01_Lagged-Datset-creation/   # Time series feature engineering
│   │   └── 02_Parametric-LR-Models/     # Linear regression iterations
│   │       ├── In-Sample-LR-Iteration1.ipynb
│   │       ├── In-Sample-LR-Iteration2.ipynb
│   │       └── In-Sample-LR-Iteration3.ipynb
│   └── results/
│
├── Scaled-Prediction-Models/            # Portfolio-wide out-of-sample validation
│   ├── LR/                              # Out-of-sample linear regression
│   │   ├── OutofSample-Iteration1-LR-Matrix-Imputations.ipynb
│   │   ├── OutofSample-Iteration2-LR-Matrix-Imputations.ipynb
│   │   └── OutofSample-Iteration3-LR-Matrix-Imputations.ipynb
│   └── Imputation_quality_check.ipynb   # Matrix completion validation
│
├── Utility/                             # Data preprocessing pipelines
│   ├── Dataexplorations/                # EDA notebooks
│   ├── Basic-Dataprep-imputations/      # Basic imputation methods
│   ├── Advanced-Dataprep-imputations/   # Matrix completion (IterativeImputer)
│   └── results/
│
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

---

## Methodology

### Track 1: Two-Stage Residual Inclusion (2SRI)

**Stage 1: Competitor Price Decomposition**

For each competitor c, decompose prices into predictable and simultaneous components:

```
P_c,t = α_c + Σ γ_c,j × Z_j,t + ε_c,t
```

Where Z includes 7 temporal instruments:
- sin/cos(2π × day/7) — day-of-week cycles
- sin/cos(2π × month/12) — monthly seasonality  
- sin/cos(2π × week/52) — weekly patterns
- Holiday indicator

**Stage 1 Results**:

| Competitor | R² | F-Statistic | Status |
|------------|-----|-------------|--------|
| Competitor 1 | 0.306 | 22.5 | Strong |
| Competitor 2 | 0.214 | 13.9 | Strong |
| Competitor 3 | 0.391 | 32.8 | Strong |
| Competitor 4 | 0.254 | 17.4 | Strong |
| Competitor 5 | 0.486 | 48.2 | Strong |

All F-statistics exceed the weak instrument threshold (F > 10).

**Stage 2: Causal Estimation with Residual Inclusion**

```
P_focal,t = α + Σ β_c × P_c,t + Σ θ_c × ε̂_c,t + Σ γ_j × Z_j,t + u_t
```

- β_c = causal competitive effects (target estimates)
- θ_c = endogeneity correction terms (significant θ confirms bias was present)

**Stage 2 Results**:

| Competitor | β (Causal Effect) | θ (Endogeneity) | Strategy |
|------------|-------------------|-----------------|----------|
| Competitor 1 | +0.563 | -0.597 | Complementary |
| Competitor 2 | -0.862 | +1.144* | Competitive |
| Competitor 3 | +0.105 | +0.202 | Neutral |
| Competitor 4 | +1.758** | -1.659** | Strong Complementary |
| Competitor 5 | -0.241 | +0.330 | Competitive |

*p<0.05, **p<0.01

---

### Track 2: Predictive Modeling with Adaptive Log-Detrending

**Iteration 1 (Direct Modeling)**:
- Standard regression on price levels
- Features: Competitor lags (1-5), temporal indicators
- Problem: Severe overfitting (Train R² = 0.69, CV R² = -0.53)

**Iteration 2 (Detrended Modeling)**:
1. Calculate 30-day rolling average trend
2. Model deviations: `dev_t = (P_t - trend_t) / trend_t`
3. Reconstruct: `P̂_t = trend_t × (1 + dev̂_t)`

**Result**: Median CV R² improved from -0.53 to +0.25 (147% improvement)

---

## Data Overview

### Principal Focal Hotel (Causal Analysis)
- **Observations**: 365 daily base rates
- **Time Period**: September 2025 – September 2026
- **Price Range**: $209 – $379
- **Competitors**: 5 hotels (masked as Competitor 1-5)
- **Data Quality**: 100% complete after preprocessing

### Full Portfolio (Predictive Modeling)
- **Hotels**: 44 total, 34 successfully modeled (masked as Hotel_01 to Hotel_44)
- **Average Time Series**: 507 days per hotel
- **Average Competitors**: 7.1 per hotel
- **Missing Data**: ~20% (handled via IterativeImputer matrix completion)

### Data Preprocessing Pipeline

1. **Missing Values**: Forward-fill + group median imputation
2. **Outliers**: IQR-based filtering (3.0 multiplier)
3. **Base Rates**: Daily minimum across room types
4. **Normalization**: MAD-based cross-hotel scaling
5. **Alignment**: Complete temporal overlap ensured

---

## Key Findings

### Causal Insights
1. **Endogeneity is real**: 40% of competitor relationships show significant simultaneity bias
2. **Complementary pricing dominates**: Total effect = +1.32 (hotels move prices together)
3. **Clear price leadership**: Competitor 4 drives market coordination (β = 1.758**)

### Predictive Insights
1. **Competitor lag-1 is primary driver**: Significant in 94% of hotels
2. **Detrending is essential**: Models that fit levels fail out-of-sample
3. **Hotel heterogeneity matters**: R² ranges from 0.18 to 0.89

---

## Methodology Comparison

| Aspect | 2SRI (Causal) | Linear Regression (Predictive) |
|--------|---------------|--------------------------------|
| **Goal** | Unbiased coefficients | Forecast accuracy |
| **Sample** | Full dataset | Train/test splits |
| **Endogeneity** | Corrected via instruments | Ignored |
| **R²** | 0.512 | 0.603 (in-sample) |
| **Key Finding** | 40% bias detected | Detrending essential |
| **Best Use** | Strategic decisions | Operational pricing |

---

## Running the Code

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Causal Models (2SRI)

```bash
cd Causal-Inference/scripts/

# Run notebooks in order:
# 1. 04_LinearModels-Stage1-causal-Iteration1.ipynb  (Stage 1: instruments)
# 2. 05_LinearModels_Stage2_causal-Iteration1.ipynb  (Stage 2: causal effects)
```

### Predictive Models (In-Sample)

```bash
cd Predictive-Models/scripts/02_Parametric-LR-Models/

# Run in-sample iterations:
# In-Sample-LR-Iteration1.ipynb
# In-Sample-LR-Iteration2.ipynb
# In-Sample-LR-Iteration3.ipynb
```

### Portfolio Out-of-Sample Validation

```bash
cd Scaled-Prediction-Models/LR/

# Run out-of-sample iterations:
# OutofSample-Iteration1-LR-Matrix-Imputations.ipynb
# OutofSample-Iteration2-LR-Matrix-Imputations.ipynb (with detrending)
# OutofSample-Iteration3-LR-Matrix-Imputations.ipynb
```

---

## Limitations

- No demand-side data (occupancy, booking pace, local events)
- Exclusion restriction is economically plausible but untestable
- Competitor sets are fixed; real markets are more dynamic
- Three hotels failed detrending due to insufficient data

---

## References

1. Terza, J. V., Basu, A., & Rathouz, P. J. (2008). Two-stage residual inclusion estimation. *Journal of Health Economics*, 27(3), 531-543.

2. Staiger, D. & Stock, J. H. (1997). Instrumental variables regression with weak instruments. *Econometrica*, 65(3), 557-586.