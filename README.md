# Causal Inference and Predictive Modeling for Competitive Hotel Pricing

**CSE 847: Machine Learning - Final Project**

Nandan Keshav Hegde & Adithya Hassan Hemakantharaju  
Michigan State University

---

## Project Overview

This project implements two complementary approaches to hotel pricing analysis:

1. **Causal Inference Track**: Two-Stage Residual Inclusion (2SRI) methodology for understanding competitive pricing dynamics and identifying causal effects
2. **Predictive Modeling Track**: Linear regression models with adaptive log-detrending optimized for out-of-sample forecasting accuracy

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
| Endogeneity Detected | 40% of competitors |
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
├── Causal-Inference/                  # 2SRI causal models
│   ├── 04_LinearModels-Stage1-causal-Iteration1.ipynb
│   ├── 05_LinearModels_Stage2_causal-Iteration1.ipynb
│   └── results/
│       ├── stage1_linear_results/     # Stage 1 outputs (F-stats, residuals)
│       └── stage2_2sri_results/       # Stage 2 outputs (coefficients, predictions)
│
├── Predictive-Models/                 # Forecasting models
│   ├── scripts/
│   │   ├── 01_Lagged-Dataset-creation/      # Time series feature engineering
│   │   ├── 02_Parametric-LR-Models/         # Linear regression variants
│   │   └── 03_Non-Parametric-Models/        # Advanced ML models (future)
│   └── results/
│
├── Utility/                           # Data preprocessing pipelines
│   ├── Dataexplorations/
│   │   └── 01_Data_Exploration.ipynb
│   ├── Basic-Dataprep-imputations/
│   └── Advanced-Dataprep-imputations/
│
├── Diagnostics/                       # Data quality analysis
│   ├── Missing_Value_pattern_recognitions.ipynb
│   └── adhoc_diagnostics.ipynb
│
├── data/
│   ├── dataraw/                       # Original datasets (not included)
│   ├── dataprocessed/                 # Cleaned and feature-engineered data
│   └── full-data/                     # Portfolio-wide datasets
│
├── docs/                              # Technical documentation & reports
│   ├── CSE847_Final_Report.pdf
│   └── CSE847_Presentation.pdf
│
└── requirements.txt                   # Python dependencies
```

---

## Data Overview

### Principal Focal Hotel (Causal Analysis)
- **Observations**: 365 daily base rates
- **Time Period**: September 2025 – September 2026
- **Price Range**: $209 – $379
- **Competitors**: 5 hotels with complete coverage
- **Data Quality**: 100% complete after preprocessing

### Full Portfolio (Predictive Modeling)
- **Hotels**: 44 total, 34 successfully modeled
- **Average Time Series**: 507 days per hotel
- **Average Competitors**: 7.1 per hotel
- **Missing Data**: ~20% (handled via IterativeImputer)

### Data Preprocessing Pipeline

1. **Missing Values**: Forward-fill + group median imputation
2. **Outliers**: IQR-based filtering (3.0 multiplier)
3. **Base Rates**: Daily minimum across room types
4. **Normalization**: MAD-based cross-hotel scaling
5. **Alignment**: Complete temporal overlap ensured

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

**Time-Series Cross-Validation**:
- Minimum training window: 300 days
- Test window: 50 days (non-overlapping)
- Expanding window approach

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

### Feature Importance (Portfolio-Wide)

| Feature | Hotels with p<0.05 | Typical Effect |
|---------|-------------------|----------------|
| Competitor lag-1 | 94% | Primary driver |
| Saturday | 85% | +$8–25 |
| Friday | 79% | +$5–18 |
| Competitor lag-2 | 82% | Secondary |
| December | 71% | +$6–34 |
| Summer | 65% | +$15–80 |

---

## Methodology Comparison

| Aspect | 2SRI (Causal) | Linear Reg (Predictive) |
|--------|---------------|------------------------|
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
cd Causal-Inference/

# Run notebooks in order:
# 1. 04_LinearModels-Stage1-causal-Iteration1.ipynb  (Stage 1: instruments)
# 2. 05_LinearModels_Stage2_causal-Iteration1.ipynb  (Stage 2: causal effects)
```

### Predictive Models

```bash
cd Predictive-Models/scripts/

# Step 1: Create lagged dataset
cd 01_Lagged-Dataset-creation/
# Run: lagged_data_preparation.ipynb

# Step 2: Run linear models
cd ../02_Parametric-LR-Models/
# Run: In-Sample-LR.ipynb
# Run: Out-of-Sample-LR.ipynb (with detrending)
```

---

## Dependencies

Key packages (see `requirements.txt` for full list):

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Limitations

- No demand-side data (occupancy, booking pace, local events)
- Exclusion restriction is economically plausible but untestable
- Competitor sets are fixed; real markets are more dynamic
- Three hotels failed detrending due to insufficient data for 30-day rolling averages

---

## Future Work

- [ ] Extend 2SRI causal analysis across full portfolio
- [ ] Incorporate demand-side features (occupancy, events)
- [ ] Non-linear Stage 1 models (Random Forest, XGBoost)
- [ ] Dynamic competitor selection
- [ ] Real-time pricing system integration

---

## References

1. Terza, J. V., Basu, A., & Rathouz, P. J. (2008). Two-stage residual inclusion estimation. *Journal of Health Economics*, 27(3), 531-543.

2. Staiger, D. & Stock, J. H. (1997). Instrumental variables regression with weak instruments. *Econometrica*, 65(3), 557-586.

3. Athey, S. & Imbens, G. W. (2019). Machine learning methods that economists should know about. *Annual Review of Economics*, 11, 685-725.

4. Bergmeir, C. & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

---

## License

This project was developed for CSE 847: Machine Learning at Michigan State University (Fall 2025).

---

## Contact

- Nandan Keshav Hegde: hegdenan@msu.edu
- Adithya Hassan Hemakantharaju: hassanad@msu.edu