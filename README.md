# Dynamic Pricing Model Experiments

Advanced hotel pricing models combining **causal inference** (2SRI) and **predictive analytics** for competitive pricing optimization.

## Project Overview

This project implements two complementary approaches to hotel pricing:
1. **Causal Inference Track**: Two-Stage Residual Insertion (2SRI) methodology following Professor Dan Zhang's framework for understanding competitive pricing dynamics
2. **Predictive Modeling Track**: LASSO-regularized regression models optimized for forecasting accuracy

## Current Status

**Data**: 365 days of pricing data across 1 focal hotel (14 room types) and 5 competitor hotels

**Best Model Performance** (Predictive Track - Iteration 3):
- Adjusted R² = 0.607
- MAPE = 6.30%
- RMSE = $21.23 (~8% of average base rate)
- 52% of predictions within 5% error
- 76% of predictions within 10% error

**Causal Models** (2SRI Track):
- Stage 1: All 5 competitors successfully modeled (F-statistics: 13.9-48.2)
- Stage 2: Focal hotel response with endogeneity controls

## Project Structure

```
├── Causal-Inference/                  # 2SRI causal models
│   ├── 04_LinearModels-Stage1-causal-Iteration1.ipynb
│   └── 05_LinearModels_Stage2_causal-Iteration1.ipynb
├── Predictive-Models/                 # Forecasting models
│   ├── scripts/
│   │   ├── 01_Lagged-Dataset-creation/      # Time series feature engineering
│   │   ├── 02_Parametric-LR-Models/         # Linear regression variants
│   │   └── 03_Non-Parametric-Models/        # Advanced ML models
│   └── results/
├── Utility/                           # Data preprocessing pipelines
│   ├── Dataexplorations/
│   │   └── 01_Data_Exploration.ipynb
│   ├── Basic-Dataprep-imputations/
│   └── Advanced-Dataprep-imputations/
├── Diagnostics/                       # Data quality analysis
│   ├── Missing_Value_pattern_recognitions.ipynb
│   └── adhoc_diagnostics.ipynb
├── data/
│   ├── dataraw/                       # Original datasets
│   ├── dataprocessed/                 # Cleaned and feature-engineered data
│   ├── stage1_linear_results/         # 2SRI Stage 1 outputs
│   └── stage2_2sri_results/           # 2SRI Stage 2 outputs
├── docs/                              # Technical documentation
└── requirements.txt                   # Python dependencies
```

## Data Overview

### Focal Hotel
- **Observations**: 5,110 across 14 room types
- **Time Period**: 365 days (full year)
- **Price Range**: $219 - $999
- **Data Quality**: 100% complete, no missing values

### Competitor Hotels (5 properties)
- **Observations**: 1,820 total
- **Hotels**:
  - Aqua Pacific Monarch
  - Castle Kamaole Sands
  - Courtyard by Marriott Maui Kahului Airport
  - Kohea Kai Resort Maui
  - Ohana Waikiki Malia
- **Missing Rate**: <5% (manageable with imputation)
- **Price Range**: $207-$904

### Data Readiness Score: 100%
- ✓ Sufficient temporal overlap (364 days)
- ✓ Clean price data
- ✓ Meaningful price correlations (0.28-0.53)
- ✓ Multiple competitors
- ✓ Strong instruments available

---

## Track 1: Causal Inference (2SRI Methodology)

Following Professor Dan Zhang's Two-Stage Residual Insertion framework for identifying causal pricing effects.

### Stage 1: Competitor Price Decomposition
**Objective**: Isolate exogenous vs endogenous components of competitor pricing

**Implementation**:
- Model each competitor's price using temporal instruments (day-of-week, seasonal patterns)
- Extract residuals as proxies for unobserved demand shocks
- Linear regression approach with 7 temporal features

**Results**:
| Competitor | R² | F-Statistic | Instrument Strength |
|------------|-----|-------------|-------------------|
| Aqua Pacific Monarch | 0.306 | 22.5 | Strong |
| Castle Kamaole Sands | 0.214 | 13.9 | Strong |
| Courtyard Marriott | 0.391 | 32.8 | Strong |
| Kohea Kai Resort | 0.254 | 17.4 | Strong |
| Ohana Waikiki Malia | 0.486 | 48.2 | Strong |

**Key Finding**: All instruments exceed F>10 threshold, validating 2SRI approach.

### Stage 2: Focal Hotel Response Model
**Objective**: Estimate causal effect of competitor prices on focal hotel pricing

**Implementation**:
- Dependent variable: Focal hotel base rate
- Endogenous regressors: Competitor prices
- Instruments: Stage 1 residuals
- Controls: Temporal features

**Status**: Implemented with endogeneity controls via residual insertion

---

## Track 2: Predictive Models

Time series forecasting models optimized for accuracy in pricing predictions.

### Data Preprocessing

**Imputation Strategies Tested**:
1. Basic: Forward-fill + group median fallback
2. KNN imputation
3. MICE (Multiple Imputation by Chained Equations)
4. Time-series specific methods

**Feature Engineering**:
- Lagged competitor prices (1-7 days)
- Temporal features:
  - Cyclical: sin/cos transformations for weeks
  - Polynomial: week centered, squared, cubic, quartic
- Seasonal indicators:
  - Holiday periods (December, January, July 4th, Thanksgiving)
  - Peak season (Summer + Winter holidays)
  - Summer indicator

### Model Evolution

**Iteration 1**: Baseline linear regression
**Iteration 2**: Ridge/ElasticNet regularization
**Iteration 3**: LASSO feature selection + OLS refit (Current Best)

### Best Model Specification (Iteration 3)

**Feature Selection**: LassoCV with 5-fold cross-validation
- Input features: 22-25 candidates
- Selected features: 11-13 (depending on imputation method)
- Optimal alpha: 1.25 (advanced imputation)

**Final Equation** (Advanced Imputation):
```
base_rate = 97.58
           + 26.38 × cos_week ***
           - 0.62 × week_centered ***
           + 22.48 × is_peak_season ***
           - 2.40 × is_holiday
           + 0.35 × Courtyard_lag_1 ***
           - 0.26 × Ohana_lag_3 ***
           + 0.10 × Castle_lag_1 ***
           + 0.02 × Kohea_lag_5 ***
           + [3 additional competitor lags]
```

**Performance Metrics**:
- **Adjusted R²**: 0.607
- **RMSE**: $21.23 (7.94% of average base rate)
- **MAPE**: 6.30%
- **Median APE**: 4.48%

**Error Distribution**:
- 52% of predictions within 5% error
- 76% within 10% error
- Only 3% exceed 20% error

**Key Drivers**:
1. **Courtyard Marriott lag-1**: Strongest competitor effect (+$0.35 per dollar)
2. **Peak Season**: Adds ~$22 to base rate
3. **Weekly Seasonality**: Up to $26 swing via cosine term

---

## Methodology Comparison

| Aspect | Causal (2SRI) | Predictive (LASSO) |
|--------|---------------|-------------------|
| **Primary Goal** | Understand pricing mechanisms | Forecast future prices |
| **Interpretation** | Causal effects | Correlational patterns |
| **Endogeneity** | Explicitly controlled | Not directly addressed |
| **Feature Selection** | Theory-driven | Data-driven (LASSO) |
| **Best Use** | Strategic pricing decisions | Daily operational pricing |
| **R²** | ~0.33 (Stage 1 avg) | 0.607 |
| **Complexity** | Two-stage estimation | Single-stage |

---

## Key Insights

### Competitive Dynamics
1. **Courtyard Marriott** is the dominant price leader (strongest coefficient across all models)
2. **Ohana Waikiki Malia** shows negative relationship at lag-3 (possible substitution effect)
3. Different lags matter for different competitors (1-5 day windows)

### Temporal Patterns
1. Strong **weekly seasonality** captured by cosine transformations
2. **Peak season premium** consistently ~$20-22
3. Holiday effects are present but less significant than peak season

### Model Performance
1. Predictive models achieve **<8% MAPE**, suitable for operational use
2. 2SRI successfully validates instruments (all F-stats >10)
3. Room-level correlations vary from 0.27-0.53 with competitors

---

## Implementation Guide

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

### Running Models

**Data Exploration**:
```bash
# Navigate to Utility folder
cd Utility/Dataexplorations/
# Open: 01_Data_Exploration.ipynb
```

**Causal Models (2SRI)**:
```bash
# Navigate to Causal-Inference folder
cd Causal-Inference/
# Run notebooks in order:
# 1. 04_LinearModels-Stage1-causal-Iteration1.ipynb
# 2. 05_LinearModels_Stage2_causal-Iteration1.ipynb
```

**Predictive Models**:
```bash
# Navigate to Predictive-Models
cd Predictive-Models/scripts/

# Step 1: Create lagged dataset
cd 01_Lagged-Dataset-creation/
# Run: lagged_data_preparation_iteration3.ipynb (or advanced imputation variant)

# Step 2: Run parametric models
cd ../02_Parametric-LR-Models/
# Run: In-Sample-LR-Iteration3.ipynb
```

---

## Next Steps & Future Work

### Short-term Enhancements
- [ ] Out-of-sample validation with train/test split
- [ ] Non-parametric models (Random Forest, XGBoost) comparison
- [ ] Room-type specific models for 14 segments
- [ ] Cross-validation for temporal robustness

### Medium-term Development
- [ ] Real-time pricing API integration
- [ ] Confidence intervals for price recommendations
- [ ] Multi-step ahead forecasting (7-14 days)
- [ ] Integration of external demand signals (weather, events)

### Long-term Vision
- [ ] Multi-property universal model
- [ ] Reinforcement learning for dynamic pricing
- [ ] A/B testing framework for price optimization
- [ ] Human-in-the-loop pricing adjustment interface

---

## Dependencies

See [requirements.txt](requirements.txt) for full list. Key packages:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: ML models and preprocessing
- `statsmodels`: Statistical inference and diagnostics
- `matplotlib`, `seaborn`: Visualization
- `jupyter`: Interactive notebooks
