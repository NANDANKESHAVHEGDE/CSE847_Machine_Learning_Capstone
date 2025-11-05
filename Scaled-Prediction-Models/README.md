# Adaptive Log Detrending Model - Production

## Overview

Production-ready implementation of the Adaptive Log Detrending approach for hotel price prediction using XGBoost. This system automatically tests multiple configurations per hotel and selects the optimal settings.

### Key Features
- **Adaptive Configuration Selection**: Tests 24 configurations per hotel (4 windows × 3 train sizes × 2 feature sets)
- **Log Detrending**: Removes trend from price series for improved prediction accuracy
- **Matrix Imputation**: Uses Soft-Impute or IterativeImputer for missing data handling
- **Competitor Price Features**: Leverages competitor pricing data with log-detrending
- **Temporal Encoding**: Cyclical encodings for day/week/year patterns
- **Production-Ready**: Modular, logged, configurable, and reproducible

### Actual Performance (From Your Notebooks)
- **Mean Test R²**: 0.4659
- **Median Test R²**: Higher than baseline
- **Improvement**: 303% over baseline (0.1156 → 0.4659)
- **Success Rate**: 95.5% (21/22 hotels)

---

## Architecture

```
Scaled-Prediction-Models/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_fetching.py         # Database operations (PostgreSQL)
│   ├── matrix_imputation.py     # Soft-Impute / IterativeImputer
│   ├── feature_engineering.py   # Lagged features + temporal encoding
│   ├── model_training.py        # Adaptive XGBoost training
│   └── utils.py                 # Logging and helper functions
├── config.yaml                  # Configuration file
├── main.py                      # Main orchestration script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (DB password)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database with hotel pricing data
- pip package manager

### Setup Steps

1. **Clone or navigate to the repository**
```bash
cd ~/OneDrive/Documents/Open_Source_projects/PricingService.ai/Dynamic-Pricing-Model-Experiments/Scaled-Prediction-Models
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Optional: Install fancyimpute for Soft-Impute method**
```bash
pip install fancyimpute
```
*If not installed, system will automatically fall back to sklearn's IterativeImputer*

5. **Create .env file for database password**
```bash
echo "DB_PASSWORD=your_actual_password" > .env
```

6. **Verify config.yaml has correct database settings**
```bash
nano config.yaml
```

Update the database section:
```yaml
database:
  host: "pricingai-dev-db.cluster-cvew44igkrv2.us-west-2.rds.amazonaws.com"
  port: 5432
  database: "pricing-service-dev-db"
  user: "nandan"
  password: "YOUR_DB_PASSWORD"  # Will be loaded from .env
```

---

## Configuration

### Database Configuration
```yaml
database:
  host: "your-db-host.com"
  port: 5432
  database: "your-database-name"
  user: "your-username"
  password: "YOUR_DB_PASSWORD"  # Set via .env file
```

### Model Configuration (Actual Values from Notebooks)
```yaml
model:
  max_depth: 4
  n_estimators: 100  # NOT 200!
  learning_rate: 0.05
  reg_alpha: 0.3
  reg_lambda: 2.0
  min_child_weight: 5  # Critical parameter
  gamma: 0.1           # Critical parameter
```

### Adaptive Configuration Testing
```yaml
adaptive:
  trend_windows: [7, 14, 21, 28]      # Detrending window sizes
  min_train_sizes: [180, 200, 250]    # Minimum training days
  feature_sets: ["simple", "extended"] # Simple (lags 1-5) vs Extended (all lags)
  test_window: 30                     # Test set size for CV
```

---

## Usage

### Run the Complete Pipeline
```bash
python main.py
```

### Pipeline Steps

The pipeline executes in 7 steps:

1. **Load Configuration** - Reads config.yaml and environment variables
2. **Fetch Data** - Retrieves focal and competitor hotel data from database
3. **Matrix Imputation** - Fills missing prices using Soft-Impute/IterativeImputer
4. **Feature Engineering** - Creates lagged features and temporal encodings
5. **Adaptive Training** - Tests 24 configurations per hotel, selects best
6. **Results Summary** - Displays performance statistics
7. **Baseline Comparison** - Compares to your baseline results

### Output Files

After running, you'll find:

```
../data/full-data/
├── hotel_mapping.csv                    # Hotel ID mapping
├── raw/
│   ├── Hotel_01_focal.csv              # Focal hotel prices
│   └── Hotel_01_competitors.csv        # Competitor prices
├── processed/
│   ├── Hotel_01_focal_matrix_completion.csv
│   ├── Hotel_01_competitors_matrix_completion.csv
│   ├── Hotel_01_competitor_price_matrix.csv
│   ├── Hotel_01_lagged_dataset.csv     # Ready for training
│   ├── Hotel_01_lag_metadata.json
│   ├── matrix_completion_summary.csv
│   └── lagged_features_summary.csv

../models/adaptive_log_method/
├── Hotel_01_model.pkl                  # Trained model
├── Hotel_02_model.pkl
├── ...
├── adaptive_summary.json               # All results
└── deployment_adaptive.csv             # Deployment table

logs/
└── adaptive_model_YYYYMMDD_HHMMSS.log # Execution log
```

---

## Model Details

### Adaptive Configuration Testing

For **each hotel**, the model tests **24 configurations**:

| Component | Options |
|-----------|---------|
| Detrending Windows | 7, 14, 21, 28 days |
| Min Training Size | 180, 200, 250 days |
| Feature Sets | Simple (lags 1-5), Extended (all lags + aggregates) |

The configuration with the highest **test R²** (from time-series cross-validation) is selected.

### XGBoost Hyperparameters

These parameters are **fixed across all hotels**:

```python
max_depth = 4
n_estimators = 100
learning_rate = 0.05
reg_alpha = 0.3
reg_lambda = 2.0
min_child_weight = 5
gamma = 0.1
random_state = 42
```

### Feature Engineering

**Simple Feature Set** (lags 1-5):
- Competitor price lags 1-5 (log-detrended)
- Temporal features: day_of_week, month, is_weekend, day_of_year
- Cyclical encodings: sin/cos for day_of_week, month, day_of_year

**Extended Feature Set** (all lags):
- All available competitor price lags (log-detrended)
- Market aggregates: comp_mean_log, comp_min_log, comp_max_log, comp_std_log
- All temporal features from Simple set

---

## Making Predictions

To use trained models for prediction:

```python
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
with open('../models/adaptive_log_method/Hotel_01_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']
trend_window = model_data['trend_window']
y_trend_last = model_data['y_trend_last']

# Prepare your features (must match training features)
# X_new = ... (your new data with same features)

# Scale features
X_scaled = scaler.transform(X_new[feature_cols])

# Predict detrended value
y_pred_detrended = model.predict(X_scaled)

# Add back trend (you'll need to calculate current trend)
y_pred_log = y_pred_detrended + current_trend
y_pred_price = np.exp(y_pred_log)
```

---

## Database Schema

### Expected Tables

**hotels** (focal hotels):
- `global_id`: Hotel identifier
- `status`: Hotel status (filter out 'inactive')

**pricing_data** (focal prices):
- `global_id`: Hotel identifier
- `stay_date`: Date
- `price`: Price value

**competitor_pricing** (competitor prices):
- `focal_hotel_id`: Focal hotel identifier
- `competitor_hotel_id`: Competitor identifier
- `stay_date`: Date
- `price`: Price value

*Adjust table/column names in `src/data_fetching.py` to match your actual schema.*

---

## Troubleshooting

### Common Issues

**1. Database Connection Fails**
```
ERROR: Failed to connect to database
```
- Check credentials in `config.yaml`
- Verify `DB_PASSWORD` in `.env` file
- Test network connectivity to database host
- Check database firewall rules

**2. ImportError: No module named 'fancyimpute'**
```
WARNING: fancyimpute not available, falling back to IterativeImputer
```
- This is expected if fancyimpute is not installed
- System will automatically use sklearn's IterativeImputer
- For better results, install: `pip install fancyimpute`

**3. Hotel Has Insufficient Data**
```
Hotel_XX: status = 'insufficient_data'
```
- Hotel needs at least 210 days of data (180 train + 30 test)
- Check if hotel has enough historical data in database
- Adjust `min_train_sizes` in config.yaml if needed

**4. All Configurations Failed**
```
Hotel_XX: status = 'all_configs_failed'
```
- Check if competitor price data exists for this hotel
- Verify lag features were created successfully
- Review logs for detailed error messages

**5. Import Errors**
```
ModuleNotFoundError: No module named 'xgboost'
```
- Install all dependencies: `pip install -r requirements.txt`
- Verify Python version >= 3.8

### Logs

All execution details are logged to `logs/` directory:
```bash
# View latest log
tail -f logs/adaptive_model_*.log

# Search for errors
grep ERROR logs/adaptive_model_*.log

# Search for specific hotel
grep "Hotel_01" logs/adaptive_model_*.log
```

---

## Performance Metrics

The model tracks these metrics (all on **original price scale**):

- **R² Score**: Variance explained (higher is better, 0-1 range)
- **RMSE**: Root Mean Squared Error (lower is better, in price units)
- **MAE**: Mean Absolute Error (lower is better, in price units)
- **MAPE**: Mean Absolute Percentage Error (lower is better, %)

Metrics are calculated using time-series cross-validation to avoid data leakage.

---

## Comparison to Baseline

### Your Baseline (from notebooks)
- Method: Non-parametric method
- Mean R²: 0.1156
- Median R²: 0.2236
- Hotels: 22

### Adaptive Log Detrending
- Method: Adaptive XGBoost with log-detrending
- Mean R²: 0.4659
- Improvement: **+303%**
- Hotels: 21

---

## Technical Notes

### Why Adaptive?

Different hotels have different characteristics:
- **Trend stability**: Some hotels have more stable trends (longer windows work better)
- **Data volume**: Some hotels have more historical data (can use larger training sets)
- **Feature complexity**: Some hotels benefit from extended features, others prefer simple

**Solution**: Test all configurations and let each hotel select its optimal settings.

### Log Detrending Process

1. Take log of prices: `log_price = log(price)`
2. Calculate rolling mean: `trend = rolling_mean(log_price, window)`
3. Detrend: `detrended = log_price - trend`
4. Train model on detrended values
5. Predict detrended value
6. Add back trend: `pred_log = pred_detrended + trend`
7. Exp to get price: `pred_price = exp(pred_log)`

### Matrix Imputation Methods

**Soft-Impute** (preferred):
- Uses low-rank matrix approximation
- Better preserves correlation structure
- Requires `fancyimpute` package

**IterativeImputer** (fallback):
- Uses BayesianRidge regression
- Available in sklearn (no extra install)
- Slightly less accurate but still effective

---

## Development

### Adding New Features

To add new features, edit `src/feature_engineering.py`:

```python
def create_lagged_features(self, ...):
    # Add your custom features here
    df_with_temporal['your_feature'] = ...
```

### Modifying Configurations

To test different hyperparameters, edit `config.yaml`:

```yaml
adaptive:
  trend_windows: [7, 14, 21, 28, 35]  # Add 35-day window
  min_train_sizes: [150, 180, 200]    # Add 150-day option
```

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review this README troubleshooting section
3. Contact the development team

---

## Acknowledgments

Based on extensive experimentation documented in:
- `Matrix-Imputations-Scaled.ipynb`
- `Lagged_Features_Creation-Matrix-Imputation.ipynb`
- `Adaptive_log_training.ipynb`
- `Adaptive_log_method.ipynb`