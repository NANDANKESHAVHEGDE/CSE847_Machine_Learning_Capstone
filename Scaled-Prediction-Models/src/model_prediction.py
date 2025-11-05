import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import argparse
import warnings
import yaml
warnings.filterwarnings('ignore')

# Try to import fancyimpute for SoftImpute
try:
    from fancyimpute import SoftImpute
    SOFTIMPUTE_AVAILABLE = True
except ImportError:
    print("WARNING: fancyimpute not available, will use IterativeImputer")
    SOFTIMPUTE_AVAILABLE = False
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

# Load environment variables
load_dotenv()


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    # Try multiple possible locations
    possible_paths = [
        Path(config_path),                    # Current directory
        Path('config') / config_path,         # config/ subdirectory
        Path('..') / config_path,             # Parent directory
        Path('../config') / config_path,      # Parent's config/ subdirectory
    ]
    
    config_file = None
    for path in possible_paths:
        if path.exists():
            config_file = path
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Config file not found. Tried: {[str(p) for p in possible_paths]}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace DB_PASSWORD placeholder with environment variable
    if config['database']['password'] == 'YOUR_DB_PASSWORD':
        config['database']['password'] = os.getenv('DB_PASSWORD')
    
    return config


class MatrixImputer:
    """Matrix imputation for competitor prices - EXACT training implementation"""
    
    def __init__(self, method='auto', max_rank=None, max_iter=10, random_state=42):
        """
        Initialize imputer - EXACT match to training
        
        Args:
            method: 'auto', 'soft-impute', or 'iterative'
            max_rank: Max rank for SoftImpute (default: min(3, min(shape)-1))
            max_iter: Max iterations for IterativeImputer
            random_state: Random seed
        """
        self.method = method
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.random_state = random_state
        
    def impute(self, df_matrix):
        """
        Impute missing values in price matrix - EXACT training logic
        
        Args:
            df_matrix: DataFrame with dates as index, competitors as columns
            
        Returns:
            Imputed DataFrame
        """
        if df_matrix.isnull().sum().sum() == 0:
            print("No missing values - skipping imputation")
            return df_matrix
        
        missing_count = df_matrix.isnull().sum().sum()
        missing_pct = (missing_count / df_matrix.size) * 100
        print(f"Missing values: {missing_count} / {df_matrix.size} ({missing_pct:.1f}%)")
        
        # Convert to matrix
        price_matrix = df_matrix.values
        
        # EXACT TRAINING LOGIC: Try SoftImpute first, fallback to IterativeImputer
        try:
            from fancyimpute import SoftImpute
            # EXACT: max_rank = min(3, min(shape) - 1)
            max_rank = self.max_rank if self.max_rank else min(3, min(price_matrix.shape) - 1)
            imputer = SoftImpute(max_rank=max_rank, verbose=False)
            completed_matrix = imputer.fit_transform(price_matrix)
            method_used = "Soft-Impute"
            print(f"Using imputation method: Soft-Impute (max_rank={max_rank})")
        except ImportError:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.linear_model import BayesianRidge
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0
            )
            completed_matrix = imputer.fit_transform(price_matrix)
            method_used = "IterativeImputer"
            print(f"Using imputation method: IterativeImputer")
        
        # EXACT TRAINING LOGIC: Fill any remaining NaNs with column mean
        remaining_missing = np.isnan(completed_matrix).sum()
        if remaining_missing > 0:
            print(f"Filling {remaining_missing} remaining NaN values with column means")
            for i in range(completed_matrix.shape[1]):
                col_mean = np.nanmean(completed_matrix[:, i])
                if np.isnan(col_mean):
                    col_mean = np.nanmean(completed_matrix)
                completed_matrix[np.isnan(completed_matrix[:, i]), i] = col_mean
        
        # EXACT TRAINING LOGIC: Clip negative values to 0
        negatives = (completed_matrix < 0).sum()
        if negatives > 0:
            print(f"Clipping {negatives} negative values to 0")
            completed_matrix = np.maximum(completed_matrix, 0)
        
        # Convert back to DataFrame
        df_imputed = pd.DataFrame(
            completed_matrix,
            index=df_matrix.index,
            columns=df_matrix.columns
        )
        
        print(f"{method_used} imputation completed successfully")
        return df_imputed


class HotelPricePredictor:
    """On-demand hotel price prediction system with imputation"""
    
    def __init__(self, models_dir='models/adaptive_log_method', db_config=None, 
                 imputation_window_days=60):
        """
        Initialize predictor
        
        Args:
            models_dir: Directory containing trained models
            db_config: Database configuration
            imputation_window_days: Days of history to fetch for imputation
        """
        self.models_dir = Path(models_dir)
        self.db_config = db_config
        self.connection = None
        self.model_cache = {}
        self.hotel_mapping = None
        self.imputation_window_days = imputation_window_days
        
        # Initialize imputer - EXACT training parameters
        self.imputer = MatrixImputer(method='auto', max_rank=None, max_iter=10, random_state=42)
        
        # Load hotel mapping
        self._load_hotel_mapping()
        
    def _load_hotel_mapping(self):
        """Load hotel mapping from CSV"""
        # Try multiple possible locations
        possible_paths = [
            Path('data/full-data/hotel_mapping.csv'),
            Path('../data/full-data/hotel_mapping.csv'),
            Path('../../data/full-data/hotel_mapping.csv'),
        ]
        
        mapping_file = None
        for path in possible_paths:
            if path.exists():
                mapping_file = path
                break
        
        if mapping_file and mapping_file.exists():
            self.hotel_mapping = pd.read_csv(mapping_file)
            print(f"Loaded mapping for {len(self.hotel_mapping)} hotels")
        else:
            print(f"WARNING: Hotel mapping not found")
            self.hotel_mapping = None
    
    def get_masked_id(self, hotel_name):
        """
        Convert actual hotel name to masked ID
        
        Args:
            hotel_name: Actual hotel name (e.g., 'Ozarker_Lodge')
            
        Returns:
            Tuple of (masked_id, actual_id) or None
        """
        if self.hotel_mapping is None:
            print("ERROR: Hotel mapping not loaded")
            return None
        
        # Try exact match first
        match = self.hotel_mapping[self.hotel_mapping['actual_id'] == hotel_name]
        
        # Try case-insensitive match
        if len(match) == 0:
            match = self.hotel_mapping[
                self.hotel_mapping['actual_id'].str.lower() == hotel_name.lower()
            ]
        
        # Try partial match
        if len(match) == 0:
            match = self.hotel_mapping[
                self.hotel_mapping['actual_id'].str.contains(hotel_name, case=False, na=False)
            ]
        
        if len(match) == 0:
            print(f"ERROR: Hotel '{hotel_name}' not found in mapping")
            print("\nAvailable hotels:")
            for _, row in self.hotel_mapping.head(10).iterrows():
                print(f"  {row['actual_id']} -> {row['masked_id']}")
            if len(self.hotel_mapping) > 10:
                print(f"  ... and {len(self.hotel_mapping) - 10} more")
            return None
        
        if len(match) > 1:
            print(f"WARNING: Multiple matches found for '{hotel_name}':")
            for _, row in match.iterrows():
                print(f"  {row['actual_id']} -> {row['masked_id']}")
            print(f"Using first match: {match.iloc[0]['masked_id']}")
        
        masked_id = match.iloc[0]['masked_id']
        actual_id = match.iloc[0]['actual_id']
        
        print(f"Hotel: {actual_id} -> {masked_id}")
        return masked_id, actual_id
    
    def list_available_hotels(self):
        """List all available hotels"""
        if self.hotel_mapping is None:
            print("ERROR: Hotel mapping not loaded")
            return
        
        print(f"\nAvailable Hotels ({len(self.hotel_mapping)} total):")
        print("-" * 80)
        for _, row in self.hotel_mapping.iterrows():
            print(f"  {row['actual_id']:<40} -> {row['masked_id']}")
        print("-" * 80)
        
    def list_available_hotels(self):
        """List all available hotels with trained models"""
        if self.hotel_mapping is None:
            print("ERROR: Hotel mapping not loaded")
            return
        
        # Find which hotels have trained models
        trained_hotels = []
        for _, row in self.hotel_mapping.iterrows():
            model_file = self.models_dir / f"{row['masked_id']}_model.pkl"
            if model_file.exists():
                trained_hotels.append(row['actual_id'])
        
        trained_hotels.sort()
        
        print("\n" + "=" * 80)
        print(f"AVAILABLE HOTELS WITH TRAINED MODELS ({len(trained_hotels)} total)")
        print("=" * 80)
        print(f"\n{'Hotel Name':<50} {'Status'}")
        print("-" * 80)
        
        for hotel_name in trained_hotels:
            print(f"{hotel_name:<50} ✓ Trained")
        
        print("-" * 80)
        print("\nYou can use these hotel names with --hotel option")
        print('Example: --hotel "Ozarker_Lodge"')
        print("=" * 80 + "\n")
        
    def connect_db(self):
        """Connect to database"""
        if self.connection is None and self.db_config:
            try:
                self.connection = psycopg2.connect(**self.db_config)
                print("Database connection established")
            except Exception as e:
                print(f"Database connection failed: {e}")
                self.connection = None
    
    def disconnect_db(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Database connection closed")
    
    def load_model(self, hotel_id):
        """
        Load trained model for a hotel
        
        Args:
            hotel_id: Hotel identifier (e.g., 'Hotel_01')
            
        Returns:
            Dictionary with model, scaler, config, feature_cols or None
        """
        # Check cache first
        if hotel_id in self.model_cache:
            return self.model_cache[hotel_id]
        
        model_file = self.models_dir / f'{hotel_id}_model.pkl'
        
        if not model_file.exists():
            print(f"ERROR: Model not found for {hotel_id}")
            print(f"Expected location: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Cache the model
            self.model_cache[hotel_id] = model_data
            
            print(f"Loaded model for {hotel_id}")
            print(f"  Configuration: trend_window={model_data['config']['trend_window']}, "
                  f"feature_set={model_data['config']['feature_set']}")
            print(f"  Features: {len(model_data['feature_cols'])}")
            
            return model_data
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None
    
    def get_required_competitors(self, hotel_id):
        """
        Get list of competitors needed for prediction
        
        Args:
            hotel_id: Hotel identifier
            
        Returns:
            List of competitor identifiers or None
        """
        model_data = self.load_model(hotel_id)
        if model_data is None:
            return None
        
        feature_cols = model_data['feature_cols']
        
        # Extract competitor names from feature columns
        competitors = set()
        for col in feature_cols:
            if '_lag_' in col and col.endswith('_log_detrended'):
                # Remove '_lag_X_log_detrended' to get competitor name
                base_name = col.replace('_log_detrended', '')
                parts = base_name.rsplit('_lag_', 1)
                if len(parts) == 2:
                    competitors.add(parts[0])
        
        return sorted(list(competitors))
    
    def fetch_competitor_data(self, hotel_id, prediction_date, actual_hotel_id=None):
        """
        Fetch competitor price data for imputation window
        
        Args:
            hotel_id: Masked hotel ID
            prediction_date: Date to predict for
            actual_hotel_id: Actual hotel ID (optional)
            
        Returns:
            DataFrame with competitor prices or None
        """
        if self.connection is None:
            print("ERROR: Database not connected")
            return None
        
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date)
        
        # Get actual hotel ID if not provided
        if actual_hotel_id is None:
            mapping_file = Path('data/full-data/hotel_mapping.csv')
            if mapping_file.exists():
                mapping = pd.read_csv(mapping_file)
                match = mapping[mapping['masked_id'] == hotel_id]
                if len(match) > 0:
                    actual_hotel_id = match.iloc[0]['actual_id']
        
        if actual_hotel_id is None:
            print("ERROR: Cannot determine actual hotel ID")
            return None
        
        # Fetch data from (prediction_date - imputation_window_days) to prediction_date
        end_date = prediction_date.date()
        start_date = (prediction_date - timedelta(days=self.imputation_window_days)).date()
        
        print(f"Fetching competitor data from {start_date} to {end_date}")
        
        try:
            query = """
                SELECT hotel_id, stay_date, price
                FROM public.ota_rates
                WHERE hotel_id IN (
                    SELECT competitor ->> 'hotel_id' AS extracted_hotel_id
                    FROM public.hotels,
                    jsonb_array_elements(hotel_settings::jsonb -> 'competitors') AS competitor
                    WHERE global_id = %s
                )
                AND stay_date BETWEEN %s AND %s
                AND is_active = True
                ORDER BY stay_date, hotel_id
            """
            
            df = pd.read_sql_query(query, self.connection, params=(actual_hotel_id, start_date, end_date))
            
            if len(df) == 0:
                print("ERROR: No data fetched from database")
                return None
            
            print(f"Fetched {len(df)} price records from {df['hotel_id'].nunique()} competitors")
            
            # Check for required competitors
            required_competitors = self.get_required_competitors(hotel_id)
            if required_competitors:
                available_competitors = df['hotel_id'].unique()
                missing = set(required_competitors) - set(available_competitors)
                if missing:
                    print(f"WARNING: Missing {len(missing)} required competitors in fetched data")
            
            return df
            
        except Exception as e:
            print(f"ERROR fetching data: {e}")
            return None
    
    def prepare_imputation_matrix(self, df_prices):
        """
        Convert price data to matrix format for imputation
        
        Args:
            df_prices: DataFrame with columns [hotel_id, stay_date, price]
            
        Returns:
            DataFrame with dates as index, competitors as columns
        """
        # Pivot to wide format
        df_matrix = df_prices.pivot(index='stay_date', columns='hotel_id', values='price')
        df_matrix = df_matrix.sort_index()
        
        print(f"Price matrix shape: {df_matrix.shape}")
        missing_pct = (df_matrix.isnull().sum().sum() / df_matrix.size) * 100
        print(f"Missing values: {missing_pct:.1f}%")
        
        return df_matrix
    
    def create_lagged_features(self, df_imputed, prediction_date):
        """
        Create lagged features from imputed data
        
        Args:
            df_imputed: Imputed price matrix
            prediction_date: Date to predict for
            
        Returns:
            DataFrame with lagged features
        """
        # Create lags 1-5
        lagged_data = {}
        for col in df_imputed.columns:
            for lag in range(1, 6):
                lag_col = f'{col}_lag_{lag}'
                lagged_data[lag_col] = df_imputed[col].shift(lag)
        
        df_lagged = pd.DataFrame(lagged_data, index=df_imputed.index)
        
        print(f"Created {len(df_lagged.columns)} lagged features")
        
        return df_lagged
    
    def apply_log_detrending(self, df_lagged, trend_window=7):
        """
        Apply log transformation and detrending
        
        Args:
            df_lagged: DataFrame with lagged features
            trend_window: Window size for trend calculation
            
        Returns:
            DataFrame with log-detrended features
        """
        df_log = df_lagged.copy()
        
        # Apply log transformation
        df_log = np.log(df_log.replace(0, np.nan))
        
        # Calculate rolling mean (trend)
        df_trend = df_log.rolling(window=trend_window, min_periods=1).mean()
        
        # Detrend: subtract trend from log prices
        df_detrended = df_log - df_trend
        
        # Add suffix to indicate processing
        df_detrended.columns = [f"{col}_log_detrended" for col in df_detrended.columns]
        
        return df_detrended, df_trend
    
    def extract_features_for_prediction(self, df_detrended, prediction_date, feature_cols):
        """
        Extract feature values for a specific prediction date
        
        Args:
            df_detrended: Detrended lagged features
            prediction_date: Date to predict for
            feature_cols: Required feature columns from model
            
        Returns:
            Feature array or None
        """
        pred_date = pd.to_datetime(prediction_date).date()
        
        if pred_date not in df_detrended.index:
            print(f"ERROR: No data available for prediction date {pred_date}")
            print(f"Available date range: {df_detrended.index.min()} to {df_detrended.index.max()}")
            return None
        
        # Extract feature values
        X_dict = {}
        
        for feat_col in feature_cols:
            if feat_col in df_detrended.columns:
                # Competitor lag feature
                X_dict[feat_col] = df_detrended.loc[pred_date, feat_col]
            elif feat_col == 'day_of_week':
                X_dict[feat_col] = pred_date.weekday()
            elif feat_col == 'month':
                X_dict[feat_col] = pred_date.month
            elif feat_col == 'is_weekend':
                X_dict[feat_col] = 1 if pred_date.weekday() >= 5 else 0
            elif feat_col == 'day_of_year':
                X_dict[feat_col] = pred_date.timetuple().tm_yday
            elif feat_col.startswith('sin_day_of_week'):
                dow = pred_date.weekday()
                X_dict[feat_col] = np.sin(2 * np.pi * dow / 7)
            elif feat_col.startswith('cos_day_of_week'):
                dow = pred_date.weekday()
                X_dict[feat_col] = np.cos(2 * np.pi * dow / 7)
            elif feat_col.startswith('sin_month'):
                X_dict[feat_col] = np.sin(2 * np.pi * pred_date.month / 12)
            elif feat_col.startswith('cos_month'):
                X_dict[feat_col] = np.cos(2 * np.pi * pred_date.month / 12)
            elif feat_col.startswith('sin_day_of_year'):
                doy = pred_date.timetuple().tm_yday
                X_dict[feat_col] = np.sin(2 * np.pi * doy / 365)
            elif feat_col.startswith('cos_day_of_year'):
                doy = pred_date.timetuple().tm_yday
                X_dict[feat_col] = np.cos(2 * np.pi * doy / 365)
            elif feat_col.startswith('comp_'):
                # Aggregate features (comp_mean_log, comp_min_log, etc.)
                # Calculate from lag_1 prices
                lag1_cols = [c for c in df_detrended.columns if '_lag_1_log_detrended' in c]
                if lag1_cols:
                    lag1_values = df_detrended.loc[pred_date, lag1_cols].dropna()
                    if feat_col == 'comp_mean_log':
                        X_dict[feat_col] = lag1_values.mean() if len(lag1_values) > 0 else 0
                    elif feat_col == 'comp_min_log':
                        X_dict[feat_col] = lag1_values.min() if len(lag1_values) > 0 else 0
                    elif feat_col == 'comp_max_log':
                        X_dict[feat_col] = lag1_values.max() if len(lag1_values) > 0 else 0
                    elif feat_col == 'comp_std_log':
                        X_dict[feat_col] = lag1_values.std() if len(lag1_values) > 1 else 0
                    else:
                        X_dict[feat_col] = 0
                else:
                    X_dict[feat_col] = 0
            else:
                print(f"WARNING: Unknown feature {feat_col}, setting to 0")
                X_dict[feat_col] = 0
        
        # Build feature array in correct order
        X = np.array([X_dict.get(col, 0) for col in feature_cols])
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"WARNING: {nan_count} NaN values in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        return X.reshape(1, -1)
    
    def predict(self, hotel_identifier, prediction_date, return_details=False):
        """
        Make prediction for a hotel on a specific date
        
        Complete pipeline:
        1. Load model
        2. Fetch competitor data (imputation window)
        3. Impute missing values
        4. Create lagged features
        5. Apply log detrending
        6. Extract features and scale
        7. Predict
        
        Args:
            hotel_identifier: Hotel name or masked ID
            prediction_date: Date to predict for
            return_details: If True, return detailed prediction info
            
        Returns:
            Predicted base_rate or dict with details
        """
        print(f"\n{'='*60}")
        print(f"PREDICTION REQUEST")
        print(f"{'='*60}")
        
        # Convert hotel name to masked ID if needed
        if hotel_identifier.startswith('Hotel_'):
            hotel_id = hotel_identifier
            actual_hotel_id = None
        else:
            result = self.get_masked_id(hotel_identifier)
            if result is None:
                return None
            hotel_id, actual_hotel_id = result
        
        print(f"Prediction date: {prediction_date}")
        
        # Step 1: Load model
        print(f"\nStep 1/7: Loading model...")
        model_data = self.load_model(hotel_id)
        if model_data is None:
            return None
        
        config = model_data['config']
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        trend_window = config['trend_window']
        
        # Step 2: Fetch competitor data
        print(f"\nStep 2/7: Fetching competitor data...")
        df_prices = self.fetch_competitor_data(hotel_id, prediction_date, actual_hotel_id)
        if df_prices is None:
            return None
        
        # Step 3: Impute missing values
        print(f"\nStep 3/7: Imputing missing values...")
        df_matrix = self.prepare_imputation_matrix(df_prices)
        df_imputed = self.imputer.impute(df_matrix)
        
        # Step 4: Create lagged features
        print(f"\nStep 4/7: Creating lagged features...")
        df_lagged = self.create_lagged_features(df_imputed, prediction_date)
        
        # Step 5: Apply log detrending
        print(f"\nStep 5/7: Applying log detrending (window={trend_window})...")
        df_detrended, df_trend = self.apply_log_detrending(df_lagged, trend_window)
        
        # Step 6: Extract features
        print(f"\nStep 6/7: Extracting and scaling features...")
        X = self.extract_features_for_prediction(df_detrended, prediction_date, feature_cols)
        if X is None:
            return None
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Step 7: Predict
        print(f"\nStep 7/7: Making prediction...")
        y_pred_detrended = model.predict(X_scaled)[0]
        
        # Get the trend value for the prediction date
        pred_date = pd.to_datetime(prediction_date).date()
        if pred_date in df_trend.index:
            # Use the actual trend from the data
            # Average trend across all lag_1 features
            trend_cols = [c for c in df_trend.columns if '_lag_1' in c]
            y_trend = df_trend.loc[pred_date, trend_cols].mean()
        else:
            # Fallback to last known trend
            y_trend = model_data.get('y_trend_last', 0)
        
        # Add back trend
        y_pred_log = y_pred_detrended + y_trend
        
        # Inverse log transform
        predicted_rate = np.exp(y_pred_log)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Hotel: {actual_hotel_id or hotel_identifier} ({hotel_id})")
        print(f"Date: {prediction_date}")
        print(f"Predicted Base Rate: ${predicted_rate:.2f}")
        print(f"Model Configuration: trend_window={trend_window}, feature_set={config['feature_set']}")
        print(f"{'='*60}")
        
        if return_details:
            return {
                'success': True,
                'hotel_id': hotel_id,
                'hotel_name': actual_hotel_id or hotel_identifier,
                'date': str(prediction_date),
                'predicted_base_rate': float(predicted_rate),
                'model_config': {
                    'trend_window': trend_window,
                    'feature_set': config['feature_set'],
                    'n_features': len(feature_cols)
                },
                'data_stats': {
                    'records_fetched': len(df_prices),
                    'competitors': df_matrix.shape[1],
                    'date_range': f"{df_matrix.index.min()} to {df_matrix.index.max()}",
                    'missing_pct_before_imputation': f"{(df_matrix.isnull().sum().sum() / df_matrix.size) * 100:.1f}%"
                }
            }
        
        return predicted_rate


def main():
    """Command-line interface for predictions"""
    parser = argparse.ArgumentParser(description='Hotel Price Prediction System with Imputation')
    parser.add_argument('--hotel', help='Hotel name')
    parser.add_argument('--date', help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--list', action='store_true',
                       help='List all available hotels with trained models')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--models-dir', default='models/adaptive_log_method', 
                       help='Directory containing trained models')
    parser.add_argument('--imputation-window', type=int, default=60,
                       help='Days of history to fetch for imputation (default: 60)')
    parser.add_argument('--output', help='Save result to JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    db_config = {
        'host': config['database']['host'],
        'port': config['database']['port'],
        'database': config['database']['database'],
        'user': config['database']['user'],
        'password': config['database']['password']
    }
    
    # Initialize predictor
    predictor = HotelPricePredictor(
        models_dir=args.models_dir,
        db_config=db_config,
        imputation_window_days=args.imputation_window
    )
    
    # If --list, show available hotels and exit
    if args.list:
        predictor.list_available_hotels()
        return
    
    # Check required arguments for prediction
    if not args.hotel or not args.date:
        print("ERROR: --hotel and --date are required (unless using --list)")
        print("\nUsage:")
        print("  List hotels:    python model_prediction.py --list")
        print('  Predict:        python model_prediction.py --hotel "Ozarker_Lodge" --date 2026-01-15')
        return
    
    # Verify db_config is valid
    if not db_config['password']:
        print("ERROR: Database password not found in environment variable DB_PASSWORD")
        return
    
    print(f"Database configuration loaded from {args.config}")
    
    # Connect to database
    predictor.connect_db()
    
    try:
        # Make prediction
        result = predictor.predict(
            args.hotel, 
            args.date, 
            return_details=True
        )
        
        if result and args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")
    
    finally:
        predictor.disconnect_db()


if __name__ == "__main__":
    main()