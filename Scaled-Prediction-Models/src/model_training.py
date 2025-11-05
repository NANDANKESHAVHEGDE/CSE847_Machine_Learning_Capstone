import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveModelTrainer:
    """Trains adaptive log-detrended XGBoost models for hotel price prediction"""
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.models = {}
        self.best_configs = {}
        self.all_results = {}
        
        # XGBoost hyperparameters - ACTUAL PARAMS FROM NOTEBOOK
        self.xgb_params = {
            'max_depth': config['model'].get('max_depth', 4),
            'n_estimators': config['model'].get('n_estimators', 100),
            'learning_rate': config['model'].get('learning_rate', 0.05),
            'reg_alpha': config['model'].get('reg_alpha', 0.3),
            'reg_lambda': config['model'].get('reg_lambda', 2.0),
            'min_child_weight': config['model'].get('min_child_weight', 5),
            'gamma': config['model'].get('gamma', 0.1),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Generate all configurations to test
        self.configurations = self._generate_configurations(config)
        
        logger.info(f"Initialized with {len(self.configurations)} configurations to test")
    
    def _generate_configurations(self, config: dict) -> List[Dict]:
        """Generate all configuration combinations"""
        configurations = []
        
        trend_windows = config['adaptive'].get('trend_windows', [7, 14, 21, 28])
        min_train_sizes = config['adaptive'].get('min_train_sizes', [180, 200, 250])
        feature_sets = config['adaptive'].get('feature_sets', ['simple', 'extended'])
        test_window = config['adaptive'].get('test_window', 30)
        
        for trend_window in trend_windows:
            for min_train in min_train_sizes:
                for feature_set in feature_sets:
                    configurations.append({
                        'trend_window': trend_window,
                        'min_train_days': min_train,
                        'test_window': test_window,
                        'feature_set': feature_set,
                        'name': f'trend{trend_window}_train{min_train}_{feature_set}'
                    })
        
        return configurations
    
    def detrend_log_series(self, series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Detrend log-transformed series with flexible window
        
        Args:
            series: Price series
            window: Rolling window size
            
        Returns:
            Tuple of (detrended_series, trend_series)
        """
        trend = series.rolling(window=window, min_periods=1, center=False).mean()
        detrended = series - trend
        return detrended, trend
    
    def prepare_features(self, df: pd.DataFrame, config: Dict) -> Optional[Tuple]:
        """
        Prepare features with ADAPTIVE configuration
        
        Args:
            df: Lagged dataset
            config: Configuration dictionary with trend_window and feature_set
            
        Returns:
            Tuple of (X, y_detrended, y_trend, y_original, feature_cols) or None
        """
        df_processed = df.copy()
        feature_cols = []
        
        trend_window = config['trend_window']
        feature_set = config['feature_set']
        
        # ========================================================================
        # COMPETITOR FEATURES
        # ========================================================================
        
        # Get all competitor lag columns (excluding base_rate)
        all_comp_lag_cols = [col for col in df.columns 
                            if '_lag_' in col 
                            and 'base_rate' not in col.lower()
                            and any(currency in col for currency in ['-USD', '-EUR', '-HKD', '-CNY'])]
        
        if len(all_comp_lag_cols) == 0:
            return None
        
        # Filter based on feature set
        if feature_set == 'simple':
            # Only lags 1-5 (like baseline)
            comp_lag_cols = [col for col in all_comp_lag_cols 
                            if any(f'_lag_{i}' in col for i in [1, 2, 3, 4, 5])]
        else:
            # All available lags (extended)
            comp_lag_cols = all_comp_lag_cols
        
        if len(comp_lag_cols) == 0:
            return None
        
        # Log + detrend with ADAPTIVE window
        for col in comp_lag_cols:
            log_prices = np.log(df_processed[col].replace(0, np.nan))
            detrended, _ = self.detrend_log_series(log_prices, window=trend_window)
            df_processed[f'{col}_log_detrended'] = detrended
            feature_cols.append(f'{col}_log_detrended')
        
        # Market aggregates (only for extended)
        if feature_set == 'extended':
            lag1_cols = [col for col in comp_lag_cols if '_lag_1' in col]
            
            if len(lag1_cols) > 1:
                for col in lag1_cols:
                    df_processed[f'{col}_log'] = np.log(df_processed[col].replace(0, np.nan))
                
                lag1_log_cols = [f'{col}_log' for col in lag1_cols]
                
                df_processed['comp_mean_log'] = df_processed[lag1_log_cols].mean(axis=1)
                df_processed['comp_min_log'] = df_processed[lag1_log_cols].min(axis=1)
                df_processed['comp_max_log'] = df_processed[lag1_log_cols].max(axis=1)
                df_processed['comp_std_log'] = df_processed[lag1_log_cols].std(axis=1)
                
                feature_cols.extend(['comp_mean_log', 'comp_min_log', 
                                    'comp_max_log', 'comp_std_log'])
        
        # ========================================================================
        # TEMPORAL FEATURES
        # ========================================================================
        
        temporal_cols = ['day_of_week', 'month', 'is_weekend', 'day_of_year']
        temporal_cols = [col for col in temporal_cols if col in df_processed.columns]
        feature_cols.extend(temporal_cols)
        
        # Cyclical encoding
        if 'day_of_week' in df_processed.columns:
            df_processed['sin_day_of_week'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['cos_day_of_week'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
            feature_cols.extend(['sin_day_of_week', 'cos_day_of_week'])
        
        if 'month' in df_processed.columns:
            df_processed['sin_month'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['cos_month'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            feature_cols.extend(['sin_month', 'cos_month'])
        
        if 'day_of_year' in df_processed.columns:
            df_processed['sin_day_of_year'] = np.sin(2 * np.pi * df_processed['day_of_year'] / 365)
            df_processed['cos_day_of_year'] = np.cos(2 * np.pi * df_processed['day_of_year'] / 365)
            feature_cols.extend(['sin_day_of_year', 'cos_day_of_year'])
        
        # ========================================================================
        # TARGET (with ADAPTIVE detrending window)
        # ========================================================================
        
        y_original = df_processed['base_rate']
        y_log = np.log(y_original.replace(0, np.nan))
        y_detrended, y_trend = self.detrend_log_series(y_log, window=trend_window)
        
        X = df_processed[feature_cols].copy()
        
        # Drop NaN
        valid_idx = ~(X.isnull().any(axis=1) | y_detrended.isnull() | y_log.isnull())
        X = X[valid_idx].reset_index(drop=True)
        y_detrended = y_detrended[valid_idx].reset_index(drop=True)
        y_trend = y_trend[valid_idx].reset_index(drop=True)
        y_original = y_original[valid_idx].reset_index(drop=True)
        
        return X, y_detrended, y_trend, y_original, feature_cols
    
    def time_series_cv_splits(self, n_samples: int, min_train: int, 
                             test_window: int) -> Optional[List[Dict]]:
        """
        Time-series CV with ADAPTIVE split sizes
        
        Args:
            n_samples: Total number of samples
            min_train: Minimum training size
            test_window: Test window size
            
        Returns:
            List of split dictionaries or None if insufficient data
        """
        splits = []
        train_end = min_train
        
        while train_end + test_window <= n_samples and len(splits) < 6:
            splits.append({
                'train_idx': list(range(0, train_end)),
                'test_idx': list(range(train_end, min(train_end + test_window, n_samples)))
            })
            train_end += test_window
        
        if len(splits) < 3:
            return None
        
        return splits
    
    def evaluate_configuration(self, X_all: pd.DataFrame, y_detrended: pd.Series,
                              y_trend: pd.Series, y_original: pd.Series, 
                              config: Dict) -> Optional[Dict]:
        """
        Evaluate one configuration with CV
        
        Args:
            X_all: Feature matrix
            y_detrended: Detrended target
            y_trend: Trend component
            y_original: Original target
            config: Configuration to evaluate
            
        Returns:
            Dictionary with CV metrics or None
        """
        splits = self.time_series_cv_splits(len(X_all), config['min_train_days'], 
                                           config['test_window'])
        
        if splits is None:
            return None
        
        test_r2_scores = []
        train_r2_scores = []
        
        for split in splits:
            X_train = X_all.iloc[split['train_idx']]
            X_test = X_all.iloc[split['test_idx']]
            y_train = y_detrended.iloc[split['train_idx']]
            y_test_detrended = y_detrended.iloc[split['test_idx']]
            y_trend_train = y_trend.iloc[split['train_idx']]
            y_trend_test = y_trend.iloc[split['test_idx']]
            y_train_original = y_original.iloc[split['train_idx']]
            y_test_original = y_original.iloc[split['test_idx']]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = XGBRegressor(**self.xgb_params)
            model.fit(X_train_scaled, y_train, verbose=False)
            
            # Train predictions (on original scale)
            y_train_pred_detrended = model.predict(X_train_scaled)
            y_train_pred_log = y_train_pred_detrended + y_trend_train.values
            y_train_pred = np.exp(y_train_pred_log)
            train_r2 = r2_score(y_train_original, y_train_pred)
            
            # Test predictions (on original scale)
            y_test_pred_detrended = model.predict(X_test_scaled)
            y_test_pred_log = y_test_pred_detrended + y_trend_test.values
            y_test_pred = np.exp(y_test_pred_log)
            test_r2 = r2_score(y_test_original, y_test_pred)
            
            test_r2_scores.append(test_r2)
            train_r2_scores.append(train_r2)
        
        return {
            'mean_test_r2': float(np.mean(test_r2_scores)),
            'mean_train_r2': float(np.mean(train_r2_scores)),
            'n_folds': len(splits)
        }
    
    def train_final_model(self, X_all: pd.DataFrame, 
                         y_detrended: pd.Series) -> Tuple:
        """
        Train final model on all data
        UPDATED: Also returns scaled train/test splits for SHAP analysis
        
        Args:
            X_all: Feature matrix
            y_detrended: Detrended target
            
        Returns:
            Tuple of (model, scaler, X_train_scaled, X_test_scaled)
        """
        # Split data for saving (80/20 split for SHAP analysis)
        split_idx = int(len(X_all) * 0.8)
        X_train = X_all.iloc[:split_idx]
        X_test = X_all.iloc[split_idx:]
        y_train = y_detrended.iloc[:split_idx]
        y_test = y_detrended.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train on all data (not just train split)
        X_all_scaled = scaler.transform(X_all)
        model = XGBRegressor(**self.xgb_params)
        model.fit(X_all_scaled, y_detrended, verbose=False)
        
        return model, scaler, X_train_scaled, X_test_scaled
    
    def process_hotel_adaptive(self, hotel_id: str, 
                               data_path: Path,
                               data_end_date: Optional[str] = None) -> Dict:
        """
        Process one hotel with ADAPTIVE approach:
        1. Try all configurations
        2. Select best based on test R²
        3. Train final model with best config
        
        Args:
            hotel_id: Hotel identifier
            data_path: Path to processed data directory
            data_end_date: Optional date cutoff (for train/test split)
            
        Returns:
            Dictionary with results
        """
        try:
            data_file = data_path / f'{hotel_id}_lagged_dataset.csv'
            
            if not data_file.exists():
                return {'hotel_id': hotel_id, 'status': 'missing_file'}
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            if data_end_date:
                df = df[df['date'] <= data_end_date].copy()
            
            df = df.sort_values('date').reset_index(drop=True)
            
            # Need minimum data for any config
            if len(df) < 180 + 30:  # Smallest min_train + test
                return {
                    'hotel_id': hotel_id, 
                    'status': 'insufficient_data',
                    'observations': len(df)
                }
            
            # Try all configurations
            config_results = {}
            
            for config in self.configurations:
                # Prepare features
                result = self.prepare_features(df, config)
                
                if result is None:
                    continue
                
                X, y_detrended, y_trend, y_original, feature_cols = result
                
                # Need enough data for this config
                if len(X) < config['min_train_days'] + config['test_window']:
                    continue
                
                # Evaluate
                eval_result = self.evaluate_configuration(X, y_detrended, y_trend, 
                                                         y_original, config)
                
                if eval_result is not None:
                    config_results[config['name']] = {
                        'config': config,
                        'test_r2': eval_result['mean_test_r2'],
                        'train_r2': eval_result['mean_train_r2'],
                        'n_folds': eval_result['n_folds'],
                        'n_features': len(feature_cols)
                    }
            
            if len(config_results) == 0:
                return {'hotel_id': hotel_id, 'status': 'all_configs_failed'}
            
            # Select BEST configuration
            best_config_name = max(config_results.keys(), 
                                  key=lambda k: config_results[k]['test_r2'])
            best_config = config_results[best_config_name]['config']
            best_test_r2 = config_results[best_config_name]['test_r2']
            
            # Train final model with best config
            result = self.prepare_features(df, best_config)
            X, y_detrended, y_trend, y_original, feature_cols = result
            
            model, scaler, X_train_scaled, X_test_scaled = self.train_final_model(X, y_detrended)
            
            # Save model with preprocessed data for SHAP
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'config': best_config,
                'trend_window': best_config['trend_window'],
                'y_trend_last': y_trend.iloc[-1],
                'method': 'adaptive_log_detrending',
                'X_train': X_train_scaled,  # ADDED for SHAP
                'X_test': X_test_scaled     # ADDED for SHAP
            }
            
            self.models[hotel_id] = model_data
            self.best_configs[hotel_id] = best_config
            
            return {
                'hotel_id': hotel_id,
                'status': 'success',
                'best_config_name': best_config_name,
                'trend_window': best_config['trend_window'],
                'min_train_days': best_config['min_train_days'],
                'feature_set': best_config['feature_set'],
                'test_r2': best_test_r2,
                'train_r2': config_results[best_config_name]['train_r2'],
                'n_features': len(feature_cols),
                'n_folds': config_results[best_config_name]['n_folds'],
                'n_configs_tried': len(config_results),
                'all_configs': {k: {'test_r2': v['test_r2'], 'n_features': v['n_features']} 
                               for k, v in config_results.items()}
            }
            
        except Exception as e:
            logger.error(f"{hotel_id}: Error - {str(e)}")
            return {'hotel_id': hotel_id, 'status': 'error', 'error': str(e)}
    
    def train_all_hotels(self, processed_data_path: str,
                        hotel_mapping_file: str,
                        output_dir: str,
                        data_end_date: Optional[str] = None) -> Dict:
        """
        Train models for all hotels with adaptive configuration selection
        
        Args:
            processed_data_path: Path to processed data directory
            hotel_mapping_file: Path to hotel_mapping.csv
            output_dir: Directory to save models
            data_end_date: Optional date cutoff
            
        Returns:
            Summary dictionary
        """
        data_path = Path(processed_data_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load hotel mapping
        mapping_df = pd.read_csv(hotel_mapping_file)
        hotel_list = mapping_df['masked_id'].tolist()
        
        logger.info(f"Processing {len(hotel_list)} hotels with ADAPTIVE configurations...")
        logger.info(f"Testing {len(self.configurations)} configurations per hotel")
        
        successful = []
        
        for idx, hotel_id in enumerate(hotel_list, 1):
            logger.info(f"[{idx}/{len(hotel_list)}] {hotel_id}...")
            
            result = self.process_hotel_adaptive(hotel_id, data_path, data_end_date)
            self.all_results[hotel_id] = result
            
            if result['status'] == 'success':
                successful.append(hotel_id)
                config_info = f"trend={result['trend_window']}, {result['feature_set']}"
                logger.info(f"R²={result['test_r2']:.4f} ({config_info})")
            else:
                logger.warning(f"{result['status']}")
        
        # Save all models
        for hotel_id in successful:
            model_file = output_path / f'{hotel_id}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(self.models[hotel_id], f)
        
        # Save summary
        with open(output_path / 'adaptive_summary.json', 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Create deployment CSV
        deployment_df = pd.DataFrame([
            {
                'hotel_id': h,
                'test_r2': self.all_results[h]['test_r2'],
                'train_r2': self.all_results[h]['train_r2'],
                'trend_window': self.all_results[h]['trend_window'],
                'feature_set': self.all_results[h]['feature_set'],
                'n_features': self.all_results[h]['n_features'],
                'n_folds': self.all_results[h]['n_folds'],
                'n_configs_tried': self.all_results[h]['n_configs_tried'],
                'model_file': f'{h}_model.pkl'
            }
            for h in successful
        ])
        deployment_df = deployment_df.sort_values('test_r2', ascending=False)
        deployment_df.to_csv(output_path / 'deployment_adaptive.csv', index=False)
        
        logger.info(f"\nModels saved to: {output_path}/")
        
        return {
            'total': len(hotel_list),
            'successful': len(successful),
            'failed': len(hotel_list) - len(successful),
            'successful_hotels': successful,
            'all_results': self.all_results
        }