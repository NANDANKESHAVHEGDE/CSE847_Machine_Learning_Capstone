import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def detrend_log_series(series, window):
    """
    Detrend with rolling window
    EXACT match to notebook function (line 86-90)
    """
    trend = series.rolling(window=window, min_periods=1, center=False).mean()
    detrended = series - trend
    return detrended, trend


def calculate_detailed_metrics(hotel_id, model_path, data_path, config):
    """
    Calculate detailed error metrics for a hotel
    EXACTLY matches training notebook preprocessing
    """
    try:
        # Load model
        model_file = model_path / f'{hotel_id}_model.pkl'
        if not model_file.exists():
            print(f"Model file not found: {model_file}")
            return None
            
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        saved_config = model_data['config']
        feature_cols = model_data['feature_cols']
        
        # Load lagged dataset
        data_file = data_path / f'{hotel_id}_lagged_dataset.csv'
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            return None
            
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) == 0:
            print(f"Empty dataset")
            return None
        
        # Replicate EXACT notebook preprocessing
        trend_window = saved_config['trend_window']
        feature_set = saved_config['feature_set']
        
        df_processed = df.copy()
        
        # ========================================================================
        # COMPETITOR FEATURES (Lines 112-137 in notebook)
        # ========================================================================
        
        # Get lag columns
        comp_lag_cols = [c for c in df.columns if c.endswith(('_lag_1', '_lag_2', '_lag_3', '_lag_4', '_lag_5'))]
        
        if len(comp_lag_cols) == 0:
            print(f"No lag columns found")
            return None
        
        # Log + detrend with ADAPTIVE window (EXACT match to notebook line 132-137)
        for col in comp_lag_cols:
            # CRITICAL: .replace(0, np.nan) BEFORE log (notebook line 134)
            log_prices = np.log(df_processed[col].replace(0, np.nan))
            detrended, _ = detrend_log_series(log_prices, window=trend_window)
            df_processed[f'{col}_log_detrended'] = detrended
        
        # Market aggregates (ONLY for extended) (Lines 140-154 in notebook)
        if feature_set == 'extended':
            # CRITICAL: Only lag_1 columns (notebook line 141)
            lag1_cols = [col for col in comp_lag_cols if '_lag_1' in col]
            
            if len(lag1_cols) > 1:
                # CRITICAL: Log of RAW prices, not detrended (notebook line 145)
                for col in lag1_cols:
                    df_processed[f'{col}_log'] = np.log(df_processed[col].replace(0, np.nan))
                
                lag1_log_cols = [f'{col}_log' for col in lag1_cols]
                
                # Calculate aggregates across competitors (notebook lines 149-152)
                df_processed['comp_mean_log'] = df_processed[lag1_log_cols].mean(axis=1)
                df_processed['comp_min_log'] = df_processed[lag1_log_cols].min(axis=1)
                df_processed['comp_max_log'] = df_processed[lag1_log_cols].max(axis=1)
                df_processed['comp_std_log'] = df_processed[lag1_log_cols].std(axis=1)
        
        # ========================================================================
        # TARGET (Lines 184-186 in notebook)
        # ========================================================================
        
        y_original = df_processed['base_rate']
        # CRITICAL: .replace(0, np.nan) BEFORE log (notebook line 185)
        y_log = np.log(y_original.replace(0, np.nan))
        y_detrended, y_trend = detrend_log_series(y_log, window=trend_window)
        
        # ========================================================================
        # BUILD FEATURE MATRIX (Lines 188-196 in notebook)
        # ========================================================================
        
        # Build feature matrix matching model's expected features
        X_list = []
        missing_cols = []
        for feat_col in feature_cols:
            if feat_col in df_processed.columns:
                X_list.append(df_processed[feat_col].values)
            else:
                missing_cols.append(feat_col)
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        X = np.column_stack(X_list)
        
        # Drop NaN (EXACT match to notebook line 191)
        valid_idx = ~(pd.DataFrame(X).isnull().any(axis=1) | y_detrended.isnull() | y_log.isnull())
        
        if valid_idx.sum() == 0:
            print(f"No valid rows after removing NaN")
            return None
        
        if valid_idx.sum() < 10:
            print(f"Too few valid rows: {valid_idx.sum()}")
            return None
        
        # Apply mask (notebook lines 192-195)
        X_clean = X[valid_idx]
        y_detrended_clean = y_detrended[valid_idx].values
        y_trend_clean = y_trend[valid_idx].values
        y_original_clean = y_original[valid_idx].values
        
        # ========================================================================
        # SCALE AND PREDICT
        # ========================================================================
        
        X_scaled = scaler.transform(X_clean)
        
        # Make predictions
        y_pred_detrended = model.predict(X_scaled)
        
        # Reconstruct predictions (add back trend and inverse log)
        y_pred_log = y_pred_detrended + y_trend_clean
        y_pred = np.exp(y_pred_log)
        
        # ========================================================================
        # CALCULATE METRICS
        # ========================================================================
        
        rmse = calculate_rmse(y_original_clean, y_pred)
        mae = calculate_mae(y_original_clean, y_pred)
        mape = calculate_mape(y_original_clean, y_pred)
        
        # Log scale RMSE
        try:
            y_original_log = np.log(y_original_clean)
            rmse_log = calculate_rmse(y_original_log, y_pred_log)
        except:
            rmse_log = 0.0
        
        return {
            'hotel_id': hotel_id,
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'rmse_log': float(rmse_log),
            'n_samples': len(y_original_clean),
            'n_samples_total': len(y_original),
            'pct_valid': 100 * len(y_original_clean) / len(y_original),
            'mean_price': float(np.mean(y_original_clean)),
            'std_price': float(np.std(y_original_clean))
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main report generation function"""
    
    print("COMPREHENSIVE ADAPTIVE MODEL REPORT")
    print("Using EXACT notebook preprocessing")
    
    # Paths
    output_path = Path('models/adaptive_log_method')
    processed_data_path = Path('data/full-data/processed')
    report_dir = Path('reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\nLoading training results...")
    with open(output_path / 'adaptive_summary.json', 'r') as f:
        results = json.load(f)
    
    successful = {h: r for h, r in results.items() if r.get('status') == 'success'}
    
    print(f"Found {len(successful)} successfully trained hotels")
    
    # Calculate detailed metrics
    print("\nCalculating detailed metrics for each hotel...")
    detailed_metrics = []
    
    for hotel_id in successful.keys():
        print(f"  Processing {hotel_id}...", end=' ')
        metrics = calculate_detailed_metrics(
            hotel_id, output_path, processed_data_path, 
            successful[hotel_id]
        )
        if metrics:
            # Merge with training results
            metrics.update({
                'train_r2': successful[hotel_id].get('train_r2', 0.0),
                'test_r2': successful[hotel_id].get('test_r2', 0.0),
                'trend_window': successful[hotel_id].get('trend_window', 7),
                'min_train_days': successful[hotel_id].get('min_train_days', 180),
                'feature_set': successful[hotel_id].get('feature_set', 'simple'),
                'n_features': successful[hotel_id].get('n_features', 0)
            })
            detailed_metrics.append(metrics)
            print("done")
        else:
            print("failed")
    
    if len(detailed_metrics) == 0:
        print("\nERROR: No metrics could be calculated")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(detailed_metrics)
    
    print(f"\nSuccessfully calculated metrics for {len(detailed_metrics)} hotels")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nPerformance Metrics:")
    print(f"  Mean Test R²: {results_df['test_r2'].mean():.4f}")
    print(f"  Median Test R²: {results_df['test_r2'].median():.4f}")
    print(f"  Std Test R²: {results_df['test_r2'].std():.4f}")
    
    print("\nError Metrics:")
    print(f"  Mean RMSE: ${results_df['rmse'].mean():.2f}")
    print(f"  Mean MAE: ${results_df['mae'].mean():.2f}")
    print(f"  Mean MAPE: {results_df['mape'].mean():.2f}%")
    
    print("\nConfiguration Distribution:")
    print(f"  Trend Window 7: {(results_df['trend_window'] == 7).sum()} hotels")
    print(f"  Trend Window 14: {(results_df['trend_window'] == 14).sum()} hotels")
    print(f"  Trend Window 21: {(results_df['trend_window'] == 21).sum()} hotels")
    print(f"  Trend Window 28: {(results_df['trend_window'] == 28).sum()} hotels")
    
    # Top performers
    print("\n" + "="*80)
    print("TOP 10 PERFORMERS (by Test R²)")
    print("="*80)
    
    top_10 = results_df.nlargest(10, 'test_r2')
    display_cols = ['hotel_id', 'test_r2', 'rmse', 'mape', 'trend_window', 'feature_set']
    print("\n" + top_10[display_cols].to_string(index=False))
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_file = report_dir / 'comprehensive_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved comprehensive results to {results_file}")
    
    top_performers_file = report_dir / 'top_performers.csv'
    top_10.to_csv(top_performers_file, index=False)
    print(f"Saved top performers to {top_performers_file}")
    
    summary_stats = {
        'total_hotels': len(results_df),
        'mean_test_r2': float(results_df['test_r2'].mean()),
        'median_test_r2': float(results_df['test_r2'].median()),
        'std_test_r2': float(results_df['test_r2'].std()),
        'mean_rmse': float(results_df['rmse'].mean()),
        'mean_mae': float(results_df['mae'].mean()),
        'mean_mape': float(results_df['mape'].mean()),
        'best_hotel': results_df.loc[results_df['test_r2'].idxmax(), 'hotel_id'],
        'best_r2': float(results_df['test_r2'].max())
    }
    
    summary_file = report_dir / 'summary_statistics.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved summary statistics to {summary_file}")
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll reports saved to: {report_dir}/")
    print("  - comprehensive_results.csv")
    print("  - top_performers.csv")
    print("  - summary_statistics.json")


if __name__ == "__main__":
    main()