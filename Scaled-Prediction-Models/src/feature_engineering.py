import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates lagged features from matrix completion outputs - EXACT notebook implementation"""
    
    def __init__(self, lags: list = None, rolling_windows: list = None):
        """
        Initialize feature engineer
        
        Args:
            lags: List of lag periods (kept for compatibility, actual value from notebook is [1,2,3,4,5])
            rolling_windows: Not used (kept for compatibility)
        """
        # EXACT notebook values
        self.final_lags = [1, 2, 3, 4, 5]
        logger.info(f"Using lags from notebook: {self.final_lags}")
    
    def process_hotel(self, hotel_masked_id: str, 
                     processed_data_path: Path,
                     output_path: Path) -> Dict:
        """
        Process a single hotel - EXACT notebook logic
        
        Args:
            hotel_masked_id: Hotel masked ID
            processed_data_path: Path to processed data directory
            output_path: Path to save output (same as input in notebook)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # EXACT NOTEBOOK: Read focal_matrix_completion.csv
            focal_file = processed_data_path / f'{hotel_masked_id}_focal_matrix_completion.csv'
            if not focal_file.exists():
                logger.warning(f"{hotel_masked_id}: Focal file not found")
                return {
                    'success': False,
                    'hotel_id': hotel_masked_id,
                    'reason': 'Focal file not found'
                }
            
            focal = pd.read_csv(focal_file)
            focal['date'] = pd.to_datetime(focal['date'])
            
            logger.debug(f"{hotel_masked_id}: Focal: {len(focal)} rows ({focal['date'].min()} to {focal['date'].max()})")
            
            # EXACT NOTEBOOK: Read competitor_price_matrix.csv
            comp_matrix_file = processed_data_path / f'{hotel_masked_id}_competitor_price_matrix.csv'
            if not comp_matrix_file.exists():
                logger.warning(f"{hotel_masked_id}: Competitor matrix not found")
                return {
                    'success': False,
                    'hotel_id': hotel_masked_id,
                    'reason': 'Competitor matrix not found'
                }
            
            # EXACT NOTEBOOK: Read competitor matrix (hotels as rows, dates as columns)
            comp_matrix = pd.read_csv(comp_matrix_file, index_col=0)
            comp_matrix.columns = pd.to_datetime(comp_matrix.columns)
            
            logger.debug(f"{hotel_masked_id}: Competitor matrix: {comp_matrix.shape}")
            
            # EXACT NOTEBOOK: Merge focal with competitors
            merged_data = []
            for date in focal['date']:
                row = {
                    'date': date,
                    'base_rate': focal[focal['date'] == date]['base_rate'].values[0]
                }
                # Add competitor prices for this date
                if date in comp_matrix.columns:
                    for hotel in comp_matrix.index:
                        row[hotel] = comp_matrix.loc[hotel, date]
                merged_data.append(row)
            
            df_final = pd.DataFrame(merged_data)
            
            # EXACT NOTEBOOK: Add temporal features
            df_final['day_of_week'] = df_final['date'].dt.dayofweek
            df_final['month'] = df_final['date'].dt.month
            df_final['is_weekend'] = (df_final['day_of_week'] >= 5).astype(int)
            
            logger.debug(f"{hotel_masked_id}: After merge: {df_final.shape}")
            
            # EXACT NOTEBOOK: Get all price columns
            price_columns = [col for col in df_final.columns 
                           if col not in ['date', 'day_of_week', 'month', 'is_weekend']]
            
            # EXACT NOTEBOOK: Create lags [1,2,3,4,5] for ALL price columns
            df_lagged = df_final.copy()
            for col in price_columns:
                for lag in self.final_lags:
                    df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
            
            logger.debug(f"{hotel_masked_id}: After lagging: {df_lagged.shape}")
            logger.debug(f"{hotel_masked_id}: Missing values: {df_lagged.isnull().sum().sum()}")
            
            # EXACT NOTEBOOK: Drop NaN rows
            df_lagged_clean = df_lagged.dropna()
            
            logger.debug(f"{hotel_masked_id}: After dropping NaN: {df_lagged_clean.shape}")
            data_retention = len(df_lagged_clean) / len(df_final) * 100
            logger.debug(f"{hotel_masked_id}: Data retention: {data_retention:.1f}%")
            
            # EXACT NOTEBOOK: Add cyclical encoding
            df_with_temporal = df_lagged_clean.copy()
            df_with_temporal['day_of_year'] = df_with_temporal['date'].dt.dayofyear
            df_with_temporal['sin_day_of_week'] = np.sin(2 * np.pi * df_with_temporal['day_of_week'] / 7)
            df_with_temporal['cos_day_of_week'] = np.cos(2 * np.pi * df_with_temporal['day_of_week'] / 7)
            df_with_temporal['sin_month'] = np.sin(2 * np.pi * df_with_temporal['month'] / 12)
            df_with_temporal['cos_month'] = np.cos(2 * np.pi * df_with_temporal['month'] / 12)
            df_with_temporal['sin_day_of_year'] = np.sin(2 * np.pi * df_with_temporal['day_of_year'] / 365)
            df_with_temporal['cos_day_of_year'] = np.cos(2 * np.pi * df_with_temporal['day_of_year'] / 365)
            
            logger.debug(f"{hotel_masked_id}: Final shape: {df_with_temporal.shape}")
            
            # EXACT NOTEBOOK: Save lagged dataset
            output_file = output_path / f'{hotel_masked_id}_lagged_dataset.csv'
            df_with_temporal.to_csv(output_file, index=False)
            
            # EXACT NOTEBOOK: Calculate metadata
            total_observations = len(df_with_temporal)
            total_features = len(df_with_temporal.columns) - 1  # Exclude 'date'
            lag_features = len([col for col in df_with_temporal.columns if 'lag' in col])
            temporal_features = 7  # sin/cos for day_of_week, month, day_of_year, plus is_weekend
            
            # EXACT NOTEBOOK: Create and save metadata
            lag_metadata = {
                'hotel_id': hotel_masked_id,
                'imputation_method': 'matrix_completion',
                'selected_lags': self.final_lags,
                'focal_column': 'base_rate',
                'total_lag_features': lag_features,
                'temporal_features': ['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month',
                                     'sin_day_of_year', 'cos_day_of_year', 'is_weekend'],
                'final_observations': total_observations,
                'original_observations': len(df_final),
                'data_retention_pct': round(data_retention, 1),
                'feature_summary': {
                    'total_features': total_features,
                    'lag_features': lag_features,
                    'temporal_features': temporal_features,
                    'price_features': len(price_columns)
                }
            }
            
            metadata_file = output_path / f'{hotel_masked_id}_lag_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(lag_metadata, f, indent=2)
            
            logger.info(f"Saved {hotel_masked_id}: lagged dataset and metadata")
            
            return {
                'success': True,
                'hotel_id': hotel_masked_id,
                'original_rows': len(df_final),
                'final_rows': total_observations,
                'rows_lost': len(df_final) - total_observations,
                'data_retention_pct': round(data_retention, 1),
                'total_features': total_features,
                'lag_features': lag_features,
                'price_columns': len(price_columns),
                'num_competitors': len(price_columns) - 1  # Exclude base_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to process {hotel_masked_id}: {e}")
            return {
                'success': False,
                'hotel_id': hotel_masked_id,
                'reason': str(e)
            }
    
    def process_all_hotels(self, processed_data_path: str,
                          hotel_mapping_file: str) -> Dict:
        """
        Process all hotels
        
        Args:
            processed_data_path: Path to processed data directory (where matrix completion outputs are)
            hotel_mapping_file: Path to hotel mapping CSV
            
        Returns:
            Dictionary with summary results
        """
        data_path = Path(processed_data_path)
        # Output to same directory as input (as in notebook)
        output_path = data_path
        
        # Read hotel mapping
        mapping_df = pd.read_csv(hotel_mapping_file)
        hotel_list = mapping_df['masked_id'].tolist()
        
        logger.info(f"Creating lagged features for {len(hotel_list)} focal hotels")
        
        summary_results = []
        
        for hotel_masked_id in hotel_list:
            result = self.process_hotel(hotel_masked_id, data_path, output_path)
            
            # Convert numpy types to Python types for JSON serialization
            if result['success']:
                result_dict = {
                    'hotel_id': result['hotel_id'],
                    'original_rows': int(result['original_rows']),
                    'final_rows': int(result['final_rows']),
                    'rows_lost': int(result['rows_lost']),
                    'data_retention_pct': float(result['data_retention_pct']),
                    'total_features': int(result['total_features']),
                    'lag_features': int(result['lag_features']),
                    'price_columns': int(result['price_columns']),
                    'num_competitors': int(result['num_competitors']),
                    'status': 'Success'
                }
            else:
                result_dict = {
                    'hotel_id': result['hotel_id'],
                    'original_rows': 0,
                    'final_rows': 0,
                    'rows_lost': 0,
                    'data_retention_pct': 0.0,
                    'total_features': 0,
                    'lag_features': 0,
                    'price_columns': 0,
                    'num_competitors': 0,
                    'status': f"Failed: {result.get('reason', 'Unknown error')}"
                }
            
            summary_results.append(result_dict)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_results)
        
        # Save summary
        summary_file = output_path / 'lagged_features_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        successful = len(summary_df[summary_df['status'] == 'Success'])
        failed = len(summary_df) - successful
        
        logger.info(f"Lagged features complete: {successful}/{len(hotel_list)} successful")
        
        # Print statistics (as in notebook)
        success_df = summary_df[summary_df['status'] == 'Success']
        if len(success_df) > 0:
            logger.info(f"Average original rows: {success_df['original_rows'].mean():.0f}")
            logger.info(f"Average final rows: {success_df['final_rows'].mean():.0f}")
            logger.info(f"Average rows lost to lagging: {success_df['rows_lost'].mean():.1f}")
            logger.info(f"Average data retention: {success_df['data_retention_pct'].mean():.1f}%")
            logger.info(f"Average total features: {success_df['total_features'].mean():.0f}")
            logger.info(f"Average lag features: {success_df['lag_features'].mean():.0f}")
            logger.info(f"Average competitors: {success_df['num_competitors'].mean():.1f}")
        
        return {
            'total': len(hotel_list),
            'successful': int(successful),
            'failed': int(failed),
            'results': summary_results
        }