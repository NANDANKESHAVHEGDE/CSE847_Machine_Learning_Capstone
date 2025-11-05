import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class MatrixImputer:
    """Imputes missing competitor prices using matrix completion - EXACT notebook implementation"""
    
    def __init__(self, method: str = 'auto', 
                 max_rank: int = None,
                 max_iter: int = 10,
                 random_state: int = 42):
        """
        Initialize imputer
        
        Args:
            method: Imputation method (kept for compatibility, actual method auto-selected)
            max_rank: Maximum rank for SoftImpute (default: min(3, min(shape)-1))
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.method = method
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.random_state = random_state
    
    def process_hotel(self, raw_focal_file: Path, 
                     raw_comp_file: Path,
                     output_file: Path) -> Dict:
        """
        Process a single hotel - EXACT notebook logic
        
        Args:
            raw_focal_file: Path to raw focal CSV
            raw_comp_file: Path to raw competitor CSV
            output_file: Path to save processed CSV
            
        Returns:
            Dictionary with processing results
        """
        hotel_masked_id = raw_focal_file.stem.replace('_raw', '')
        
        try:
            # Check files exist
            if not raw_focal_file.exists():
                logger.warning(f"{hotel_masked_id}: Focal file not found")
                return {
                    'success': False,
                    'reason': 'Focal file not found'
                }
            
            if not raw_comp_file.exists():
                logger.warning(f"{hotel_masked_id}: No competitor file found")
                return {
                    'success': False,
                    'reason': 'No competitor file'
                }
            
            # Read focal data
            focal = pd.read_csv(raw_focal_file)
            focal['stay_date'] = pd.to_datetime(focal['stay_date'])
            
            # EXACT NOTEBOOK LOGIC: groupby with .min() (not .mean()!)
            focal = focal.groupby('stay_date')['price'].min().reset_index()
            focal.columns = ['date', 'base_rate']
            focal = focal.sort_values('date').reset_index(drop=True)
            
            logger.debug(f"{hotel_masked_id}: Focal: {len(focal)} days ({focal['date'].min()} to {focal['date'].max()})")
            
            # Read competitor data
            compset = pd.read_csv(raw_comp_file)
            
            if compset.empty:
                logger.warning(f"{hotel_masked_id}: Competitor file is empty")
                return {
                    'success': False,
                    'reason': 'Empty competitor file'
                }
            
            compset['stay_date'] = pd.to_datetime(compset['stay_date'])
            
            # EXACT NOTEBOOK LOGIC: groupby with .min()
            compset = compset.groupby(['stay_date', 'hotel_id'])['price'].min().reset_index()
            compset.columns = ['date', 'hotel_id', 'competitor_price']
            
            num_competitors = compset['hotel_id'].nunique()
            logger.debug(f"{hotel_masked_id}: Compset: {num_competitors} hotels")
            
            # Pivot competitor matrix
            comp_matrix = compset.pivot(index='date', columns='hotel_id', values='competitor_price')
            
            # EXACT NOTEBOOK LOGIC: how='inner' merge (not 'left'!)
            merged = focal[['date', 'base_rate']].merge(comp_matrix, on='date', how='inner')
            merged = merged.sort_values('date').reset_index(drop=True)
            
            # Store original column names BEFORE creating matrix
            original_hotel_names = ['base_rate'] + list(comp_matrix.columns)
            
            # Create price matrix
            price_matrix = merged.drop('date', axis=1).values
            dates = merged['date'].values
            
            missing_count = np.isnan(price_matrix).sum()
            missing_pct = 100 * missing_count / price_matrix.size
            
            logger.debug(f"{hotel_masked_id}: Matrix: {price_matrix.shape}")
            logger.debug(f"{hotel_masked_id}: Missing: {missing_count} / {price_matrix.size} ({missing_pct:.1f}%)")
            
            # EXACT NOTEBOOK LOGIC: Try SoftImpute first, fallback to IterativeImputer
            try:
                from fancyimpute import SoftImpute
                max_rank = self.max_rank if self.max_rank else min(3, min(price_matrix.shape) - 1)
                imputer = SoftImpute(max_rank=max_rank, verbose=False)
                completed_matrix = imputer.fit_transform(price_matrix)
                method_used = "Soft-Impute"
                logger.debug(f"{hotel_masked_id}: Method: Soft-Impute (max_rank={max_rank})")
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
                logger.debug(f"{hotel_masked_id}: Method: IterativeImputer")
            
            # EXACT NOTEBOOK LOGIC: Fill any remaining NaNs with column mean
            remaining_missing = np.isnan(completed_matrix).sum()
            if remaining_missing > 0:
                for i in range(completed_matrix.shape[1]):
                    col_mean = np.nanmean(completed_matrix[:, i])
                    if np.isnan(col_mean):
                        col_mean = np.nanmean(completed_matrix)
                    completed_matrix[np.isnan(completed_matrix[:, i]), i] = col_mean
            
            # EXACT NOTEBOOK LOGIC: Clip negative values to 0
            if (completed_matrix < 0).sum() > 0:
                completed_matrix = np.maximum(completed_matrix, 0)
            
            # EXACT NOTEBOOK LOGIC: Handle dropped columns
            if completed_matrix.shape[1] != len(original_hotel_names):
                logger.warning(f"{hotel_masked_id}: Imputer dropped {len(original_hotel_names) - completed_matrix.shape[1]} column(s) (all NaN)")
                # Find which columns survived (not all NaN)
                survived_cols = []
                for col_idx, col_name in enumerate(original_hotel_names):
                    col_data = price_matrix[:, col_idx]
                    if not np.all(np.isnan(col_data)):
                        survived_cols.append(col_name)
                hotel_names = survived_cols
                logger.debug(f"{hotel_masked_id}: Adjusted to {len(hotel_names)} columns")
            else:
                hotel_names = original_hotel_names
            
            # EXACT NOTEBOOK LOGIC: Calculate correlation preservation
            survived_indices = [i for i, col in enumerate(original_hotel_names) if col in hotel_names]
            price_matrix_survived = price_matrix[:, survived_indices]
            
            original_df = pd.DataFrame(price_matrix_survived, columns=hotel_names)
            completed_df_temp = pd.DataFrame(completed_matrix, columns=hotel_names)
            original_corr = original_df.corr()
            completed_corr = completed_df_temp.corr()
            corr_diff = np.abs(original_corr - completed_corr)
            valid_corr = ~(original_corr.isna() | completed_corr.isna())
            avg_corr_diff = corr_diff[valid_corr].mean().mean()
            corr_preserved = (1 - avg_corr_diff) * 100
            
            logger.debug(f"{hotel_masked_id}: Correlation preserved: {corr_preserved:.1f}%")
            
            # Create final DataFrame
            completed_df = pd.DataFrame(completed_matrix, columns=hotel_names)
            completed_df.insert(0, 'date', dates)
            
            # EXACT NOTEBOOK: Save 4 output files
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_dir = output_file.parent
            
            # File 1: Complete imputed matrix (all columns)
            matrix_file = output_dir / f'{hotel_masked_id}_matrix_completion_imputed.csv'
            completed_df.to_csv(matrix_file, index=False)
            
            # File 2: Focal prices only
            focal_completed = completed_df[['date', 'base_rate']].copy()
            focal_file = output_dir / f'{hotel_masked_id}_focal_matrix_completion.csv'
            focal_completed.to_csv(focal_file, index=False)
            
            # File 3: Competitor prices in long format
            comp_completed = completed_df.drop('base_rate', axis=1).copy()
            comp_melted = comp_completed.melt(id_vars='date', var_name='hotel_id', value_name='competitor_price')
            comp_long_file = output_dir / f'{hotel_masked_id}_competitors_matrix_completion.csv'
            comp_melted.to_csv(comp_long_file, index=False)
            
            # File 4: Competitor price matrix (hotels as rows, dates as columns)
            comp_matrix_completed = comp_completed.set_index('date').T
            comp_matrix_file = output_dir / f'{hotel_masked_id}_competitor_price_matrix.csv'
            comp_matrix_completed.to_csv(comp_matrix_file)
            
            logger.info(f"Saved {hotel_masked_id}: {len(completed_df)} rows, "
                       f"{missing_count} missing values imputed")
            
            return {
                'success': True,
                'rows': len(completed_df),
                'num_competitors': len(hotel_names) - 1,
                'missing_before': int(missing_count),
                'missing_after': 0,
                'correlation_preserved_pct': float(corr_preserved),
                'method': method_used
            }
            
        except Exception as e:
            logger.error(f"Failed to process {hotel_masked_id}: {e}")
            return {
                'success': False,
                'reason': str(e)
            }
    
    def process_all_hotels(self, raw_data_path: str,
                          output_path: str,
                          hotel_mapping_file: str) -> Dict:
        """
        Process all hotels
        
        Args:
            raw_data_path: Path to raw data directory
            output_path: Path to output processed data directory
            hotel_mapping_file: Path to hotel mapping CSV
            
        Returns:
            Dictionary with summary results
        """
        raw_dir = Path(raw_data_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read hotel mapping
        mapping_df = pd.read_csv(hotel_mapping_file)
        hotel_list = mapping_df['masked_id'].tolist()
        
        logger.info(f"Processing competitor pricing for {len(hotel_list)} hotels")
        
        results = []
        
        for masked_id in hotel_list:
            raw_focal_file = raw_dir / f'{masked_id}_raw.csv'
            raw_comp_file = raw_dir / f'{masked_id}_competitors.csv'
            output_file = output_dir / f'{masked_id}_processed.csv'
            
            result = self.process_hotel(raw_focal_file, raw_comp_file, output_file)
            result['hotel'] = masked_id
            results.append(result)
        
        # Create summary
        summary_df = pd.DataFrame(results)
        successful = summary_df['success'].sum()
        failed = len(summary_df) - successful
        
        # Calculate total missing values (only for successful hotels)
        successful_hotels = summary_df[summary_df['success'] == True]
        total_missing_before = successful_hotels['missing_before'].sum() if len(successful_hotels) > 0 else 0
        total_missing_after = successful_hotels['missing_after'].sum() if len(successful_hotels) > 0 else 0
        
        logger.info(f"Matrix imputation complete: {successful}/{len(hotel_list)} successful")
        logger.info(f"Total missing values: {total_missing_before} → {total_missing_after}")
        
        # Save summary
        summary_file = output_dir.parent / 'imputation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved imputation summary to {summary_file}")
        
        return {
            'total': len(hotel_list),
            'successful': int(successful),
            'failed': int(failed),
            'missing_before': int(total_missing_before),
            'missing_after': int(total_missing_after),
            'results': results
        }