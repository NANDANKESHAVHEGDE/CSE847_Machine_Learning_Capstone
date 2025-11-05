"""
Main Orchestration Script
Runs the complete adaptive log detrending pipeline
Based on actual notebook workflow
"""

import yaml
import logging
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_fetching import DataFetcher
from src.matrix_imputation import MatrixImputer
from src.feature_engineering import FeatureEngineer
from src.model_training import AdaptiveModelTrainer
from src.utils import (setup_logging, ensure_directory,
                      print_summary_stats, print_configuration_analysis)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution function"""
    
    print("="*80)
    print("ADAPTIVE LOG DETRENDING MODEL - PRODUCTION")
    print("="*80)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config()
    
    # Load environment variables (for DB password)
    load_dotenv()
    if config['database']['password'] == 'YOUR_DB_PASSWORD':
        config['database']['password'] = os.getenv('DB_PASSWORD')
        if not config['database']['password']:
            print("ERROR: DB_PASSWORD not found in environment variables")
            sys.exit(1)
    
    # Setup logging
    setup_logging(
        log_dir=config['output']['logs_dir'],
        log_level=config['logging']['level']
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptive log detrending pipeline")
    
    # Create output directories
    for dir_key in ['base_dir', 'raw_dir', 'processed_dir', 'models_dir', 'logs_dir']:
        ensure_directory(config['output'][dir_key])
    
    # STEP 1: FETCH DATA FROM DATABASE
    print("\n" + "="*80)
    print("[2/7] FETCHING DATA FROM DATABASE")
    print("="*80)
    
    fetcher = DataFetcher(config['database'])
    
    try:
        summary = fetcher.fetch_complete_dataset(
            output_dir=config['output']['base_dir'],
            start_date=config['data'].get('start_date'),
            end_date=config['data'].get('end_date')
        )
        
        print(f"\nFetched data for {summary['successful']}/{summary['total_hotels']} hotels")
        
        if summary['failed'] > 0:
            print(f"Warning: {summary['failed']} hotels failed to fetch")
        
    finally:
        fetcher.close()
    
    hotel_mapping_file = Path(config['output']['base_dir']) / 'hotel_mapping.csv'
    
    # STEP 2: MATRIX IMPUTATION
    print("\n" + "="*80)
    print("[3/7] MATRIX IMPUTATION")
    print("="*80)
    
    imputer = MatrixImputer(
        method=config['imputation']['method'],
        max_rank=config['imputation']['max_rank'],
        max_iter=config['imputation']['max_iter'],
        random_state=config['imputation']['random_state']
    )
    
    imputation_summary = imputer.process_all_hotels(
        raw_data_path=config['output']['raw_dir'],
        output_path=config['output']['processed_dir'],
        hotel_mapping_file=str(hotel_mapping_file)
    )
    
    print(f"\nSuccessfully imputed {imputation_summary['successful']}/{imputation_summary['total']} hotels")
    
    if imputation_summary['failed'] > 0:
        print(f"Warning: {imputation_summary['failed']} hotels failed")
    
    # STEP 3: FEATURE ENGINEERING
    print("\n" + "="*80)
    print("[4/7] FEATURE ENGINEERING")
    print("="*80)
    
    engineer = FeatureEngineer(lags=config['features']['lags'])
    
    feature_summary = engineer.process_all_hotels(
        processed_data_path=config['output']['processed_dir'],
        hotel_mapping_file=str(hotel_mapping_file)
    )
    
    print(f"\nCreated features for {feature_summary['successful']}/{feature_summary['total']} hotels")
    
    if feature_summary['failed'] > 0:
        print(f"Warning: {feature_summary['failed']} hotels failed")
    
    # STEP 4: ADAPTIVE MODEL TRAINING
    print("\n" + "="*80)
    print("[5/7] ADAPTIVE MODEL TRAINING")
    print("="*80)
    print("Testing multiple configurations per hotel - this may take a while...")
    print(f"  Trend windows: {config['adaptive']['trend_windows']}")
    print(f"  Min train sizes: {config['adaptive']['min_train_sizes']}")
    print(f"  Feature sets: {config['adaptive']['feature_sets']}")
    
    # Calculate total configurations
    n_configs = (len(config['adaptive']['trend_windows']) * 
                 len(config['adaptive']['min_train_sizes']) * 
                 len(config['adaptive']['feature_sets']))
    print(f"  Total configurations per hotel: {n_configs}")
    
    trainer = AdaptiveModelTrainer(config)
    
    training_summary = trainer.train_all_hotels(
        processed_data_path=config['output']['processed_dir'],
        hotel_mapping_file=str(hotel_mapping_file),
        output_dir=config['output']['models_dir'],
        data_end_date=config['data'].get('end_date')
    )
    
    print(f"\nSuccessfully trained {training_summary['successful']}/{training_summary['total']} hotels")
    
    if training_summary['failed'] > 0:
        print(f"Warning: {training_summary['failed']} hotels failed")
    
    # STEP 5: RESULTS SUMMARY
    print("\n" + "="*80)
    print("[6/7] TRAINING SUMMARY")
    print("="*80)
    
    successful_results = {
        h: r for h, r in training_summary['all_results'].items() 
        if r.get('status') == 'success'
    }
    
    if len(successful_results) > 0:
        # Print statistics
        print_summary_stats(successful_results)
        
        # Print configuration analysis
        print_configuration_analysis(successful_results)
        
        print(f"\nModels saved to: {config['output']['models_dir']}/")
        print(f"Summary: adaptive_summary.json")
        print(f"Deployment: deployment_adaptive.csv")
    
    # STEP 6: COMPARISON TO BASELINE
    if len(successful_results) > 0:
        import numpy as np
        
        test_r2_values = [r['test_r2'] for r in successful_results.values()]
        mean_r2 = np.mean(test_r2_values)
        median_r2 = np.median(test_r2_values)
        
        print("\n" + "="*80)
        print("[7/7] COMPARISON TO BASELINE")
        print("="*80)
        
        # These are from your actual baseline results
        baseline_mean = 0.1156
        baseline_median = 0.2236
        baseline_hotels = 22
        
        print(f"\nYOUR BASELINE:")
        print(f"  Mean R2: {baseline_mean:.4f}")
        print(f"  Median R2: {baseline_median:.4f}")
        print(f"  Hotels: {baseline_hotels}")
        
        print(f"\nADAPTIVE METHOD:")
        print(f"  Mean R2: {mean_r2:.4f}")
        print(f"  Median R2: {median_r2:.4f}")
        print(f"  Hotels: {len(successful_results)}")
        
        improvement_mean = ((mean_r2 - baseline_mean) / baseline_mean) * 100
        improvement_median = ((median_r2 - baseline_median) / baseline_median) * 100
        
        print(f"\nIMPROVEMENT:")
        print(f"  Mean: {improvement_mean:+.1f}%")
        print(f"  Median: {improvement_median:+.1f}%")
    
    # COMPLETION
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nEach hotel got its optimal configuration")
    print("="*80)
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
