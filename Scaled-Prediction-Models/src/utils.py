import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = 'logs', log_level: str = 'INFO'):
    """
    Configure logging to both file and console
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'pipeline_{timestamp}.log'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")
    
    return log_file


def ensure_directory(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def print_summary_stats(results: dict):
    """
    Print summary statistics from training results
    
    Args:
        results: Dictionary of hotel results
    """
    test_r2_values = [r['test_r2'] for r in results.values()]
    
    print("\nPerformance Summary:")
    print(f"  Mean Test R2: {np.mean(test_r2_values):.4f}")
    print(f"  Median Test R2: {np.median(test_r2_values):.4f}")
    print(f"  Std Test R2: {np.std(test_r2_values):.4f}")
    print(f"  Min Test R2: {np.min(test_r2_values):.4f}")
    print(f"  Max Test R2: {np.max(test_r2_values):.4f}")
    
    # Count by performance bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts, _ = np.histogram(test_r2_values, bins=bins)
    
    print("\nR2 Distribution:")
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {counts[i]} hotels")


def print_configuration_analysis(results: dict):
    """
    Analyze which configurations were selected
    
    Args:
        results: Dictionary of hotel results
    """
    # FIXED: Extract configuration parameters directly (not nested in 'best_config')
    trend_windows = [r['trend_window'] for r in results.values()]
    train_sizes = [r['min_train_days'] for r in results.values()]
    feature_sets = [r['feature_set'] for r in results.values()]
    
    print("\nConfiguration Analysis:")
    print("\nTrend Windows Selected:")
    for window in sorted(set(trend_windows)):
        count = trend_windows.count(window)
        pct = (count / len(trend_windows)) * 100
        print(f"  {window}: {count} hotels ({pct:.1f}%)")
    
    print("\nTrain Sizes Selected:")
    for size in sorted(set(train_sizes)):
        count = train_sizes.count(size)
        pct = (count / len(train_sizes)) * 100
        print(f"  {size}: {count} hotels ({pct:.1f}%)")
    
    print("\nFeature Sets Selected:")
    for fset in sorted(set(feature_sets)):
        count = feature_sets.count(fset)
        pct = (count / len(feature_sets)) * 100
        print(f"  {fset}: {count} hotels ({pct:.1f}%)")