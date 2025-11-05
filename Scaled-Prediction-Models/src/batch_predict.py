import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the predictor
from model_prediction import HotelPricePredictor, load_config


def get_available_hotels():
    """Get list of hotels with trained models"""
    # Find hotel mapping
    mapping_paths = [
        Path('data/full-data/hotel_mapping.csv'),
        Path('../data/full-data/hotel_mapping.csv'),
    ]
    
    mapping_file = None
    for path in mapping_paths:
        if path.exists():
            mapping_file = path
            break
    
    if mapping_file is None:
        print("ERROR: Hotel mapping not found")
        return None
    
    # Load mapping
    mapping = pd.read_csv(mapping_file)
    
    # Find trained models
    models_paths = [
        Path('models/adaptive_log_method'),
        Path('../models/adaptive_log_method'),
    ]
    
    models_dir = None
    for path in models_paths:
        if path.exists():
            models_dir = path
            break
    
    if models_dir is None:
        print("ERROR: Models directory not found")
        return None
    
    # Get all model files
    model_files = list(models_dir.glob('Hotel_*_model.pkl'))
    trained_hotels = [f.stem.replace('_model', '') for f in model_files]
    
    # Filter mapping to only trained hotels
    trained_mapping = mapping[mapping['masked_id'].isin(trained_hotels)].copy()
    trained_mapping = trained_mapping.sort_values('actual_id')
    
    return trained_mapping


def list_available_hotels():
    """Display all available hotels with trained models"""
    trained_mapping = get_available_hotels()
    
    if trained_mapping is None:
        return
    
    print("\n" + "=" * 80)
    print(f"AVAILABLE HOTELS WITH TRAINED MODELS ({len(trained_mapping)} total)")
    print("=" * 80)
    print(f"\n{'Hotel Name':<50} {'Status'}")
    print("-" * 80)
    
    for _, row in trained_mapping.iterrows():
        print(f"{row['actual_id']:<50} ✓ Trained")
    
    print("-" * 80)
    print("\nYou can use these hotel names with --hotels option")
    print("Example: --hotels \"Ozarker_Lodge\" \"Another_Hotel_Name\"")
    print("=" * 80 + "\n")


def batch_predict(hotels, prediction_date, output_dir='predictions', config_path='config.yaml'):
    """
    Make predictions for multiple hotels and save results
    
    Args:
        hotels: List of hotel names or IDs
        prediction_date: Date to predict for (YYYY-MM-DD)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("=" * 80)
    print("BATCH PREDICTION")
    print("=" * 80)
    print(f"Prediction date: {prediction_date}")
    print(f"Hotels to test: {len(hotels)}")
    print("=" * 80)
    
    # Load config
    config = load_config(config_path)
    db_config = {
        'host': config['database']['host'],
        'port': config['database']['port'],
        'database': config['database']['database'],
        'user': config['database']['user'],
        'password': config['database']['password']
    }
    
    # Initialize predictor
    predictor = HotelPricePredictor(
        models_dir='models/adaptive_log_method',
        db_config=db_config,
        imputation_window_days=60
    )
    
    # Connect to database
    predictor.connect_db()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    successful = 0
    failed = 0
    
    try:
        for i, hotel in enumerate(hotels, 1):
            print(f"\n[{i}/{len(hotels)}] Predicting for {hotel}...")
            print("-" * 80)
            
            # Make prediction
            result = predictor.predict(
                hotel,
                prediction_date,
                return_details=True
            )
            
            if result and result.get('success'):
                results.append(result)
                successful += 1
                print(f"SUCCESS: ${result['predicted_base_rate']:.2f}")
            else:
                failed += 1
                results.append({
                    'success': False,
                    'hotel_name': hotel,
                    'date': prediction_date,
                    'reason': 'Prediction failed'
                })
                print("FAILED")
            
            print("-" * 80)
        
        # Save individual results as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = output_path / f'batch_predictions_{timestamp}.json'
        
        with open(json_file, 'w') as f:
            json.dump({
                'prediction_date': prediction_date,
                'timestamp': timestamp,
                'total_hotels': len(hotels),
                'successful': successful,
                'failed': failed,
                'predictions': results
            }, f, indent=2)
        
        print(f"\nJSON results saved to: {json_file}")
        
        # Save as CSV (easier for analysis)
        if successful > 0:
            successful_results = [r for r in results if r.get('success')]
            
            df_results = pd.DataFrame([
                {
                    'hotel_name': r['hotel_name'],
                    'prediction_date': r['date'],
                    'predicted_base_rate': r['predicted_base_rate'],
                    'trend_window': r['model_config']['trend_window'],
                    'feature_set': r['model_config']['feature_set'],
                    'n_features': r['model_config']['n_features'],
                    'records_fetched': r['data_stats']['records_fetched'],
                    'competitors': r['data_stats']['competitors'],
                    'missing_pct': r['data_stats']['missing_pct_before_imputation']
                }
                for r in successful_results
            ])
            
            csv_file = output_path / f'batch_predictions_{timestamp}.csv'
            df_results.to_csv(csv_file, index=False)
            
            print(f"CSV results saved to: {csv_file}")
            
            # Print summary table
            print("\n" + "=" * 80)
            print("PREDICTION SUMMARY")
            print("=" * 80)
            print(df_results.to_string(index=False))
            print("=" * 80)
        
        # Print final summary
        print(f"\nBatch prediction complete:")
        print(f"  Total: {len(hotels)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        if successful > 0:
            avg_price = sum(r['predicted_base_rate'] for r in results if r.get('success')) / successful
            print(f"  Average predicted rate: ${avg_price:.2f}")
    
    finally:
        predictor.disconnect_db()
    
    return results


def main():
    """Run batch predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Hotel Price Predictions')
    parser.add_argument('--date', help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--hotels', nargs='+', help='List of hotel names')
    parser.add_argument('--all-trained', action='store_true', 
                       help='Predict for all hotels with trained models')
    parser.add_argument('--list', action='store_true',
                       help='List all available hotels with trained models')
    parser.add_argument('--output-dir', default='predictions',
                       help='Output directory for results')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # If --list, show available hotels and exit
    if args.list:
        list_available_hotels()
        return
    
    # Check if date is provided
    if not args.date:
        print("ERROR: --date is required (unless using --list)")
        print("\nUsage:")
        print("  List hotels:    python batch_predict.py --list")
        print("  Predict:        python batch_predict.py --date 2026-01-15 --all-trained")
        return
    
    # Determine which hotels to predict
    if args.all_trained:
        # Get all hotels with trained models
        trained_mapping = get_available_hotels()
        if trained_mapping is None:
            return
        
        hotels = trained_mapping['actual_id'].tolist()
        
        print(f"Found {len(hotels)} trained models")
    elif args.hotels:
        hotels = args.hotels
    else:
        print("ERROR: Must specify --hotels or --all-trained")
        print("\nTip: Use --list to see all available hotels")
        return
    
    # Run batch predictions
    batch_predict(
        hotels=hotels,
        prediction_date=args.date,
        output_dir=args.output_dir,
        config_path=args.config
    )


if __name__ == "__main__":
    main()