import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not installed. Install with: pip install shap --break-system-packages")


def load_model_and_data(hotel_id, model_path):
    """
    Load trained model with its preprocessed training data
    """
    try:
        model_file = model_path / f'{hotel_id}_model.pkl'
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if preprocessed data was saved
        if 'X_train' not in model_data or 'X_test' not in model_data:
            print(f"Preprocessed data not saved for {hotel_id}")
            return None
        
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        X_train = model_data['X_train']
        X_test = model_data['X_test']
        
        # Combine train and test data
        X = np.vstack([X_train, X_test])
        
        return model, X, feature_cols
        
    except Exception as e:
        print(f"Error loading {hotel_id}: {e}")
        return None


def calculate_shap_values(model, X, max_samples=1000):
    """
    Calculate SHAP values
    """
    try:
        # Limit samples for performance
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample, explainer
        
    except Exception as e:
        print(f"Error calculating SHAP: {e}")
        return None, None, None


def create_plots(hotel_id, shap_values, X_sample, feature_cols, explainer, output_dir):
    """
    Create SHAP plots
    """
    try:
        hotel_dir = output_dir / hotel_id
        hotel_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.title(f'SHAP Summary - {hotel_id}')
        plt.tight_layout()
        plt.savefig(hotel_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                         plot_type="bar", show=False)
        plt.title(f'Feature Importance - {hotel_id}')
        plt.tight_layout()
        plt.savefig(hotel_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample[0],
                feature_names=feature_cols
            ),
            show=False
        )
        plt.title(f'Prediction Explanation - {hotel_id}')
        plt.tight_layout()
        plt.savefig(hotel_dir / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        return False


def save_importance(hotel_id, shap_values, feature_cols, output_dir):
    """
    Save feature importance to CSV
    """
    try:
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': importance
        }).sort_values('mean_abs_shap', ascending=False)
        
        hotel_dir = output_dir / hotel_id
        df.to_csv(hotel_dir / 'feature_importance.csv', index=False)
        
        return df
        
    except Exception as e:
        print(f"Error saving importance: {e}")
        return None


def analyze_hotel(hotel_id, model_path, output_dir):
    """
    Complete SHAP analysis for one hotel
    """
    # Load model and data
    result = load_model_and_data(hotel_id, model_path)
    if result is None:
        return None
    
    model, X, feature_cols = result
    
    # Calculate SHAP values
    shap_values, X_sample, explainer = calculate_shap_values(model, X)
    if shap_values is None:
        return None
    
    # Create plots
    if not create_plots(hotel_id, shap_values, X_sample, feature_cols, explainer, output_dir):
        return None
    
    # Save importance
    importance_df = save_importance(hotel_id, shap_values, feature_cols, output_dir)
    if importance_df is None:
        return None
    
    return {
        'hotel_id': hotel_id,
        'n_features': len(feature_cols),
        'n_samples': len(X_sample),
        'top_feature': importance_df.iloc[0]['feature'],
        'top_importance': float(importance_df.iloc[0]['mean_abs_shap'])
    }


def main():
    """
    Main function
    """
    if not SHAP_AVAILABLE:
        print("\nERROR: SHAP not installed")
        print("Install with: pip install shap --break-system-packages")
        return
    
    print("SHAP Feature Importance Analysis")
    
    # Paths - relative to project root
    model_path = Path('models/adaptive_log_method')
    output_dir = Path('reports/shap_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading training results...")
    try:
        with open(model_path / 'adaptive_summary.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {model_path / 'adaptive_summary.json'}")
        print("Run main.py first to train models")
        return
    
    # Filter successful hotels
    successful = {h: r for h, r in results.items() if r.get('status') == 'success'}
    
    if not successful:
        print("No successfully trained hotels found")
        return
    
    print(f"Found {len(successful)} successfully trained hotels")
    print(f"Analyzing all {len(successful)} hotels")
    print("Estimated time: 5-10 minutes")
    
    # Process each hotel
    hotels = sorted(successful.keys())
    results = []
    
    for i, hotel_id in enumerate(hotels, 1):
        print(f"[{i}/{len(hotels)}] Processing {hotel_id}...", end=' ', flush=True)
        
        result = analyze_hotel(hotel_id, model_path, output_dir)
        
        if result:
            results.append(result)
            print("done")
        else:
            print("failed")
    
    # Save summary
    if results:
        df = pd.DataFrame(results)
        summary_file = output_dir / 'shap_analysis_summary.csv'
        df.to_csv(summary_file, index=False)
        
        print(f"\nSHAP analysis complete")
        print(f"Successfully analyzed: {len(results)}/{len(hotels)} hotels")
        
        print("\nTop features by hotel:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['hotel_id']}: {row['top_feature']} (importance: {row['top_importance']:.4f})")
        
        print(f"\nResults saved to: {output_dir}/")
        print("For each hotel generated:")
        print("  - shap_summary.png")
        print("  - shap_importance.png")
        print("  - shap_waterfall.png")
        print("  - feature_importance.csv")
        print(f"Summary file: {summary_file}")
    else:
        print("\nERROR: No hotels analyzed successfully")
        print("The models may not have saved preprocessed data")
        print("You may need to retrain models to save X_train and X_test")


if __name__ == "__main__":
    main()