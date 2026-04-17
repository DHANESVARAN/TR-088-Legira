"""
MODEL COMPARISON SCRIPT
Trains both Similar and Improved models, compares performance metrics and generates reports
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Import the existing prioritization engine
from prioritization_engine import (
    load_and_merge,
    load_statute_lookup,
    engineer_features,
    augment_demo_cases,
)

# Import our models
from model_similar import SimilarModel
from model_improved import ImprovedModel


def prepare_data(data_dir: str = "data", statute_file: str = None, augment_demo: bool = False) -> pd.DataFrame:
    """Prepare and engineer features from raw data"""
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    # Load and merge datasets
    print("Loading datasets...")
    merged = load_and_merge(data_dir)
    print(f"  ✓ Loaded {len(merged)} cases")
    
    # Augment with demo cases if requested
    if augment_demo:
        print("Augmenting with synthetic demo cases...")
        merged = augment_demo_cases(merged)
        print(f"  ✓ Total cases after augmentation: {len(merged)}")
    
    # Load statute lookup
    if statute_file is None:
        statute_file = os.path.join(data_dir, "ipc_bns_max_sentence_lookup.csv")
    
    statute_lookup = None
    if statute_file and os.path.exists(statute_file):
        print(f"Loading statute lookup from {statute_file}...")
        statute_lookup = load_statute_lookup(statute_file)
        print(f"  ✓ Loaded {len(statute_lookup)} statutes")
    
    # Engineer features
    print("Engineering features...")
    featured = engineer_features(merged, statute_lookup)
    print(f"  ✓ Feature engineering complete")
    
    return featured


def train_model(model_instance, df: pd.DataFrame, name: str) -> dict:
    """Train a model and return results"""
    print(f"\n{'='*70}")
    print(f"TRAINING {name.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Train classification models
    print(f"Training classification models...")
    df_with_predictions = model_instance.train(df)
    
    print(f"Clustering cases...")
    # Add clustering
    df_with_predictions = model_instance.cluster(df_with_predictions)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{name} Metrics:")
    for key, value in model_instance.metrics.items():
        if not isinstance(value, dict):
            print(f"  • {key}: {value:.4f}" if isinstance(value, float) else f"  • {key}: {value}")
    
    print(f"Training time: {elapsed_time:.2f} seconds")
    
    return {
        "name": name,
        "model": model_instance,
        "dataframe": df_with_predictions,
        "metrics": model_instance.metrics,
        "training_time": elapsed_time,
    }


def compare_models(similar_result: dict, improved_result: dict, output_dir: str = "comparison_outputs"):
    """Compare two model results and generate comparison report"""
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON ANALYSIS")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    similar_metrics = similar_result["metrics"]
    improved_metrics = improved_result["metrics"]
    
    # Create comparison dataframe
    comparison_data = {
        "Metric": [],
        "Similar Model": [],
        "Improved Model": [],
        "Improvement": [],
        "Winner": [],
    }
    
    # Compare key metrics
    key_metrics = [
        "urgency_classification_accuracy",
        "urgency_classification_f1",
        "track_classification_accuracy",
        "track_classification_f1",
        "undertrial_detection_recall",
    ]
    
    print("\nPerformance Comparison:\n")
    print(f"{'Metric':<40} {'Similar':<15} {'Improved':<15} {'Winner':<15}")
    print("-" * 85)
    
    for metric in key_metrics:
        similar_val = similar_metrics.get(metric, 0)
        improved_val = improved_metrics.get(metric, 0)
        
        if isinstance(similar_val, (int, float)) and isinstance(improved_val, (int, float)):
            improvement = ((improved_val - similar_val) / abs(similar_val) * 100) if similar_val != 0 else 0
            winner = "IMPROVED ✓" if improved_val > similar_val else "SIMILAR" if similar_val > improved_val else "TIE"
            
            comparison_data["Metric"].append(metric)
            comparison_data["Similar Model"].append(f"{similar_val:.4f}")
            comparison_data["Improved Model"].append(f"{improved_val:.4f}")
            comparison_data["Improvement"].append(f"{improvement:+.2f}%")
            comparison_data["Winner"].append(winner)
            
            print(f"{metric:<40} {similar_val:<15.4f} {improved_val:<15.4f} {winner:<15}")
    
    # Training time comparison
    print(f"\n{'Training Time':<40} {similar_result['training_time']:<15.2f}s {improved_result['training_time']:<15.2f}s", end="")
    speedup = similar_result['training_time'] / improved_result['training_time']
    print(f" {speedup:.2f}x speedup" if speedup < 1 else f" (Improved is {1/speedup:.2f}x slower)")
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison saved to {comparison_path}")
    
    # Save detailed metrics
    metrics_comparison = {
        "similar_model": similar_metrics,
        "improved_model": improved_metrics,
        "comparison_summary": comparison_data,
    }
    metrics_path = os.path.join(output_dir, "detailed_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_comparison, f, indent=2)
    print(f"✓ Detailed metrics saved to {metrics_path}")
    
    # Analyze predictions
    print(f"\n{'='*70}")
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    print("Similar Model Urgency Predictions:")
    print(similar_result["dataframe"]["predicted_urgency"].value_counts())
    
    print("\nImproved Model Urgency Predictions:")
    print(improved_result["dataframe"]["predicted_urgency"].value_counts())
    
    # Calculate agreement
    agreement = (similar_result["dataframe"]["predicted_urgency"] == 
                 improved_result["dataframe"]["predicted_urgency"]).mean()
    print(f"\nPrediction Agreement: {agreement*100:.2f}%")
    
    # Save sample predictions
    sample_size = min(100, len(similar_result["dataframe"]))
    sample_cases = pd.DataFrame({
        "case_id": similar_result["dataframe"]["case_id"].head(sample_size),
        "priority_score": similar_result["dataframe"]["priority_score"].head(sample_size),
        "similar_urgency": similar_result["dataframe"]["predicted_urgency"].head(sample_size),
        "improved_urgency": improved_result["dataframe"]["predicted_urgency"].head(sample_size),
        "similar_track": similar_result["dataframe"]["predicted_case_type"].head(sample_size),
        "improved_track": improved_result["dataframe"]["predicted_case_type"].head(sample_size),
        "similar_cluster": similar_result["dataframe"]["cluster_label"].head(sample_size),
        "improved_cluster": improved_result["dataframe"]["cluster_label"].head(sample_size),
    })
    
    sample_path = os.path.join(output_dir, "sample_predictions.csv")
    sample_cases.to_csv(sample_path, index=False)
    print(f"\n✓ Sample predictions saved to {sample_path}")
    
    return comparison_df


def main():
    """Main execution function"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  LEGAL CASE PRIORITIZATION: MODEL COMPARISON".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Configuration
    data_dir = "data"
    statute_file = os.path.join(data_dir, "ipc_bns_max_sentence_lookup.csv")
    output_dir = "comparison_outputs"
    augment_demo = True
    
    # Verify data exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Error: Data directory '{data_dir}' not found")
        sys.exit(1)
    
    # Prepare data
    featured_df = prepare_data(data_dir, statute_file, augment_demo)
    
    # Train Similar Model
    similar_model = SimilarModel()
    similar_result = train_model(similar_model, featured_df.copy(), "Similar Model")
    
    # Train Improved Model
    improved_model = ImprovedModel()
    improved_result = train_model(improved_model, featured_df.copy(), "Improved Model")
    
    # Compare models
    comparison_df = compare_models(similar_result, improved_result, output_dir)
    
    # Save models
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")
    
    similar_model.save_models(os.path.join(output_dir, "similar_models"))
    improved_model.save_models(os.path.join(output_dir, "improved_models"))
    
    print("\n✓ Similar model saved to comparison_outputs/similar_models/")
    print("✓ Improved model saved to comparison_outputs/improved_models/")
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to '{output_dir}' directory:")
    print(f"  • model_comparison.csv - Side-by-side metrics")
    print(f"  • detailed_metrics.json - Full metrics and reports")
    print(f"  • sample_predictions.csv - Sample case predictions")
    print(f"  • similar_models/ - Saved Similar model artifacts")
    print(f"  • improved_models/ - Saved Improved model artifacts")


if __name__ == "__main__":
    main()
