#!/usr/bin/env python3
"""Test optimized ALS training performance with MLflow integration."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import after adding to path
from src.retrieval.models.als_model_optimized import OptimizedALSModel, compare_performance, list_experiments
from scipy.sparse import load_npz
import pickle
import json

def test_optimized_training():
    """Test the optimized ALS training with MLflow."""
    print("🚀 Testing Optimized ALS Training with MLflow")
    print("=" * 50)
    
    # Load data
    data_dir = Path("data/processed")
    
    print("📊 Loading data...")
    user_item_matrix = load_npz(data_dir / "user_item_matrix.npz")
    
    with open(data_dir / "user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    
    with open(data_dir / "item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)
    
    print(f"✅ Matrix loaded: {user_item_matrix.shape}")
    print(f"✅ Users: {len(user_mapping):,}")
    print(f"✅ Items: {len(item_mapping):,}")
    print(f"✅ Interactions: {user_item_matrix.nnz:,}")
    
    # Test parameters
    n_factors = 100
    n_iterations = 15  # Reduced for testing
    
    print(f"\n🎯 Training Parameters:")
    print(f"   Factors: {n_factors}")
    print(f"   Iterations: {n_iterations}")
    print(f"   Matrix: {user_item_matrix.shape[0]:,} × {user_item_matrix.shape[1]:,}")
    
    # Test optimized model with MLflow
    print(f"\n⚡ Training Optimized Model with MLflow...")
    print("-" * 30)
    
    optimized_model = OptimizedALSModel(
        n_factors=n_factors,
        n_iterations=n_iterations,
        batch_size=1000,
        use_blas=True,
        experiment_name="als_optimization_test",
        run_name=f"test_run_{int(time.time())}"
    )
    
    start_time = time.time()
    optimized_model.fit(user_item_matrix)
    total_time = time.time() - start_time
    
    print(f"\n🎉 Training Completed!")
    print(f"   ⚡ Total Time: {total_time:.2f}s")
    print(f"   📊 Final Loss: {optimized_model.training_loss[-1]:.4f}")
    print(f"   📈 Loss History: {[f'{loss:.4f}' for loss in optimized_model.training_loss[-5:]]}")
    
    # Performance analysis
    avg_iter_time = sum(optimized_model.training_times) / len(optimized_model.training_times)
    print(f"   ⏱️  Average Iteration: {avg_iter_time:.2f}s")
    print(f"   🚀 Fastest Iteration: {min(optimized_model.training_times):.2f}s")
    print(f"   🐌 Slowest Iteration: {max(optimized_model.training_times):.2f}s")
    
    # Save optimized model
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "als_model_optimized.pkl"
    optimized_model.save_model(str(model_path))
    print(f"\n💾 Optimized model saved to: {model_path}")
    
    # End MLflow run
    optimized_model.end_run()
    
    # Show MLflow experiments
    print(f"\n🔍 MLflow Experiments:")
    list_experiments()
    
    return optimized_model

def run_performance_comparison():
    """Run performance comparison between different model configurations."""
    print("\n🔍 Running Performance Comparison...")
    print("=" * 50)
    
    data_dir = Path("data/processed")
    user_item_matrix = load_npz(data_dir / "user_item_matrix.npz")
    
    # Compare different configurations
    results = compare_performance(
        user_item_matrix, 
        n_factors=100, 
        n_iterations=10, 
        experiment_name="als_config_comparison"
    )
    
    return results

if __name__ == "__main__":
    try:
        print("🚀 Starting Optimized ALS Training Test...")
        
        # Test single model training
        model = test_optimized_training()
        print(f"\n✅ Single model training completed successfully!")
        
        # Ask user if they want to run comparison
        print(f"\n🔍 Would you like to run performance comparison? (y/n)")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            results = run_performance_comparison()
            print(f"\n✅ Performance comparison completed!")
        else:
            print(f"\n⏭️  Skipping performance comparison.")
        
        print(f"\n🎯 Phase 3: Candidate Generation is ready!")
        print(f"📊 Model saved: data/processed/als_model_optimized.pkl")
        print(f"🔍 Check MLflow UI for experiment tracking!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
