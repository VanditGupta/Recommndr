#!/usr/bin/env python3
"""Comprehensive ALS Model Validation and Testing."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pickle
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.models.als_model_optimized import OptimizedALSModel
from retrieval.similarity.faiss_search import FaissSimilaritySearch

def test_model_sanity(model, user_item_matrix):
    """Basic sanity checks for the trained model."""
    print("🔍 **MODEL SANITY CHECKS**")
    print("=" * 50)
    
    # Check 1: Model dimensions
    n_users, n_items = user_item_matrix.shape
    print(f"📊 Matrix dimensions: {n_users} × {n_items}")
    print(f"   User factors: {model.user_factors.shape}")
    print(f"   Item factors: {model.item_factors.shape}")
    
    # Check 2: Factor ranges
    print(f"\n📈 Factor statistics:")
    print(f"   User factors range: [{model.user_factors.min():.4f}, {model.user_factors.max():.4f}]")
    print(f"   Item factors range: [{model.item_factors.min():.4f}, {model.item_factors.max():.4f}]")
    print(f"   User factors mean: {model.user_factors.mean():.4f}")
    print(f"   Item factors mean: {model.item_factors.mean():.4f}")
    
    # Check 3: Bias values
    print(f"\n⚖️  Bias statistics:")
    print(f"   Global bias: {model.global_bias:.4f}")
    print(f"   User biases range: [{model.user_biases.min():.4f}, {model.user_biases.max():.4f}]")
    print(f"   Item biases range: [{model.item_biases.min():.4f}, {model.item_biases.max():.4f}]")
    
    # Check 4: Training convergence
    print(f"\n📉 Training convergence:")
    print(f"   Final loss: {model.training_loss[-1]:.4f}")
    print(f"   Loss improvement: {model.training_loss[0] - model.training_loss[-1]:.4f}")
    print(f"   Loss stable: {'Yes' if len(set(model.training_loss[-3:])) < 3 else 'No'}")
    
    return True

def test_prediction_accuracy(model, user_item_matrix, n_samples=100):
    """Test prediction accuracy on known interactions."""
    print(f"\n🎯 **PREDICTION ACCURACY TEST**")
    print("=" * 50)
    
    # Sample some user-item pairs with known interactions
    user_indices, item_indices = user_item_matrix.nonzero()
    
    if len(user_indices) == 0:
        print("❌ No interactions found in matrix")
        return False
    
    # Sample random interactions
    sample_indices = np.random.choice(len(user_indices), min(n_samples, len(user_indices)), replace=False)
    sample_users = user_indices[sample_indices]
    sample_items = item_indices[sample_indices]
    
    # Get actual ratings - handle sparse matrix properly
    actual_ratings = []
    for user_id, item_id in zip(sample_users, sample_items):
        rating = user_item_matrix[user_id, item_id]
        if hasattr(rating, 'toarray'):
            actual_ratings.append(rating.toarray()[0, 0])
        else:
            actual_ratings.append(float(rating))
    
    actual_ratings = np.array(actual_ratings)
    
    # Get predictions
    predictions = []
    for user_id, item_id in zip(sample_users, sample_items):
        pred = model.predict(user_id, np.array([item_id]))[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = np.mean((actual_ratings - predictions) ** 2)
    mae = np.mean(np.abs(actual_ratings - predictions))
    correlation = np.corrcoef(actual_ratings, predictions)[0, 1]
    
    print(f"📊 Prediction Metrics (on {len(sample_users)} samples):")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Correlation: {correlation:.4f}")
    
    # Show some examples
    print(f"\n📋 Sample Predictions:")
    for i in range(min(5, len(sample_users))):
        print(f"   User {sample_users[i]}, Item {sample_items[i]}: "
              f"Actual={actual_ratings[i]:.2f}, Predicted={predictions[i]:.2f}")
    
    return mse < 10.0  # Reasonable threshold

def test_recommendation_quality(model, user_item_matrix, n_users=10):
    """Test recommendation quality and diversity."""
    print(f"\n🎁 **RECOMMENDATION QUALITY TEST**")
    print("=" * 50)
    
    # Initialize similarity search
    faiss_search = FaissSimilaritySearch(use_faiss=False)
    item_embeddings = model.get_item_embeddings()
    faiss_search.build_index(item_embeddings)
    
    # Test recommendations for random users
    user_indices = np.random.choice(user_item_matrix.shape[0], min(n_users, user_item_matrix.shape[0]), replace=False)
    
    total_diversity = 0
    total_relevance = 0
    
    for i, user_id in enumerate(user_indices):
        print(f"\n👤 User {user_id}:")
        
        # Get user embedding
        user_embedding = model.get_user_embeddings([user_id])[0]
        
        # Get recommendations
        distances, item_indices = faiss_search.search_similar(user_embedding, k=10)
        
        # Check diversity (unique items)
        diversity = len(set(item_indices))
        total_diversity += diversity
        
        # Check relevance (items user has interacted with)
        user_items = set(user_item_matrix[user_id].nonzero()[1])
        recommended_items = set(item_indices)
        overlap = len(user_items.intersection(recommended_items))
        relevance = overlap / len(recommended_items) if len(recommended_items) > 0 else 0
        total_relevance += relevance
        
        print(f"   Recommendations: {item_indices[:5]}...")
        print(f"   Diversity: {diversity}/10 unique items")
        print(f"   Relevance: {overlap}/{len(recommended_items)} items user interacted with")
    
    avg_diversity = total_diversity / len(user_indices)
    avg_relevance = total_relevance / len(user_indices)
    
    print(f"\n📊 Overall Quality Metrics:")
    print(f"   Average Diversity: {avg_diversity:.2f}/10")
    print(f"   Average Relevance: {avg_relevance:.2f}")
    
    return avg_diversity > 7.0 and avg_relevance < 0.5  # Good diversity, not too much overlap

def test_model_consistency(model, user_item_matrix):
    """Test model consistency and stability."""
    print(f"\n🔄 **MODEL CONSISTENCY TEST**")
    print("=" * 50)
    
    # Test 1: Same user, same items should give same predictions
    user_id = 0
    item_ids = np.array([0, 1, 2])
    
    pred1 = model.predict(user_id, item_ids)
    pred2 = model.predict(user_id, item_ids)
    
    consistency = np.allclose(pred1, pred2)
    print(f"✅ Prediction consistency: {'Yes' if consistency else 'No'}")
    
    # Test 2: Embedding consistency
    user_emb1 = model.get_user_embeddings([user_id])
    user_emb2 = model.get_user_embeddings([user_id])
    
    emb_consistency = np.allclose(user_emb1, user_emb2)
    print(f"✅ Embedding consistency: {'Yes' if emb_consistency else 'No'}")
    
    # Test 3: Factor norms (should be reasonable)
    user_norms = np.linalg.norm(model.user_factors, axis=1)
    item_norms = np.linalg.norm(model.item_factors, axis=1)
    
    print(f"📏 Factor norms:")
    print(f"   User factors: mean={user_norms.mean():.4f}, std={user_norms.std():.4f}")
    print(f"   Item factors: mean={item_norms.mean():.4f}, std={item_norms.std():.4f}")
    
    # Check for reasonable ranges
    reasonable_norms = (user_norms.mean() < 2.0 and item_norms.mean() < 2.0)
    print(f"✅ Reasonable factor norms: {'Yes' if reasonable_norms else 'No'}")
    
    return consistency and emb_consistency and reasonable_norms

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🚀 **COMPREHENSIVE ALS MODEL VALIDATION**")
    print("=" * 60)
    
    # Load data and model
    print("📂 Loading data and model...")
    user_item_matrix = load_npz('data/processed/user_item_matrix.npz')
    
    # Try to load existing model, or train new one
    model_path = Path('data/processed/als_full_test_2.pkl')
    if model_path.exists():
        print(f"📂 Loading existing model: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = OptimizedALSModel(n_factors=50)
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_biases = model_data['user_biases']
        model.item_biases = model_data['item_biases']
        model.global_bias = model_data['global_bias']
        model.training_loss = model_data.get('training_loss', [])
        model.training_times = model_data.get('training_times', [])
        model.is_trained = True
    else:
        print("🔄 Training new model for validation...")
        model = OptimizedALSModel(n_factors=50, n_iterations=15, experiment_name='validation_test')
        model.fit(user_item_matrix)
    
    print(f"✅ Model loaded: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} items")
    
    # Run all validation tests
    results = {}
    
    results['sanity'] = test_model_sanity(model, user_item_matrix)
    results['accuracy'] = test_prediction_accuracy(model, user_item_matrix)
    results['quality'] = test_recommendation_quality(model, user_item_matrix)
    results['consistency'] = test_model_consistency(model, user_item_matrix)
    
    # Summary
    print(f"\n🏆 **VALIDATION SUMMARY**")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.upper()}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 **MODEL VALIDATION SUCCESSFUL!**")
        print("   Your ALS model is working correctly!")
    else:
        print("⚠️  **MODEL VALIDATION FAILED**")
        print("   Some issues detected. Check the details above.")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_validation()
        print(f"\n🎯 Validation completed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
