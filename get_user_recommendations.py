#!/usr/bin/env python3
"""Interactive script to get recommendations for a specific user."""

import pandas as pd
import numpy as np
import pickle
import sys
from src.retrieval.main import CandidateGenerationPipeline

def get_recommendations_for_user(user_id, top_k=10):
    """Get personalized recommendations for a specific user."""
    
    print(f'🔍 Getting recommendations for User {user_id}...')
    
    # Initialize the pipeline
    pipeline = CandidateGenerationPipeline()
    pipeline.load_data()
    
    # Load the trained ALS model
    with open('models/phase3/als_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Load mappings
    with open('data/processed/user_mapping.pkl', 'rb') as f:
        user_mapping = pickle.load(f)
    with open('data/processed/item_mapping.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    
    # Reverse mappings for display
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    
    # Check if user exists
    if user_id not in user_mapping:
        print(f'❌ User {user_id} not found in the dataset.')
        return
    
    user_idx = user_mapping[user_id]
    
    # Load data for display
    users_df = pd.read_parquet('data/processed/users_cleaned.parquet')
    products_df = pd.read_parquet('data/processed/products_cleaned.parquet')
    interactions_df = pd.read_parquet('data/processed/interactions_cleaned.parquet')
    
    # Get user info
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    
    print()
    print('=' * 80)
    print(f'👤 USER PROFILE: {user_id}')
    print('=' * 80)
    print(f'Age: {user_info["age"]}')
    print(f'Gender: {user_info["gender"]}')
    print(f'Location: {user_info["location"]}')
    print(f'Income Level: {user_info["income_level"]}')
    print(f'Device Type: {user_info["device_type"]}')
    print(f'Preference Category: {user_info["preference_category"]}')
    
    # Get user's interaction history
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    print()
    print(f'📊 INTERACTION HISTORY: {len(user_interactions)} total interactions')
    
    if len(user_interactions) > 0:
        # Group by interaction type
        interaction_summary = user_interactions['interaction_type'].value_counts()
        print('   Interaction breakdown:')
        for interaction_type, count in interaction_summary.items():
            print(f'   - {interaction_type}: {count}')
        
        print()
        print('   Recent interactions:')
        recent_interactions = user_interactions.sort_values('timestamp', ascending=False).head(5)
        for _, interaction in recent_interactions.iterrows():
            product_info = products_df[products_df['product_id'] == interaction['product_id']].iloc[0]
            print(f'   - {interaction["interaction_type"].title()}: {product_info["name"]} ({product_info["category"]}) - ${product_info["price"]:.2f}')
    
    # Generate recommendations
    user_factors = model_data['user_factors'][user_idx]
    item_factors = model_data['item_factors']
    
    # Calculate scores for all items
    scores = np.dot(item_factors, user_factors)
    
    # Get items user hasn't interacted with
    user_items = set(user_interactions['product_id'].values)
    
    print()
    print('=' * 80)
    print(f'🎁 TOP {top_k} PERSONALIZED RECOMMENDATIONS')
    print('=' * 80)
    
    recommendations = []
    
    for item_idx in np.argsort(scores)[::-1]:
        item_id = reverse_item_mapping.get(item_idx)
        if item_id and item_id not in user_items:
            product_info = products_df[products_df['product_id'] == item_id].iloc[0]
            recommendations.append({
                'product_id': item_id,
                'name': product_info['name'],
                'category': product_info['category'],
                'price': product_info['price'],
                'rating': product_info['rating'],
                'score': scores[item_idx],
                'brand': product_info['brand'],
                'description': product_info['description'][:100] + '...'
            })
            if len(recommendations) >= top_k:
                break
    
    for i, rec in enumerate(recommendations, 1):
        print()
        print(f'{i:2d}. {rec["name"]}')
        print(f'    Category: {rec["category"]} | Brand: {rec["brand"]}')
        print(f'    Price: ${rec["price"]:.2f} | Rating: {rec["rating"]:.1f}/5 | ML Score: {rec["score"]:.3f}')
        print(f'    Description: {rec["description"]}')
    
    print()
    print('=' * 80)
    print('🎯 RECOMMENDATION INSIGHTS')
    print('=' * 80)
    
    # Analyze recommendations by category
    rec_categories = {}
    for rec in recommendations:
        category = rec['category']
        rec_categories[category] = rec_categories.get(category, 0) + 1
    
    print('📊 Recommended categories:')
    for category, count in sorted(rec_categories.items(), key=lambda x: x[1], reverse=True):
        print(f'   - {category}: {count} items')
    
    # Price range analysis
    prices = [rec['price'] for rec in recommendations]
    print()
    print(f'💰 Price range: ${min(prices):.2f} - ${max(prices):.2f} (avg: ${np.mean(prices):.2f})')
    
    # Rating analysis
    ratings = [rec['rating'] for rec in recommendations]
    print(f'⭐ Average rating: {np.mean(ratings):.1f}/5')
    
    return recommendations

def main():
    """Main function to handle user input."""
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
            top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            get_recommendations_for_user(user_id, top_k)
        except ValueError:
            print("❌ Please provide a valid user ID (integer)")
    else:
        # Test with a random user
        print("🎲 No user ID provided, testing with random users...")
        
        # Load user mapping to get valid user IDs
        with open('data/processed/user_mapping.pkl', 'rb') as f:
            user_mapping = pickle.load(f)
        
        # Test with 2 random users
        test_users = np.random.choice(list(user_mapping.keys()), size=2, replace=False)
        
        for user_id in test_users:
            get_recommendations_for_user(user_id, top_k=5)
            print('\n' + '='*100 + '\n')

if __name__ == "__main__":
    main()
