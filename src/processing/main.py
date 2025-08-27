"""Main data processing pipeline for Recommndr."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from config.settings import settings, get_data_path, ensure_directories
from src.processing.cleaners import DataCleaner
from src.processing.feature_engineering import FeatureEngineer
from src.processing.transformers import DataTransformer
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def process_data(
    input_dir: str = None,
    output_dir: str = None,
    save_processed: bool = True,
    save_features: bool = True
) -> Dict[str, pd.DataFrame]:
    """Process and clean e-commerce data.
    
    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
        save_processed: Whether to save cleaned data
        save_features: Whether to save feature-engineered data
        
    Returns:
        Dictionary containing all processed data
    """
    start_time = time.time()
    
    # Set default directories
    if input_dir is None:
        input_dir = str(get_data_path("raw"))
    if output_dir is None:
        output_dir = str(get_data_path("processed"))
    
    # Ensure output directories exist
    ensure_directories()
    
    logger.info("Starting data processing pipeline", extra={
        "input_dir": input_dir,
        "output_dir": output_dir,
        "save_processed": save_processed,
        "save_features": save_features
    })
    
    try:
        # Load raw data
        logger.info("Loading raw data...")
        raw_data = load_raw_data(input_dir)
        
        # Initialize processing components
        cleaner = DataCleaner()
        feature_engineer = FeatureEngineer()
        transformer = DataTransformer()
        
        # Step 1: Data Cleaning
        logger.info("Step 1: Cleaning data...")
        cleaned_data = clean_data(raw_data, cleaner)
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Creating features...")
        feature_data = create_features(cleaned_data, feature_engineer)
        
        # Step 3: Data Transformation
        logger.info("Step 3: Transforming data...")
        transformed_data = transform_data(feature_data, transformer)
        
        # Save processed data if requested
        if save_processed:
            logger.info("Saving processed data...")
            save_processed_data(cleaned_data, output_dir)
        
        if save_features:
            logger.info("Saving feature data...")
            save_feature_data(feature_data, output_dir)
        
        # Log processing statistics
        processing_stats = {
            'cleaning': cleaner.get_cleaning_stats(),
            'features': feature_engineer.get_feature_stats(),
            'transformation': transformer.get_transformation_stats()
        }
        
        duration = time.time() - start_time
        logger.info("Data processing completed successfully", extra={
            "duration_seconds": round(duration, 2),
            "processing_stats": processing_stats
        })
        
        return {
            'raw': raw_data,
            'cleaned': cleaned_data,
            'features': feature_data,
            'transformed': transformed_data,
            'stats': processing_stats
        }
        
    except Exception as e:
        logger.error("Data processing failed", exc_info=True)
        raise


def load_raw_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load raw data from parquet files."""
    logger.info(f"Loading raw data from {data_dir}")
    
    data = {}
    
    # Load users
    users_path = Path(data_dir) / "users" / "users.parquet"
    if users_path.exists():
        data['users'] = pd.read_parquet(users_path)
        logger.info(f"Loaded {len(data['users'])} users")
    else:
        raise FileNotFoundError(f"Users file not found: {users_path}")
    
    # Load products
    products_path = Path(data_dir) / "products" / "products.parquet"
    if products_path.exists():
        data['products'] = pd.read_parquet(products_path)
        logger.info(f"Loaded {len(data['products'])} products")
    else:
        raise FileNotFoundError(f"Products file not found: {products_path}")
    
    # Load interactions
    interactions_path = Path(data_dir) / "interactions" / "interactions.parquet"
    if interactions_path.exists():
        data['interactions'] = pd.read_parquet(interactions_path)
        logger.info(f"Loaded {len(data['interactions'])} interactions")
    else:
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")
    
    # Load categories
    categories_path = Path(data_dir) / "categories" / "categories.parquet"
    if categories_path.exists():
        data['categories'] = pd.read_parquet(categories_path)
        logger.info(f"Loaded {len(data['categories'])} categories")
    else:
        raise FileNotFoundError(f"Categories file not found: {categories_path}")
    
    return data


def clean_data(raw_data: Dict[str, pd.DataFrame], cleaner: DataCleaner) -> Dict[str, pd.DataFrame]:
    """Clean all data using the data cleaner."""
    logger.info("Cleaning data...")
    
    cleaned_data = {}
    
    # Clean users
    cleaned_data['users'] = cleaner.clean_users(raw_data['users'])
    
    # Clean products
    cleaned_data['products'] = cleaner.clean_products(raw_data['products'])
    
    # Clean interactions
    cleaned_data['interactions'] = cleaner.clean_interactions(raw_data['interactions'])
    
    # Keep categories as-is (they're already clean)
    cleaned_data['categories'] = raw_data['categories']
    
    logger.info("Data cleaning completed")
    return cleaned_data


def create_features(cleaned_data: Dict[str, pd.DataFrame], feature_engineer: FeatureEngineer) -> Dict[str, pd.DataFrame]:
    """Create features for all data types."""
    logger.info("Creating features...")
    
    feature_data = {}
    
    # Create user features
    feature_data['users'] = feature_engineer.create_user_features(
        cleaned_data['users'], 
        cleaned_data['interactions']
    )
    
    # Create product features
    feature_data['products'] = feature_engineer.create_product_features(
        cleaned_data['products'], 
        cleaned_data['interactions']
    )
    
    # Create interaction features
    feature_data['interactions'] = feature_engineer.create_interaction_features(
        cleaned_data['interactions'],
        feature_data['users'],
        feature_data['products']
    )
    
    # Keep categories as-is
    feature_data['categories'] = cleaned_data['categories']
    
    logger.info("Feature creation completed")
    return feature_data


def transform_data(feature_data: Dict[str, pd.DataFrame], transformer: DataTransformer) -> Dict[str, pd.DataFrame]:
    """Transform data for ML models."""
    logger.info("Transforming data...")
    
    transformed_data = {}
    
    # Prepare user-item matrix
    user_item_matrix, user_features, product_features = transformer.prepare_user_item_matrix(
        feature_data['interactions'],
        feature_data['users'],
        feature_data['products']
    )
    
    # Create training data
    X, y = transformer.create_training_data(
        feature_data['interactions'],
        feature_data['users'],
        feature_data['products']
    )
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = transformer.create_validation_split(X, y)
    
    transformed_data['user_item_matrix'] = user_item_matrix
    transformed_data['user_features'] = user_features
    transformed_data['product_features'] = product_features
    transformed_data['X_train'] = X_train
    transformed_data['X_val'] = X_val
    transformed_data['y_train'] = y_train
    transformed_data['y_val'] = y_val
    
    logger.info("Data transformation completed")
    return transformed_data


def save_processed_data(cleaned_data: Dict[str, pd.DataFrame], output_dir: str):
    """Save cleaned data to parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for data_type, df in cleaned_data.items():
        if data_type != 'categories':  # Skip categories for now
            file_path = output_path / f"{data_type}_cleaned.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved cleaned {data_type} to {file_path}")


def save_feature_data(feature_data: Dict[str, pd.DataFrame], output_dir: str):
    """Save feature-engineered data to parquet files."""
    features_path = Path(output_dir).parent / "features"
    features_path.mkdir(parents=True, exist_ok=True)
    
    for data_type, df in feature_data.items():
        if data_type != 'categories':  # Skip categories for now
            file_path = features_path / f"{data_type}_features.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved {data_type} features to {file_path}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process and feature engineer e-commerce data for Recommndr"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing raw data (default: data/raw)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save processed data (default: data/processed)"
    )
    
    parser.add_argument(
        "--no-save-processed",
        action="store_true",
        help="Skip saving cleaned data"
    )
    
    parser.add_argument(
        "--no-save-features",
        action="store_true",
        help="Skip saving feature data"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.LOG_LEVEL,
        help=f"Log level (default: {settings.LOG_LEVEL})"
    )
    
    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default=settings.LOG_FORMAT,
        help=f"Log format (default: {settings.LOG_FORMAT})"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_format=args.log_format
    )
    
    try:
        # Process data
        result = process_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_processed=not args.no_save_processed,
            save_features=not args.no_save_features
        )
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Processing Summary:")
        print(f"   👥 Users: {len(result['cleaned']['users']):,}")
        print(f"   🛍️  Products: {len(result['cleaned']['products']):,}")
        print(f"   🔗 Interactions: {len(result['cleaned']['interactions']):,}")
        print(f"   📂 Categories: {len(result['cleaned']['categories']):,}")
        print(f"   🎯 Training Samples: {len(result['transformed']['X_train']):,}")
        print(f"   🔍 Validation Samples: {len(result['transformed']['X_val']):,}")
        print(f"   📈 Features: {len(result['transformed']['X_train'].columns):,}")
        print("="*50)
        
        # Exit successfully
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Data processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Data processing failed", exc_info=True)
        print(f"\n❌ ERROR: Data processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
