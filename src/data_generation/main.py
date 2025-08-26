"""Main entry point for data generation."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

from config.settings import settings, ensure_directories
from src.data_generation.generators import DataGenerator
from src.data_generation.storage import DataStorage
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def generate_data(
    num_users: int = None,
    num_products: int = None,
    num_interactions: int = None,
    output_format: str = "parquet",
    output_dir: str = None,
    cleanup_old: bool = False
) -> Dict[str, List]:
    """Generate synthetic e-commerce data.
    
    Args:
        num_users: Number of users to generate
        num_products: Number of products to generate
        num_interactions: Number of interactions to generate
        output_format: Output format (parquet, csv, or both)
        output_dir: Custom output directory
        cleanup_old: Whether to cleanup old files
        
    Returns:
        Dictionary containing all generated data
    """
    start_time = time.time()
    
    # Override settings if provided
    if num_users:
        settings.NUM_USERS = num_users
    if num_products:
        settings.NUM_PRODUCTS = num_products
    if num_interactions:
        settings.NUM_INTERACTIONS = num_interactions
    
    logger.info("Starting data generation with parameters", extra={
        "num_users": settings.NUM_USERS,
        "num_products": settings.NUM_PRODUCTS,
        "num_interactions": settings.NUM_INTERACTIONS,
        "output_format": output_format,
        "random_seed": settings.RANDOM_SEED
    })
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Initialize data generator
        data_generator = DataGenerator()
        
        # Generate all data
        logger.info("Generating synthetic data...")
        data = data_generator.generate_all_data()
        
        # Initialize storage
        storage = DataStorage(output_dir)
        
        # Save all data
        logger.info("Saving generated data...")
        saved_files = storage.save_all_data(data, output_format)
        
        # Cleanup old files if requested
        if cleanup_old:
            storage.cleanup_old_files(["parquet"])
        
        # Get data info
        data_info = storage.get_data_info()
        
        duration = time.time() - start_time
        
        logger.info("Data generation completed successfully", extra={
            "duration_seconds": round(duration, 2),
            "saved_files": saved_files,
            "data_info": data_info
        })
        
        return data
        
    except Exception as e:
        logger.error("Data generation failed", exc_info=True)
        raise


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic e-commerce data for Recommndr"
    )
    
    parser.add_argument(
        "--users",
        type=int,
        default=settings.NUM_USERS,
        help=f"Number of users to generate (default: {settings.NUM_USERS})"
    )
    
    parser.add_argument(
        "--products",
        type=int,
        default=settings.NUM_PRODUCTS,
        help=f"Number of products to generate (default: {settings.NUM_PRODUCTS})"
    )
    
    parser.add_argument(
        "--interactions",
        type=int,
        default=settings.NUM_INTERACTIONS,
        help=f"Number of interactions to generate (default: {settings.NUM_INTERACTIONS})"
    )
    
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old files after generation"
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
    
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {settings.RANDOM_SEED})"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_format=args.log_format
    )
    
    # Set random seed
    settings.RANDOM_SEED = args.seed
    
    try:
        # Generate data
        data = generate_data(
            num_users=args.users,
            num_products=args.products,
            num_interactions=args.interactions,
            output_format=args.format,
            output_dir=args.output_dir,
            cleanup_old=args.cleanup
        )
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Generated Data Summary:")
        print(f"   👥 Users: {len(data['users']):,}")
        print(f"   🛍️  Products: {len(data['products']):,}")
        print(f"   🔗 Interactions: {len(data['interactions']):,}")
        print(f"   📂 Categories: {len(data['categories']):,}")
        print(f"   💾 Output Format: {args.format}")
        print(f"   🎲 Random Seed: {args.seed}")
        print("="*50)
        
        # Exit successfully
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Data generation failed", exc_info=True)
        print(f"\n❌ ERROR: Data generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
