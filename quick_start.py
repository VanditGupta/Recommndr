#!/usr/bin/env python3
"""Quick start script for Recommndr Phase 1."""

import sys
import time
from pathlib import Path

def main():
    """Run the complete Phase 1 pipeline."""
    print("🚀 Recommndr Phase 1: Data Generation & Validation")
    print("=" * 60)
    
    try:
        # Step 1: Generate synthetic data
        print("\n📊 Step 1: Generating synthetic e-commerce data...")
        start_time = time.time()
        
        from src.data_generation.main import generate_data
        data = generate_data(
            num_users=1000,  # Smaller scale for quick demo
            num_products=100,
            num_interactions=5000,
            output_format="parquet"
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Data generation completed in {generation_time:.2f} seconds")
        print(f"   👥 Users: {len(data['users']):,}")
        print(f"   🛍️  Products: {len(data['products']):,}")
        print(f"   🔗 Interactions: {len(data['interactions']):,}")
        print(f"   📂 Categories: {len(data['categories']):,}")
        
        # Step 2: Validate generated data
        print("\n🔍 Step 2: Validating data quality...")
        start_time = time.time()
        
        from src.validation.main import validate_data
        validation_results = validate_data(
            data_dir="data/raw",
            strict_mode=False  # Don't fail on validation issues for demo
        )
        
        validation_time = time.time() - start_time
        print(f"✅ Data validation completed in {validation_time:.2f} seconds")
        
        # Calculate overall quality score
        overall_score = sum(
            result.quality_score for result in validation_results.values()
        ) / len(validation_results)
        
        print(f"📈 Overall Data Quality Score: {overall_score:.2%}")
        
        # Step 3: Show data info
        print("\n📋 Step 3: Data storage information...")
        from src.data_generation.storage import DataStorage
        storage = DataStorage()
        data_info = storage.get_data_info()
        
        for data_type, formats in data_info.items():
            print(f"   {data_type.capitalize()}:")
            for format_name, info in formats.items():
                print(f"     {format_name}: {info['size_mb']} MB")
        
        # Step 4: Summary
        total_time = generation_time + validation_time
        print(f"\n🎉 Phase 1 completed successfully in {total_time:.2f} seconds!")
        print("\n📁 Generated files are stored in:")
        print(f"   Raw data: {Path('data/raw').absolute()}")
        print(f"   Validation report: {Path('data/validation_report.txt').absolute()}")
        
        print("\n🚀 Next steps:")
        print("   1. Run 'python -m src.data_generation.main --help' for more options")
        print("   2. Run 'python -m src.validation.main --help' for validation options")
        print("   3. Run 'python run_tests.py' to run the test suite")
        print("   4. Run 'docker-compose up -d' to start the full stack")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
