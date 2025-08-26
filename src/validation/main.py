"""Main entry point for data validation."""

import argparse
import sys
from pathlib import Path
from typing import Dict

from config.settings import settings, get_data_path
from src.utils.logging import get_logger, setup_logging
from src.utils.schemas import DataValidationResult
from src.validation.validators import DataValidator

logger = get_logger(__name__)


def validate_data(
    data_dir: str = None,
    data_types: list = None,
    output_report: str = None,
    strict_mode: bool = False
) -> Dict[str, DataValidationResult]:
    """Validate e-commerce data using Great Expectations.
    
    Args:
        data_dir: Directory containing data to validate
        data_types: Specific data types to validate (default: all)
        output_report: Path to save validation report
        strict_mode: Whether to fail on validation errors
        
    Returns:
        Dictionary containing validation results
    """
    try:
        # Initialize validator
        validator = DataValidator(data_dir)
        
        # Determine which data types to validate
        if data_types is None:
            data_types = ["categories", "users", "products", "interactions"]
        
        # Validate specified data types
        validation_results = {}
        for data_type in data_types:
            if data_type in ["categories", "users", "products", "interactions"]:
                logger.info(f"Validating {data_type}...")
                validation_results[data_type] = validator.validate_data_type(data_type)
            else:
                logger.warning(f"Unknown data type: {data_type}")
        
        # Generate validation report
        report = validator.generate_validation_report(validation_results)
        
        # Print report to console
        print(report)
        
        # Save report to file if specified
        if output_report:
            output_path = Path(output_report)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Validation report saved to: {output_path}")
        
        # Check if validation passed quality threshold
        overall_score = sum(
            result.quality_score for result in validation_results.values()
        ) / len(validation_results)
        
        if strict_mode and overall_score < settings.DATA_QUALITY_THRESHOLD:
            logger.error(f"Data quality below threshold: {overall_score:.2%} < {settings.DATA_QUALITY_THRESHOLD:.2%}")
            sys.exit(1)
        
        return validation_results
        
    except Exception as e:
        logger.error("Data validation failed", exc_info=True)
        raise


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate e-commerce data using Great Expectations"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(get_data_path("raw")),
        help=f"Directory containing data to validate (default: {get_data_path('raw')})"
    )
    
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=["categories", "users", "products", "interactions"],
        help="Specific data types to validate (default: all)"
    )
    
    parser.add_argument(
        "--output-report",
        type=str,
        help="Path to save validation report"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if data quality is below threshold"
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
        "--quality-threshold",
        type=float,
        default=settings.DATA_QUALITY_THRESHOLD,
        help=f"Minimum data quality score (default: {settings.DATA_QUALITY_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_format=args.log_format
    )
    
    # Override quality threshold if specified
    if args.quality_threshold != settings.DATA_QUALITY_THRESHOLD:
        settings.DATA_QUALITY_THRESHOLD = args.quality_threshold
        logger.info(f"Quality threshold set to: {args.quality_threshold}")
    
    try:
        # Validate data
        validation_results = validate_data(
            data_dir=args.data_dir,
            data_types=args.data_types,
            output_report=args.output_report,
            strict_mode=args.strict
        )
        
        # Print summary
        overall_score = sum(
            result.quality_score for result in validation_results.values()
        ) / len(validation_results)
        
        print(f"\n🎯 Validation Summary:")
        print(f"   Overall Quality Score: {overall_score:.2%}")
        print(f"   Quality Threshold: {settings.DATA_QUALITY_THRESHOLD:.2%}")
        print(f"   Status: {'✅ PASSED' if overall_score >= settings.DATA_QUALITY_THRESHOLD else '❌ FAILED'}")
        
        # Exit with appropriate code
        if overall_score >= settings.DATA_QUALITY_THRESHOLD:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Data validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Data validation failed", exc_info=True)
        print(f"\n❌ ERROR: Data validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
