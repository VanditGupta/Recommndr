"""Data validation using Great Expectations."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig
# from great_expectations.data_context.types.resource_identifiers import GeCloudIdentifier  # Removed for compatibility
from great_expectations.util import get_context

from config.settings import settings, get_data_path
from src.utils.logging import get_logger, log_performance_metrics, log_validation_result
from src.utils.schemas import DataValidationResult

logger = get_logger(__name__)


class DataValidator:
    """Validate data using Great Expectations."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data validator.
        
        Args:
            data_dir: Directory containing data to validate
        """
        self.data_dir = data_dir or get_data_path("raw")
        self.context = self._setup_great_expectations()
        
        # Define validation expectations for each data type
        self.expectations = self._define_expectations()
    
    def _setup_great_expectations(self) -> BaseDataContext:
        """Setup Great Expectations context."""
        try:
            # Try to get existing context
            context = get_context()
            logger.debug("Using existing Great Expectations context")
        except Exception:
            # Create new context
            logger.debug("Creating new Great Expectations context")
            context = BaseDataContext(
                project_config=DataContextConfig(
                    config_version=3.0,
                    plugins_directory=None,
                    config_variables_file_path=None,
                    stores={
                        "expectations_store": {
                            "class_name": "ExpectationsStore",
                            "store_backend": {
                                "class_name": "TupleFilesystemStoreBackend",
                                "base_directory": "expectations"
                            }
                        },
                        "validations_store": {
                            "class_name": "ValidationsStore",
                            "store_backend": {
                                "class_name": "TupleFilesystemStoreBackend",
                                "base_directory": "validations"
                            }
                        },
                        "evaluation_parameter_store": {
                            "class_name": "EvaluationParameterStore",
                            "store_backend": {
                                "class_name": "TupleFilesystemStoreBackend",
                                "base_directory": "evaluation_parameters"
                            }
                        },
                        "checkpoint_store": {
                            "class_name": "CheckpointStore",
                            "store_backend": {
                                "class_name": "TupleFilesystemStoreBackend",
                                "base_directory": "checkpoints"
                            }
                        }
                    }
                )
            )
        
        return context
    
    def _define_expectations(self) -> Dict[str, List[Dict]]:
        """Define validation expectations for each data type."""
        return {
            "users": [
                {
                    "expectation_type": "expect_table_columns_to_match_ordered_list",
                    "kwargs": {
                        "column_list": [
                            "user_id", "age", "gender", "location", "income_level",
                            "preference_category", "device_type", "language_preference",
                            "timezone", "email", "created_at", "last_active"
                        ]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "user_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "email"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "age", "min_value": 13, "max_value": 100}
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "gender",
                        "value_set": ["male", "female", "other"]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "income_level",
                        "value_set": ["low", "medium", "high", "luxury"]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "user_id"}
                }
            ],
            "products": [
                {
                    "expectation_type": "expect_table_columns_to_match_ordered_list",
                    "kwargs": {
                        "column_list": [
                            "product_id", "name", "description", "category", "subcategory",
                            "brand", "price", "discount_percentage", "stock_quantity",
                            "rating", "review_count", "shipping_cost", "weight",
                            "dimensions", "color", "size", "availability_status",
                            "image_url", "tags", "created_at"
                        ]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "product_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "price", "min_value": 0.01}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "discount_percentage", "min_value": 0, "max_value": 100}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "rating", "min_value": 0, "max_value": 5}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "stock_quantity", "min_value": 0}
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "availability_status",
                        "value_set": ["in_stock", "out_of_stock", "discontinued"]
                    }
                }
            ],
            "interactions": [
                {
                    "expectation_type": "expect_table_columns_to_match_ordered_list",
                    "kwargs": {
                        "column_list": [
                            "interaction_id", "user_id", "product_id", "interaction_type",
                            "timestamp", "rating", "review_text", "session_id",
                            "quantity", "total_amount", "payment_method", "dwell_time",
                            "scroll_depth"
                        ]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "interaction_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "interaction_type",
                        "value_set": ["view", "click", "add_to_cart", "purchase", "rating", "review"]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "rating", "min_value": 1, "max_value": 5, "allow_null": True}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "quantity", "min_value": 1}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "total_amount", "min_value": 0.01}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "dwell_time", "min_value": 0}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "scroll_depth", "min_value": 0, "max_value": 100}
                }
            ],
            "categories": [
                {
                    "expectation_type": "expect_table_columns_to_match_ordered_list",
                    "kwargs": {
                        "column_list": [
                            "category_id", "name", "parent_category", "description", "level"
                        ]
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "category_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": "level", "min_value": 1, "max_value": 2}
                }
            ]
        }
    
    def _load_data(self, data_type: str) -> pd.DataFrame:
        """Load data for validation.
        
        Args:
            data_type: Type of data to load (users, products, interactions, categories)
            
        Returns:
            Loaded DataFrame
        """
        # Try Parquet first, then CSV
        data_dir = Path(self.data_dir) if isinstance(self.data_dir, str) else self.data_dir
        parquet_path = data_dir / data_type / f"{data_type}.parquet"
        csv_path = data_dir / data_type / f"{data_type}.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            logger.debug(f"Loaded {data_type} from Parquet: {parquet_path}")
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.debug(f"Loaded {data_type} from CSV: {csv_path}")
        else:
            raise FileNotFoundError(f"No data files found for {data_type}")
        
        return df
    
    def _create_batch_request(self, df: pd.DataFrame, data_type: str) -> RuntimeBatchRequest:
        """Create a batch request for validation.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            RuntimeBatchRequest
        """
        return RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name=data_type,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": f"{data_type}_batch"}
        )
    
    def _apply_expectations(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Apply validation expectations to data.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Validation results
        """
        # Get expectations for this data type
        expectations = self.expectations.get(data_type, [])
        
        # Apply each expectation
        results = {}
        for expectation in expectations:
            expectation_type = expectation["expectation_type"]
            kwargs = expectation["kwargs"]
            
            try:
                # Apply simple validation checks
                if expectation_type == "expect_table_columns_to_match_ordered_list":
                    expected_columns = kwargs.get("column_list", [])
                    actual_columns = list(df.columns)
                    success = actual_columns == expected_columns
                    result = {
                        "success": success,
                        "result": {
                            "observed_value": actual_columns,
                            "expected_value": expected_columns
                        }
                    }
                    
                elif expectation_type == "expect_column_values_to_be_unique":
                    column = kwargs.get("column")
                    if column and column in df.columns:
                        success = df[column].is_unique
                        result = {
                            "success": success,
                            "result": {
                                "observed_value": len(df[column]),
                                "expected_value": len(df[column].unique())
                            }
                        }
                    else:
                        success = False
                        result = {"success": False, "result": None}
                        
                elif expectation_type == "expect_column_values_to_be_between":
                    column = kwargs.get("column")
                    min_value = kwargs.get("min_value")
                    max_value = kwargs.get("max_value")
                    allow_null = kwargs.get("allow_null", False)
                    if column and column in df.columns:
                        # Filter out null values if allowed
                        if allow_null:
                            valid_data = df[column].dropna()
                        else:
                            valid_data = df[column]
                        
                        if len(valid_data) == 0:
                            success = True  # No data to validate
                        elif min_value is not None and max_value is not None:
                            success = valid_data.between(min_value, max_value).all()
                        elif min_value is not None:
                            success = (valid_data >= min_value).all()
                        elif max_value is not None:
                            success = (valid_data <= max_value).all()
                        else:
                            success = True
                        result = {"success": success, "result": None}
                    else:
                        success = False
                        result = {"success": False, "result": None}
                        
                elif expectation_type == "expect_column_values_to_be_in_set":
                    column = kwargs.get("column")
                    value_set = kwargs.get("value_set", [])
                    if column and column in df.columns:
                        success = df[column].isin(value_set).all()
                        result = {"success": success, "result": None}
                    else:
                        success = False
                        result = {"success": False, "result": None}
                        
                elif expectation_type == "expect_column_values_to_not_be_null":
                    column = kwargs.get("column")
                    if column and column in df.columns:
                        success = df[column].notna().all()
                        result = {"success": success, "result": None}
                    else:
                        success = False
                        result = {"success": False, "result": None}
                        
                else:
                    logger.warning(f"Unknown expectation type: {expectation_type}")
                    continue
                
                results[expectation_type] = {
                    "success": result["success"],
                    "result": result["result"],
                    "exception_info": None
                }
                
            except Exception as e:
                logger.error(f"Error applying expectation {expectation_type}: {str(e)}")
                results[expectation_type] = {
                    "success": False,
                    "result": None,
                    "exception_info": {"exception_message": str(e)}
                }
        
        return results
    
    def validate_data_type(self, data_type: str) -> DataValidationResult:
        """Validate a specific data type.
        
        Args:
            data_type: Type of data to validate
            
        Returns:
            Validation result
        """
        start_time = time.time()
        
        try:
            # Load data
            df = self._load_data(data_type)
            
            # Apply expectations
            expectation_results = self._apply_expectations(df, data_type)
            
            # Calculate validation metrics
            total_expectations = len(expectation_results)
            successful_expectations = sum(
                1 for result in expectation_results.values() if result["success"]
            )
            
            # Calculate quality score
            quality_score = successful_expectations / total_expectations if total_expectations > 0 else 0
            
            # Collect validation errors
            validation_errors = []
            for exp_type, result in expectation_results.items():
                if not result["success"]:
                    error_msg = f"Expectation '{exp_type}' failed"
                    if result["exception_info"]:
                        error_msg += f": {result['exception_info'].get('exception_message', 'Unknown error')}"
                    validation_errors.append(error_msg)
            
            # Create validation result
            validation_result = DataValidationResult(
                table_name=data_type,
                total_rows=len(df),
                valid_rows=len(df) if quality_score >= settings.DATA_QUALITY_THRESHOLD else 0,
                invalid_rows=len(df) if quality_score < settings.DATA_QUALITY_THRESHOLD else 0,
                quality_score=quality_score,
                validation_errors=validation_errors,
                validation_timestamp=pd.Timestamp.now()
            )
            
            duration = time.time() - start_time
            log_performance_metrics(f"validate_{data_type}", duration, len(df))
            
            # Log validation result
            log_validation_result(validation_result.model_dump())
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed for {data_type}: {str(e)}", exc_info=True)
            
            # Return failed validation result
            return DataValidationResult(
                table_name=data_type,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                quality_score=0.0,
                validation_errors=[f"Validation failed: {str(e)}"],
                validation_timestamp=pd.Timestamp.now()
            )
    
    def validate_all_data(self) -> Dict[str, DataValidationResult]:
        """Validate all data types.
        
        Returns:
            Dictionary mapping data type to validation result
        """
        logger.info("Starting validation of all data")
        
        data_types = ["categories", "users", "products", "interactions"]
        validation_results = {}
        
        for data_type in data_types:
            logger.info(f"Validating {data_type}...")
            validation_results[data_type] = self.validate_data_type(data_type)
        
        # Calculate overall quality score
        overall_score = sum(
            result.quality_score for result in validation_results.values()
        ) / len(validation_results)
        
        logger.info("Validation completed", extra={
            "overall_quality_score": overall_score,
            "validation_results": {
                data_type: {
                    "quality_score": result.quality_score,
                    "total_rows": result.total_rows,
                    "valid_rows": result.valid_rows
                }
                for data_type, result in validation_results.items()
            }
        })
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, DataValidationResult]) -> str:
        """Generate a human-readable validation report.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Formatted validation report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        
        overall_score = sum(
            result.quality_score for result in validation_results.values()
        ) / len(validation_results)
        
        report_lines.append(f"Overall Data Quality Score: {overall_score:.2%}")
        report_lines.append("")
        
        for data_type, result in validation_results.items():
            report_lines.append(f"📋 {data_type.upper()}")
            report_lines.append(f"   Quality Score: {result.quality_score:.2%}")
            report_lines.append(f"   Total Rows: {result.total_rows:,}")
            report_lines.append(f"   Valid Rows: {result.valid_rows:,}")
            report_lines.append(f"   Invalid Rows: {result.invalid_rows:,}")
            
            if result.validation_errors:
                report_lines.append("   ❌ Validation Errors:")
                for error in result.validation_errors[:5]:  # Show first 5 errors
                    report_lines.append(f"      • {error}")
                if len(result.validation_errors) > 5:
                    report_lines.append(f"      ... and {len(result.validation_errors) - 5} more errors")
            
            report_lines.append("")
        
        # Overall assessment
        if overall_score >= 0.95:
            assessment = "🟢 EXCELLENT - Data quality is very high"
        elif overall_score >= 0.90:
            assessment = "🟡 GOOD - Data quality is acceptable"
        elif overall_score >= 0.80:
            assessment = "🟠 FAIR - Data quality needs attention"
        else:
            assessment = "🔴 POOR - Data quality is below acceptable threshold"
        
        report_lines.append(f"📈 OVERALL ASSESSMENT: {assessment}")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
