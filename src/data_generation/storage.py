"""Data storage and export functionality."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings, get_data_path, ensure_directories
from src.utils.logging import get_logger, log_performance_metrics
from src.utils.schemas import (
    Category,
    Interaction,
    Product,
    User,
)

logger = get_logger(__name__)


class DataStorage:
    """Handle data storage and export operations."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data storage.
        
        Args:
            data_dir: Directory to store data (defaults to config setting)
        """
        self.data_dir = data_dir or get_data_path("raw")
        ensure_directories()
        
        # Create subdirectories
        self.users_dir = self.data_dir / "users"
        self.products_dir = self.data_dir / "products"
        self.interactions_dir = self.data_dir / "interactions"
        self.categories_dir = self.data_dir / "categories"
        self.metadata_dir = self.data_dir / "metadata"
        
        for directory in [self.users_dir, self.products_dir, self.interactions_dir, 
                         self.categories_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _convert_to_dataframe(self, data: List[Any]) -> pd.DataFrame:
        """Convert Pydantic models to pandas DataFrame.
        
        Args:
            data: List of Pydantic models
            
        Returns:
            Pandas DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to dictionaries
        dict_data = [item.dict() for item in data]
        
        # Handle datetime fields
        df = pd.DataFrame(dict_data)
        
        # Convert datetime fields
        datetime_columns = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_columns:
            df[col] = pd.to_datetime(df[col])
        
        return df
    
    def _save_parquet(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save DataFrame as Parquet file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            **kwargs: Additional arguments for pyarrow.parquet.write_table
        """
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Save as Parquet
        pq.write_table(table, file_path, **kwargs)
        
        logger.debug(f"Saved Parquet file: {file_path}")
    
    def _save_csv(self, df: pd.DataFrame, file_path: Path, **kwargs) -> None:
        """Save DataFrame as CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            **kwargs: Additional arguments for pandas.to_csv
        """
        df.to_csv(file_path, index=False, **kwargs)
        logger.debug(f"Saved CSV file: {file_path}")
    
    def save_users(self, users: List[User], format: str = "parquet") -> Dict[str, Path]:
        """Save users data.
        
        Args:
            users: List of User objects
            format: Output format (parquet or csv)
            
        Returns:
            Dictionary mapping format to file path
        """
        start_time = time.time()
        
        df = self._convert_to_dataframe(users)
        
        saved_files = {}
        
        if format == "parquet" or format == "both":
            parquet_path = self.users_dir / "users.parquet"
            self._save_parquet(df, parquet_path)
            saved_files["parquet"] = parquet_path
        
        if format == "csv" or format == "both":
            csv_path = self.users_dir / "users.csv"
            self._save_csv(df, csv_path)
            saved_files["csv"] = csv_path
        
        duration = time.time() - start_time
        log_performance_metrics("save_users", duration, len(users))
        
        logger.info(f"Saved {len(users)} users in {format} format")
        return saved_files
    
    def save_products(self, products: List[Product], format: str = "parquet") -> Dict[str, Path]:
        """Save products data.
        
        Args:
            products: List of Product objects
            format: Output format (parquet or csv)
            
        Returns:
            Dictionary mapping format to file path
        """
        start_time = time.time()
        
        df = self._convert_to_dataframe(products)
        
        saved_files = {}
        
        if format == "parquet" or format == "both":
            parquet_path = self.products_dir / "products.parquet"
            self._save_parquet(df, parquet_path)
            saved_files["parquet"] = parquet_path
        
        if format == "csv" or format == "both":
            csv_path = self.products_dir / "products.csv"
            self._save_csv(df, csv_path)
            saved_files["csv"] = csv_path
        
        duration = time.time() - start_time
        log_performance_metrics("save_products", duration, len(products))
        
        logger.info(f"Saved {len(products)} products in {format} format")
        return saved_files
    
    def save_interactions(self, interactions: List[Interaction], format: str = "parquet") -> Dict[str, Path]:
        """Save interactions data.
        
        Args:
            interactions: List of Interaction objects
            format: Output format (parquet or csv)
            
        Returns:
            Dictionary mapping format to file path
        """
        start_time = time.time()
        
        df = self._convert_to_dataframe(interactions)
        
        saved_files = {}
        
        if format == "parquet" or format == "both":
            parquet_path = self.interactions_dir / "interactions.parquet"
            self._save_parquet(df, parquet_path)
            saved_files["parquet"] = parquet_path
        
        if format == "csv" or format == "csv":
            csv_path = self.interactions_dir / "interactions.csv"
            self._save_csv(df, csv_path)
            saved_files["csv"] = csv_path
        
        duration = time.time() - start_time
        log_performance_metrics("save_interactions", duration, len(interactions))
        
        logger.info(f"Saved {len(interactions)} interactions in {format} format")
        return saved_files
    
    def save_categories(self, categories: List[Category], format: str = "parquet") -> Dict[str, Path]:
        """Save categories data.
        
        Args:
            categories: List of Category objects
            format: Output format (parquet or csv)
            
        Returns:
            Dictionary mapping format to file path
        """
        start_time = time.time()
        
        df = self._convert_to_dataframe(categories)
        
        saved_files = {}
        
        if format == "parquet" or format == "both":
            parquet_path = self.categories_dir / "categories.parquet"
            self._save_parquet(df, parquet_path)
            saved_files["parquet"] = parquet_path
        
        if format == "csv" or format == "both":
            csv_path = self.categories_dir / "categories.csv"
            self._save_csv(df, csv_path)
            saved_files["csv"] = csv_path
        
        duration = time.time() - start_time
        log_performance_metrics("save_categories", duration, len(categories))
        
        logger.info(f"Saved {len(categories)} categories in {format} format")
        return saved_files
    
    def save_metadata(self, data_summary: Dict[str, Any]) -> Path:
        """Save data generation metadata.
        
        Args:
            data_summary: Summary of generated data
            
        Returns:
            Path to saved metadata file
        """
        metadata = {
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "random_seed": settings.RANDOM_SEED,
            "data_counts": {
                "users": data_summary.get("users_count", 0),
                "products": data_summary.get("products_count", 0),
                "interactions": data_summary.get("interactions_count", 0),
                "categories": data_summary.get("categories_count", 0),
            },
            "settings": {
                "num_users": settings.NUM_USERS,
                "num_products": settings.NUM_PRODUCTS,
                "num_interactions": settings.NUM_INTERACTIONS,
                "batch_size": settings.BATCH_SIZE,
                "max_workers": settings.MAX_WORKERS,
            },
            "file_formats": ["parquet", "csv"],
            "schema_version": "1.0.0",
        }
        
        metadata_path = self.metadata_dir / "generation_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    def save_all_data(
        self,
        data: Dict[str, List],
        format: str = "parquet"
    ) -> Dict[str, Dict[str, Path]]:
        """Save all generated data.
        
        Args:
            data: Dictionary containing all data
            format: Output format (parquet, csv, or both)
            
        Returns:
            Dictionary mapping data type to saved files
        """
        logger.info("Starting to save all generated data")
        
        saved_files = {}
        
        # Save each data type
        if "categories" in data:
            saved_files["categories"] = self.save_categories(data["categories"], format)
        
        if "users" in data:
            saved_files["users"] = self.save_users(data["users"], format)
        
        if "products" in data:
            saved_files["products"] = self.save_products(data["products"], format)
        
        if "interactions" in data:
            saved_files["interactions"] = self.save_interactions(data["interactions"], format)
        
        # Save metadata
        data_summary = {
            "categories_count": len(data.get("categories", [])),
            "users_count": len(data.get("users", [])),
            "products_count": len(data.get("products", [])),
            "interactions_count": len(data.get("interactions", [])),
        }
        
        metadata_path = self.save_metadata(data_summary)
        saved_files["metadata"] = {"json": metadata_path}
        
        logger.info("Completed saving all generated data")
        return saved_files
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about stored data.
        
        Returns:
            Dictionary with data information
        """
        data_info = {}
        
        for data_type in ["users", "products", "interactions", "categories"]:
            data_dir = getattr(self, f"{data_type}_dir")
            data_info[data_type] = {}
            
            for file_path in data_dir.glob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    data_info[data_type][file_path.suffix] = {
                        "path": str(file_path),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    }
        
        return data_info
    
    def cleanup_old_files(self, keep_formats: List[str] = None) -> None:
        """Clean up old data files, keeping only specified formats.
        
        Args:
            keep_formats: List of formats to keep (e.g., ["parquet"])
        """
        if keep_formats is None:
            keep_formats = ["parquet"]  # Default to keeping only Parquet
        
        logger.info(f"Cleaning up old files, keeping formats: {keep_formats}")
        
        for data_type in ["users", "products", "interactions", "categories"]:
            data_dir = getattr(self, f"{data_type}_dir")
            
            for file_path in data_dir.glob("*"):
                if file_path.is_file():
                    file_format = file_path.suffix[1:]  # Remove the dot
                    if file_format not in keep_formats:
                        file_path.unlink()
                        logger.debug(f"Deleted old file: {file_path}")
        
        logger.info("Cleanup completed")
