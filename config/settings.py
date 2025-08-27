"""Configuration settings for the Recommndr project."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    FEATURES_DIR: Path = DATA_DIR / "features"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Data generation settings
    NUM_USERS: int = Field(default=10_000, description="Number of users to generate")
    NUM_PRODUCTS: int = Field(default=1_000, description="Number of products to generate")
    NUM_INTERACTIONS: int = Field(default=100_000, description="Number of interactions to generate")
    
    # Data validation settings
    VALIDATION_STRICTNESS: str = Field(default="medium", description="Validation strictness level")
    DATA_QUALITY_THRESHOLD: float = Field(default=0.95, description="Minimum data quality score")
    
    # DVC settings
    DVC_REMOTE_NAME: str = Field(default="storage", description="DVC remote storage name")
    DVC_REMOTE_URL: Optional[str] = Field(default=None, description="DVC remote storage URL")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json or text)")
    
    # Random seed for reproducibility
    RANDOM_SEED: int = Field(default=42, description="Random seed for data generation")
    
    # Data file formats
    DEFAULT_DATA_FORMAT: str = Field(default="parquet", description="Default data file format")
    
    # Performance settings
    BATCH_SIZE: int = Field(default=1000, description="Batch size for data processing")
    MAX_WORKERS: int = Field(default=4, description="Maximum number of worker processes")
    
    # Azure configuration settings
    AZURE_RESOURCE_GROUP: str = Field(default="recommndr-rg", description="Azure resource group name")
    AZURE_LOCATION: str = Field(default="eastus", description="Azure location/region")
    AZURE_STORAGE_ACCOUNT_NAME: str = Field(default="recommndrstorage", description="Azure storage account name")
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = Field(default=None, description="Azure storage account key")
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = Field(default=None, description="Azure storage connection string")
    AZURE_ACR_NAME: str = Field(default="recommndracr", description="Azure container registry name")
    AZURE_ACR_LOGIN_SERVER: str = Field(default="recommndracr.azurecr.io", description="Azure ACR login server")
    AZURE_ACR_USERNAME: Optional[str] = Field(default=None, description="Azure ACR username")
    AZURE_ACR_PASSWORD: Optional[str] = Field(default=None, description="Azure ACR password")
    AZURE_KEY_VAULT_NAME: str = Field(default="recommndr-kv", description="Azure key vault name")
    AZURE_KEY_VAULT_URI: str = Field(default="https://recommndr-kv.vault.azure.net/", description="Azure key vault URI")
    AZURE_ML_WORKSPACE_NAME: str = Field(default="recommndr-ml", description="Azure ML workspace name")
    AZURE_MLFLOW_TRACKING_URI: str = Field(default="", description="Azure MLflow tracking URI")
    AZURE_DATA_CONTAINER: str = Field(default="data", description="Azure data container name")
    AZURE_MODELS_CONTAINER: str = Field(default="models", description="Azure models container name")
    AZURE_SUBSCRIPTION_ID: str = Field(default="", description="Azure subscription ID")
    AZURE_TENANT_ID: str = Field(default="", description="Azure tenant ID")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_data_path(data_type: str) -> Path:
    """Get the path for a specific data type."""
    data_paths = {
        "raw": settings.RAW_DATA_DIR,
        "processed": settings.PROCESSED_DATA_DIR,
        "features": settings.FEATURES_DIR,
    }
    
    if data_type not in data_paths:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return data_paths[data_type]


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.FEATURES_DIR,
        settings.LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
