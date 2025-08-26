"""Azure configuration for Recommndr project."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class AzureConfig(BaseSettings):
    """Azure configuration settings."""
    
    # Resource Group
    RESOURCE_GROUP: str = "recommndr-rg"
    LOCATION: str = "eastus"
    
    # Storage Account
    STORAGE_ACCOUNT_NAME: str = "recommndrstorage"
    STORAGE_ACCOUNT_KEY: Optional[str] = None
    STORAGE_CONNECTION_STRING: Optional[str] = None
    
    # Container Registry
    ACR_NAME: str = "recommndracr"
    ACR_LOGIN_SERVER: str = "recommndracr.azurecr.io"
    ACR_USERNAME: Optional[str] = None
    ACR_PASSWORD: Optional[str] = None
    
    # Key Vault
    KEY_VAULT_NAME: str = "recommndr-kv"
    KEY_VAULT_URI: str = "https://recommndr-kv.vault.azure.net/"
    
    # ML Workspace
    ML_WORKSPACE_NAME: str = "recommndr-ml"
    MLFLOW_TRACKING_URI: str = "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/3099dd32-fa0c-4404-ae13-4198614916f3/resourceGroups/recommndr-rg/providers/Microsoft.MachineLearningServices/workspaces/recommndr-ml"
    
    # Storage Containers
    DATA_CONTAINER: str = "data"
    MODELS_CONTAINER: str = "models"
    
    # Subscription
    SUBSCRIPTION_ID: str = "3099dd32-fa0c-4404-ae13-4198614916f3"
    TENANT_ID: str = "a8eec281-aaa3-4dae-ac9b-9a398b9215e7"
    
    class Config:
        env_file = ".env"
        env_prefix = "AZURE_"


# Azure resource URLs and endpoints
AZURE_RESOURCES = {
    "storage_account": {
        "name": "recommndrstorage",
        "blob_endpoint": "https://recommndrstorage.blob.core.windows.net/",
        "queue_endpoint": "https://recommndrstorage.queue.core.windows.net/",
        "table_endpoint": "https://recommndrstorage.table.core.windows.net/"
    },
    "container_registry": {
        "name": "recommndracr",
        "login_server": "recommndracr.azurecr.io"
    },
    "key_vault": {
        "name": "recommndr-kv",
        "uri": "https://recommndr-kv.vault.azure.net/"
    },
    "ml_workspace": {
        "name": "recommndr-ml",
        "mlflow_uri": "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/3099dd32-fa0c-4404-ae13-4198614916f3/resourceGroups/recommndr-rg/providers/Microsoft.MachineLearningServices/workspaces/recommndr-ml"
    }
}


def get_azure_config() -> AzureConfig:
    """Get Azure configuration."""
    return AzureConfig()


def get_storage_connection_string() -> str:
    """Get storage connection string."""
    config = get_azure_config()
    if config.STORAGE_CONNECTION_STRING:
        return config.STORAGE_CONNECTION_STRING
    
    # Fallback to constructing from account name and key
    if config.STORAGE_ACCOUNT_KEY:
        return f"DefaultEndpointsProtocol=https;AccountName={config.STORAGE_ACCOUNT_NAME};AccountKey={config.STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    
    raise ValueError("Storage connection string or account key not configured")


def get_acr_credentials() -> tuple[str, str]:
    """Get ACR credentials."""
    config = get_azure_config()
    if config.ACR_USERNAME and config.ACR_PASSWORD:
        return config.ACR_USERNAME, config.ACR_PASSWORD
    
    raise ValueError("ACR credentials not configured")


def get_mlflow_tracking_uri() -> str:
    """Get MLflow tracking URI."""
    config = get_azure_config()
    return config.MLFLOW_TRACKING_URI
