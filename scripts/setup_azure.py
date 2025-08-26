#!/usr/bin/env python3
"""Azure setup script for Recommndr project."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.azure_config import get_azure_config, AZURE_RESOURCES


def setup_azure_environment():
    """Set up Azure environment variables."""
    print("🚀 Setting up Azure environment for Recommndr...")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📋 Please create .env file from config/azure_credentials_template.env")
        print("🔑 You'll need to add your actual Azure credentials")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test Azure configuration
    try:
        config = get_azure_config()
        print("✅ Azure configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading Azure configuration: {e}")
        return False


def test_azure_connections():
    """Test Azure resource connections."""
    print("\n🔍 Testing Azure connections...")
    
    try:
        # Test storage account
        from azure.storage.blob import BlobServiceClient
        from config.azure_config import get_storage_connection_string
        
        connection_string = get_storage_connection_string()
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        
        # List containers
        containers = list(blob_service.list_containers())
        print(f"✅ Storage account connected - Found {len(containers)} containers")
        
        # Test ML workspace
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_ML_WORKSPACE_NAME")
        )
        
        workspace = ml_client.workspaces.get(os.getenv("AZURE_ML_WORKSPACE_NAME"))
        print(f"✅ ML workspace connected - {workspace.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Azure connections: {e}")
        return False


def print_azure_resources():
    """Print Azure resource information."""
    print("\n📋 Azure Resources Created:")
    print("=" * 50)
    
    for resource_type, details in AZURE_RESOURCES.items():
        print(f"\n🔹 {resource_type.replace('_', ' ').title()}:")
        for key, value in details.items():
            print(f"   {key}: {value}")
    
    print("\n🔑 Next Steps:")
    print("1. Copy config/azure_credentials_template.env to .env")
    print("2. Add your actual Azure credentials to .env")
    print("3. Run this script again to test connections")
    print("4. Start Phase 2: ML Pipeline Development")


def main():
    """Main setup function."""
    print("🏗️ Recommndr Azure Setup")
    print("=" * 50)
    
    # Check if running from project root
    if not (project_root / "src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Setup Azure environment
    if not setup_azure_environment():
        print_azure_resources()
        sys.exit(1)
    
    # Test connections
    if test_azure_connections():
        print("\n🎉 Azure setup completed successfully!")
        print("🚀 Ready for Phase 2: ML Pipeline Development")
    else:
        print("\n⚠️  Azure setup completed with some issues")
        print("🔧 Please check your credentials and try again")


if __name__ == "__main__":
    main()
