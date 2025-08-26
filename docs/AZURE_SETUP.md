# 🚀 Azure Setup for Recommndr

## 📋 Overview

This document outlines the Azure infrastructure setup for the Recommndr project, covering all resources needed for Phase 2 (ML Pipeline) and future phases.

## 🏗️ Azure Resources Created

### **Resource Group**
- **Name**: `recommndr-rg`
- **Location**: `eastus`
- **Purpose**: Central resource group for all Recommndr resources

### **Storage Account**
- **Name**: `recommndrstorage`
- **SKU**: `Standard_LRS`
- **Purpose**: Store data, models, and ML artifacts
- **Containers**:
  - `data` - Raw and processed data
  - `models` - Trained ML models

### **Container Registry**
- **Name**: `recommndracr`
- **Login Server**: `recommndracr.azurecr.io`
- **Purpose**: Store Docker images for ML pipeline and serving

### **Key Vault**
- **Name**: `recommndr-kv`
- **URI**: `https://recommndr-kv.vault.azure.net/`
- **Purpose**: Store secrets, API keys, and configuration

### **ML Workspace**
- **Name**: `recommndr-ml`
- **MLflow URI**: `azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/3099dd32-fa0c-4404-ae13-4198614916f3/resourceGroups/recommndr-rg/providers/Microsoft.MachineLearningServices/workspaces/recommndr-ml`
- **Purpose**: MLflow tracking, model registry, and ML pipeline management

## 🔑 Credentials & Configuration

### **Storage Account**
- **Connection String**: Available in Azure Portal
- **Account Key**: Available via Azure CLI

### **Container Registry**
- **Username**: `recommndracr`
- **Password**: Available via Azure CLI

### **Key Vault**
- **Access**: RBAC-based (requires role assignment)

## 📁 Project Structure

```
config/
├── azure_config.py              # Azure configuration class
└── azure_credentials_template.env # Template for .env file

scripts/
└── setup_azure.py               # Azure setup and testing script

docs/
└── AZURE_SETUP.md               # This file
```

## 🚀 Setup Instructions

### **1. Create .env File**
```bash
# Copy the template
cp config/azure_credentials_template.env .env

# Edit .env with your actual credentials
nano .env
```

### **2. Install Azure Dependencies**
```bash
pip install azure-storage-blob azure-ai-ml azure-identity
```

### **3. Test Azure Setup**
```bash
python scripts/setup_azure.py
```

### **4. Verify Resources**
```bash
# List all resources
az resource list --resource-group recommndr-rg

# Check storage containers
az storage container list --account-name recommndrstorage

# Check ML workspace
az ml workspace show --name recommndr-ml --resource-group recommndr-rg
```

## 🔧 Azure CLI Commands

### **Resource Management**
```bash
# Create resource group
az group create --name recommndr-rg --location eastus

# Create storage account
az storage account create --name recommndrstorage --resource-group recommndr-rg --location eastus --sku Standard_LRS

# Create container registry
az acr create --resource-group recommndr-rg --name recommndracr --sku Basic --admin-enabled true

# Create key vault
az keyvault create --name recommndr-kv --resource-group recommndr-rg --location eastus --sku standard

# Create ML workspace
az ml workspace create --name recommndr-ml --resource-group recommndr-rg --location eastus
```

### **Get Credentials**
```bash
# Storage account key
az storage account keys list --resource-group recommndr-rg --account-name recommndrstorage --query "[0].value" -o tsv

# ACR credentials
az acr credential show --name recommndracr --query "username" -o tsv
az acr credential show --name recommndracr --query "passwords[0].value" -o tsv
```

## 💰 Cost Estimation

### **Monthly Costs (Estimated)**
- **Storage Account**: ~$0.02/GB/month
- **Container Registry**: ~$0.50/month (Basic tier)
- **Key Vault**: ~$0.03/month
- **ML Workspace**: ~$0.50/month
- **Total**: **~$1.00/month** (excluding data transfer)

### **Budget Considerations**
- All resources use Basic/Standard tiers for cost optimization
- Storage uses LRS (Locally Redundant Storage) for lower costs
- Container Registry Basic tier has 2GB storage limit

## 🔒 Security & Access Control

### **Network Security**
- All resources allow public access (can be restricted later)
- Storage account has blob encryption enabled
- Key Vault uses RBAC for access control

### **Identity & Access**
- ML Workspace has system-assigned managed identity
- Key Vault uses RBAC authorization
- Storage account uses connection string authentication

## 🚀 Next Steps (Phase 2)

### **ML Pipeline Setup**
1. **Data Processing**: Upload data to Azure Storage
2. **Feature Engineering**: Create feature store in Azure
3. **Model Training**: Set up MLflow tracking
4. **Model Registry**: Store trained models
5. **Pipeline Orchestration**: Create ML pipelines

### **Infrastructure Scaling**
1. **Container Apps**: For model serving
2. **Redis Cache**: For real-time recommendations
3. **Monitoring**: Application Insights and Prometheus
4. **CI/CD**: GitHub Actions with Azure deployment

## 🆘 Troubleshooting

### **Common Issues**
1. **Authentication Errors**: Check Azure CLI login and credentials
2. **Permission Denied**: Verify role assignments and access policies
3. **Resource Not Found**: Check resource group and location
4. **Quota Exceeded**: Verify subscription limits

### **Support Resources**
- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/)

## 📞 Contact

For Azure setup issues or questions:
- Check Azure Portal for resource status
- Use Azure CLI for troubleshooting
- Review Azure Monitor for resource health
