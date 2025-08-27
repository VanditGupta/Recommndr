# 🛍️ Recommndr — Production-Grade E-Commerce Recommendation Pipeline

## 🎯 Project Overview

A full-stack, production-grade machine learning engineering project for real-time e-commerce product recommendations using a two-stage pipeline: **Collaborative Filtering + Ranking**.

## 🏗️ Architecture

- **Two-stage pipeline**: ALS (Collaborative Filtering) + LightGBM (Ranking)
- **Real-time serving**: FastAPI + ONNX Runtime
- **ML experimentation**: MLflow tracking and versioning
- **Data validation**: Great Expectations
- **Data versioning**: DVC with Azure Blob Storage
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Azure Container Apps

## 📁 Project Structure

```
recommndr/
├── data/                   # Data storage (DVC tracked)
│   ├── raw/               # Raw synthetic data
│   ├── processed/         # Cleaned and processed data
│   └── features/          # ML-ready feature data
├── src/                    # Source code
│   ├── data_generation/    # Synthetic data generation ✅ COMPLETE
│   ├── validation/         # Data validation with Great Expectations ✅ COMPLETE
│   ├── processing/         # Data processing and feature engineering ✅ COMPLETE
│   └── utils/             # Utility functions
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── docker/                 # Docker configuration
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── dvc.yaml              # DVC pipeline configuration ✅ COMPLETE
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git
- Azure CLI (for cloud storage)

### Local Development Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd recommndr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup DVC (Azure remote already configured)
dvc status              # Check pipeline status
dvc repro               # Run complete pipeline

# Or run individual stages
python -m src.data_generation.main --users 10000 --products 1000 --interactions 100000
python -m src.validation.main --quality-threshold 0.8
python -m src.processing.main

# Run tests
pytest tests/
```

### DVC Pipeline (Azure Cloud Storage)
```bash
# Check pipeline status
dvc status

# Run complete pipeline
dvc repro

# Push data to Azure
dvc push

# Pull data from Azure
dvc pull

# List pipeline stages
dvc stage list
```

### Docker Setup
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📊 Data Schema & Current Status

### ✅ Phase 1: Data Generation, Validation & Versioning - COMPLETE!

#### Users (10,000) ✅
- **Features**: 41 engineered features including interaction patterns, spending behavior, device preferences
- **Data Quality**: 80% validation score (realistic for synthetic data)
- **Storage**: Parquet format, DVC versioned, Azure cloud backup

#### Products (1,000) ✅
- **Features**: 49 engineered features including popularity scores, engagement metrics, price categories
- **Data Quality**: 100% validation score
- **Storage**: Parquet format, DVC versioned, Azure cloud backup

#### Interactions (100,000) ✅
- **Features**: 44 engineered features including session data, time patterns, user-product compatibility
- **Data Quality**: 75% validation score (realistic for interaction data)
- **Storage**: Parquet format, DVC versioned, Azure cloud backup

#### Categories (71) ✅
- **Data Quality**: 100% validation score
- **Coverage**: Comprehensive e-commerce categories

### 🎯 ML-Ready Data Output
- **Training Samples**: 80,000 (80% split)
- **Validation Samples**: 20,000 (20% split)
- **Feature Count**: 27 engineered features
- **User-Item Matrix**: 9,999 × 1,000 (99% sparse for collaborative filtering)

## 🔧 Development

### Current Pipeline Status
```bash
# All DVC stages are functional
dvc stage list
# data_generation     ✅ Outputs raw synthetic data
# data_validation     ✅ Outputs validation reports
# data_processing     ✅ Outputs ML-ready features

# Check pipeline health
dvc status
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes
3. Add tests
4. Run validation: `python -m src.validation.main --quality-threshold 0.8`
5. Commit and push: `git push origin feature/your-feature`

### Data Pipeline
```bash
# Regenerate data
dvc repro data_generation

# Update data version
dvc add data/
git add data/.gitignore data.dvc
git commit -m "Update data version"

# Push to Azure cloud storage
dvc push
```

**📚 For detailed DVC Azure operations, see [DVC Azure Operations Guide](#-dvc-azure-operations-guide) above**

## 📈 Monitoring & Quality

- **Data Quality**: Great Expectations validation reports (88.75% overall score)
- **Pipeline Health**: DVC pipeline status and Azure cloud sync
- **Performance**: Processing times tracked (1.2s for full pipeline)
- **Storage**: 13MB local cache + Azure cloud backup

## 🚀 Deployment

### Azure Integration ✅
- **Storage Account**: `recommndrstorage` (East US)
- **Container**: `data/recommndr-dvc/`
- **Authentication**: Storage account key + "Storage Blob Data Contributor" role
- **DVC Remote**: `azure://data/recommndr-dvc/`

### 🚀 DVC Azure Operations Guide

#### **1. Push Data to Azure Cloud Storage**
```bash
# Push all DVC data to Azure
dvc push

# Push specific stage to Azure
dvc push data_generation
dvc push data_validation
dvc push data_processing

# Push with specific remote
dvc push --remote azure
```

#### **2. Pull Data from Azure Cloud Storage**
```bash
# Pull all data from Azure
dvc pull

# Pull specific stage from Azure
dvc pull data_generation

# Pull with specific remote
dvc pull --remote azure
```

#### **3. Check Azure Sync Status**
```bash
# Check if local/remote are in sync
dvc status

# Check Azure remote status specifically
dvc status --remote azure

# List all DVC stages
dvc stage list
```

#### **4. View Azure Storage Contents**
```bash
# List all files in Azure DVC container
az storage blob list \
  --account-name recommndrstorage \
  --container-name data \
  --prefix recommndr-dvc/ \
  --output table

# Get specific file info
az storage blob show \
  --account-name recommndrstorage \
  --container-name data \
  --name "recommndr-dvc/files/md5/44/d91d9eb7ff94f3739eb2c3ff0a71d8"
```

#### **5. DVC Pipeline with Azure**
```bash
# Run full pipeline and push to Azure
dvc repro
dvc push

# Run specific stage and push
dvc repro data_generation
dvc push data_generation

# Check pipeline health
dvc status
dvc stage list
```

#### **6. Team Collaboration with Azure**
```bash
# Clone repository and pull data from Azure
git clone <your-repo>
cd Recommndr
dvc pull

# Update data and push to Azure
dvc repro
dvc push
git add dvc.lock
git commit -m "Update data version"
git push
```

#### **7. Azure Storage Details**
- **Total Files**: 32+ files (data + metadata)
- **Data Size**: ~15+ MB
- **Storage Tier**: Hot (frequently accessed)
- **Backup**: Automatic Azure redundancy
- **Security**: RBAC + Storage account keys

#### **8. Troubleshooting Azure DVC Issues**
```bash
# Check DVC remote configuration
dvc remote list
dvc remote show azure

# Verify Azure credentials
az storage account keys list \
  --account-name recommndrstorage \
  --resource-group recommndr-rg

# Test Azure connectivity
az storage container list \
  --account-name recommndrstorage
```

### Azure Deployment
```bash
# Deploy to Azure Container Apps
az containerapp up \
  --name recommndr-api \
  --resource-group recommndr-rg \
  --location eastus \
  --source .
```

## 🎯 Next Steps (Phase 2)

- **Streaming Ingestion**: Kafka + Flink for real-time data
- **Feature Pipeline**: Feast for feature serving
- **ML Model Training**: ALS + LightGBM implementation
- **Real-time API**: FastAPI with ONNX runtime
- **Monitoring**: Prometheus + Grafana dashboards

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**🏆 Phase 1 Status: 100% COMPLETE**  
**🚀 Ready for Phase 2: Streaming & ML Implementation**
