# 🛍️ Recommndr — Production-Grade E-Commerce Recommendation Pipeline

## 🎯 Project Overview

A full-stack, production-grade machine learning engineering project for real-time e-commerce product recommendations using a two-stage pipeline: **Collaborative Filtering + Ranking**.

## 🏗️ Architecture

- **Two-stage pipeline**: ALS (Collaborative Filtering) + LightGBM (Ranking)
- **Real-time serving**: FastAPI + ONNX Runtime
- **ML experimentation**: MLflow tracking and versioning
- **Data validation**: Great Expectations
- **Data versioning**: DVC
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Azure Container Apps

## 📁 Project Structure

```
recommndr/
├── data/                   # Data storage (DVC tracked)
├── src/                    # Source code
│   ├── data_generation/    # Synthetic data generation
│   ├── validation/         # Data validation with Great Expectations
│   ├── processing/         # Data processing and feature engineering
│   └── utils/             # Utility functions
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── docker/                 # Docker configuration
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── dvc.yaml              # DVC pipeline configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git

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

# Setup DVC
dvc init
dvc remote add -d storage azure://your-storage-account/container

# Generate synthetic data
python -m src.data_generation.main

# Validate data
python -m src.validation.main

# Run tests
pytest tests/
```

### Docker Setup
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📊 Data Schema

### Users (10K)
- user_id, age, gender, location (US cities), income_level, preference_category, device_type, language_preference, timezone (US), email, created_at, last_active

### Products (1K)
- product_id, name, description, category, subcategory, brand, price (USD), discount_percentage, stock_quantity, rating, review_count, shipping_cost (USD), weight (lbs), dimensions (inches), color, size, availability_status, image_url, tags, created_at

### Interactions (100K)
- interaction_id, user_id, product_id, interaction_type, timestamp, rating, review_text, session_id, quantity, total_amount (USD), payment_method (US), dwell_time, scroll_depth

## 🔧 Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes
3. Add tests
4. Run validation: `python -m src.validation.main`
5. Commit and push: `git push origin feature/your-feature`

### Data Pipeline
```bash
# Regenerate data
dvc repro data_generation

# Update data version
dvc add data/
git add data/.gitignore data.dvc
git commit -m "Update data version"
```

## 📈 Monitoring

- **Data Quality**: Great Expectations validation reports
- **Pipeline Health**: DVC pipeline status
- **Performance**: Latency and throughput metrics

## 🚀 Deployment

### Azure Deployment
```bash
# Deploy to Azure Container Apps
az containerapp up \
  --name recommndr-api \
  --resource-group your-rg \
  --location eastus \
  --source .
```

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
