# 🛍️ Recommndr — Production-Grade E-Commerce Recommendation Pipeline

## 🔧 Project Overview

Build a **full-stack, production-grade machine learning engineering project** for real-time e-commerce product recommendations using:

- Two-stage pipeline: **Collaborative Filtering + Ranking**
- **Real-time serving** with FastAPI + ONNX
- **ML experimentation, versioning, drift detection**
- **Low latency (<100ms)** and **budget-friendly (<$15/month)** on **Azure**

Models trained on **Google Colab (A100)**. The project simulates production-grade systems like those at Amazon, Flipkart.

## 🎯 Goal

Deliver an end-to-end recommendation system that:

- Generates personalized product recommendations using **ALS** and **LightGBM**
- Provides **"Similar Products"** and **"Frequently Bought Together"**
- Handles **cold-starts** using **recency-weighted, category-aware popularity**
- Tracks **experiments**, **hyperparams**, **A/B tests**, and **drift**
- Ensures **CI/CD**, **security**, and **monitoring**
- Achieves **<200ms latency** E2E with **Docker + ONNX**
- Uses **real product data** or **high-fidelity synthetic data**

## 📅 Project Phases

### Phase 1: Data Generation, Validation & Versioning
- 1M users × 100K products × 10M interactions (cold-start + seasonal)
- Data validated via **Great Expectations**
- **DVC** used for versioning and reproducibility

### Phase 2: Streaming Ingestion & Feature Pipeline
- **Kafka** simulates real-time clickstream events
- **Flink** processes and aggregates features
- Features served via **Feast + Azure Redis**

### Phase 3: Retrieval (Candidate Generation)
- Trained **ALS** or **Faiss** on user-item interactions
- Cold-start handled via **recency-weighted, category-aware popularity**
- Optimized ALS training on Colab (A100 GPU)

### Phase 4: Ranking
- Rank with **LightGBM** on contextual features
- Export model to **ONNX** and apply **quantization**
- Fast inference using **ONNX Runtime (<100ms)**

### Phase 5: Similarity Layer
- Compute item-item similarity via:
  - ALS latent embeddings (cosine)
  - Co-purchase matrix
- Serve via `/similar_items` endpoint

### Phase 6: Serving Layer & API (Local Development)
- **FastAPI endpoints** for recommendation serving
- **Local MLflow model serving** and experiment tracking
- **Local Feast feature store** with Redis backend
- **Recommendation API endpoints**:
  - `/recommend/{user_id}` - Get personalized recommendations
  - `/similar_items/{item_id}` - Get similar items
  - `/user_profile/{user_id}` - Get user profile and history
- **API documentation** and testing
- **Local model versioning** and rollback testing

### Phase 7: Frontend (Local Development)
- **Next.js application** for user interface
- **Dynamic product cards** and recommendation displays
- **User interaction components**:
  - Product browsing and search
  - Recommendation carousels
  - Similar items widgets
  - User preference management
- **Responsive design** for mobile and desktop
- **Local API integration** testing
- **User experience validation**

### Phase 8: Online Deployment & Production Readiness
- **Azure Container Apps** deployment for FastAPI services
- **Azure MLflow Model Registry** integration
- **Production Feast feature store** with Azure Redis
- **Production monitoring** and health checks
- **Azure Application Insights** integration
- **Production API endpoints** with load balancing
- **Environment configuration** and secrets management

### Phase 9: Production Monitoring & Drift Detection
- **Feature distribution drift** monitoring (KS-test, PSI)
- **Model performance monitoring**:
  - Real-time CTR tracking
  - Latency percentile monitoring (p50, p95, p99)
  - Error rate and exception tracking
- **Drift alerts** via Azure Monitor + Prometheus
- **Automated rollback triggers** and execution
- **Performance dashboards** and alerting
- **Model health checks** and validation

### Phase 10: Frontend Deployment & Infrastructure
- **Azure Static Web Apps** deployment for Next.js frontend
- **Production frontend** with Azure CDN
- **CI/CD pipeline** with GitHub Actions
- **Multi-stage Docker builds** and layer caching
- **Blue-green deployment** strategy for zero-downtime rollbacks
- **Security hardening** with Azure Key Vault
- **IP restrictions** and JWT authentication
- **Performance optimization** and monitoring

## 🛠️ Optimization Techniques Added

| Area | Optimization |
|------|--------------|
| **Model Serving** | ONNX quantization, async FastAPI, <100ms latency |
| **Model Training** | LightGBM with GPU, tracked via MLflow |
| **CI/CD** | Docker layer caching, GitHub Actions |
| **Docker** | Multi-stage builds to reduce image size |
| **Monitoring** | Drift + latency via Prometheus |
| **Testing** | Benchmarking via `wrk` / `locust` / `time.perf_counter()` |
| **Security** | Key Vault, IP blocks, HTTPS enforced |
| **Model Versioning** | Automated rollback triggers, blue-green deployment |

## 🔄 Model Rollback Strategy

### Automatic Rollback Triggers
1. **Performance Degradation**:
   - CTR drops below (baseline - 2 standard deviations)
   - Conversion rate drops >15% from baseline
   - User engagement metrics decline significantly

2. **Technical Issues**:
   - API latency p95 exceeds 300ms for >5 minutes
   - Error rate >5% for sustained period
   - Memory/CPU usage anomalies

3. **Data Quality Issues**:
   - Feature drift scores exceed threshold
   - Prediction confidence drops below acceptable range
   - Input validation failures spike

### Rollback Process
- **Automated**: Triggered by monitoring alerts
- **Manual**: Dashboard override capability
- **Graceful**: Traffic gradually shifted to previous model
- **Logged**: All rollback events tracked in MLflow
- **Validated**: Post-rollback health checks ensure stability

## 🏗️ Architecture Diagram

```mermaid
graph TD
  A[User] -->|clicks/views| B[Kafka]
  B --> C[Flink stream processor]
  C --> D[Feast (Local Redis)]
  E[ALS/Faiss Training] --> F[Local MLflow]
  G[LightGBM + ONNX] --> F
  H[FastAPI Local] --> D
  H --> F
  H --> I[ONNX Runtime]
  I --> J[Next.js Frontend]
  K[Production] --> L[Azure Container Apps]
  K --> M[Azure MLflow Registry]
  K --> N[Azure Redis + Feast]
  O[Monitoring] --> P[Azure Monitor]
  O --> Q[Prometheus + Grafana]
```

## 💸 Cost Estimate (Deployment Only)

| Component | Monthly Cost |
|----------|--------------|
| Azure Container Apps (FastAPI) | ~$5 |
| Azure Redis (Feast) | ~$5 |
| Azure Blob Storage | ~$1–2 |
| Azure Static Web Apps | Free |
| Azure Key Vault | Free |
| **Total** | **~$10–12/month** |

## 📊 Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Data Generation, Validation & Versioning |
| 2 | ✅ Complete | Streaming Ingestion & Feature Pipeline |
| 3 | ✅ Complete | Retrieval/Candidate Generation |
| 4 | ✅ Complete | Ranking |
| 5 | ✅ Complete | Similarity Layer |
| 6 | 🔄 In Progress | Serving Layer & API (Local Development) |
| 7 | ⏳ Pending | Frontend (Local Development) |
| 8 | ⏳ Pending | Online Deployment & Production Readiness |
| 9 | ⏳ Pending | Production Monitoring & Drift Detection |
| 10 | ⏳ Pending | Frontend Deployment & Infrastructure |

## ✅ Success Criteria

- ✅ <200ms inference time end-to-end
- ✅ <15$/month deployment budget
- ✅ Drift + latency alerts via Prometheus
- ✅ MLflow tracks all experiments
- ✅ Docker uses multi-stage + caching
- ✅ Real or high-fidelity product data used
- ✅ Security with Key Vault + IP restrictions
- ✅ 80%+ test coverage
- ✅ **Automated rollback system with <5 minute detection time**
- ✅ **Zero-downtime deployments via blue-green strategy**
