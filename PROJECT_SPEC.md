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

### Phase 6: Experiment Tracking, Versioning, Metadata
- Track experiments via **MLflow**
- Register models (ALS + LightGBM) in **MLflow Model Registry**
- Feature metadata managed via **Feast Registry**
- **Automatic model rollback triggers**:
  - Performance degradation detection (CTR < baseline - 2σ)
  - Latency threshold breach (>300ms p95)
  - Error rate spike (>5% failed requests)
  - Automated rollback to last stable model version

### Phase 7: A/B Testing Simulation
- Offline A/B test: ALS+LightGBM vs. popularity-only
- Metrics: CTR, ATC, Conversion Rate
- Results logged in **MLflow**

### Phase 8: Drift Detection & Monitoring
- Monitor **feature distribution drift** (KS-test, PSI)
- Detect **ranking score drift**
- **Model performance monitoring**:
  - Real-time CTR tracking
  - Latency percentile monitoring (p50, p95, p99)
  - Error rate and exception tracking
- Drift alerts via **Prometheus + Grafana**
- **Automated rollback integration** with drift detection

### Phase 9: Serving Layer & API
- Use **FastAPI** + **ONNX Runtime** for low-latency serving
- Implement **async endpoints** for better throughput
- **Health checks** and **performance monitoring** endpoints
- **Model versioning endpoints** for A/B traffic splitting
- **Benchmark APIs** using `time.perf_counter()` or `wrk`

### Phase 10: Frontend
- Built with **Next.js**
- Dynamic product cards, filters, similar items carousel
- Deployed on **Azure Static Web Apps**

### Phase 11: Infrastructure, Security, CI/CD
- Local stack: Docker Compose (FastAPI, Redis, MLflow, etc.)
- Production: Azure Container Apps + Static Web Apps
- Secrets in **Azure Key Vault**
- CI/CD with **GitHub Actions**
- **Multi-stage Docker builds** and **layer caching**
- **Blue-green deployment** strategy for zero-downtime rollbacks
- IP restrictions + JWT for security

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
  C --> D[Feast (Azure Redis)]
  E[ALS/Faiss Training] --> F[MLflow Registry]
  G[LightGBM + ONNX] --> F
  H[Drift Detection] --> I[Prometheus]
  J[FastAPI Serving] --> D
  J --> F
  J --> K[ONNX Runtime]
  K --> L[Frontend: Next.js]
  I --> M[Grafana]
  N[DVC + GE] --> E
  O[Model Monitor] --> P[Auto Rollback]
  P --> F
  I --> O
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
