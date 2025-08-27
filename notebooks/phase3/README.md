# 🚀 Phase 3: Retrieval (Candidate Generation)

## 🎯 **What This Phase Builds:**

**First stage** of your recommendation pipeline - finding potential products to recommend to users using collaborative filtering and similarity search.

## 🏗️ **Architecture:**

```
User Interactions → ALS Model → User/Item Embeddings → Faiss Index → Candidate Generation
```

## 📋 **Components Built:**

### **1. ALS Model (`src/retrieval/models/als_model.py`)**
- **Alternating Least Squares** for collaborative filtering
- **Latent factor learning** (100+ factors)
- **Bias terms** for user/item preferences
- **Regularization** to prevent overfitting

### **2. Faiss Search (`src/retrieval/similarity/faiss_search.py`)**
- **Fast similarity search** using Facebook's Faiss
- **Multiple index types**: IVF, HNSW, Flat
- **Cosine/Euclidean** distance metrics
- **Filtered search** capabilities

### **3. Candidate Pipeline (`src/retrieval/main.py`)**
- **End-to-end pipeline** orchestration
- **Model training** and evaluation
- **Candidate generation** for users
- **Performance metrics** calculation

## 🚀 **How to Use:**

### **Option 1: Local Training (Slower)**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/retrieval/main.py --n-factors 100 --n-iterations 20
```

### **Option 2: Google Colab with A100 GPU (Recommended)**
1. **Open** `notebooks/phase3/als_training.ipynb` in Google Colab
2. **Upload** your processed data (zip file)
3. **Run** all cells for GPU-accelerated training
4. **Download** trained models and results

## 📊 **Training Process:**

### **Phase 1: Data Loading**
- Load user-item interaction matrix
- Load user/item mappings and features
- Validate data integrity

### **Phase 2: ALS Training**
- Initialize random factors and biases
- **Alternating updates**: User factors → Item factors
- **Loss calculation** with regularization
- **Convergence monitoring**

### **Phase 3: Faiss Index Building**
- Extract item embeddings from ALS
- **Normalize vectors** for cosine similarity
- **Train index** with clustering
- **Add vectors** for fast search

### **Phase 4: Evaluation**
- **Generate candidates** for test users
- **Calculate metrics**: Coverage, Diversity, Quality
- **Performance analysis** and visualization

## ⚙️ **Hyperparameters:**

### **ALS Model:**
- **`n_factors`**: 100 (latent dimensions)
- **`n_iterations`**: 20 (training epochs)
- **`regularization`**: 0.1 (lambda parameter)

### **Faiss Index:**
- **`index_type`**: IVF (Inverted File)
- **`n_lists`**: 100 (clusters)
- **`metric`**: cosine (similarity measure)

## 📈 **Expected Performance:**

### **Training Time:**
- **Local CPU**: 2-4 hours
- **Colab A100**: 15-30 minutes
- **Speedup**: 4-8x faster with GPU

### **Model Quality:**
- **Loss reduction**: 60-80%
- **Coverage**: 80-90% of catalog
- **Diversity**: 0.7-0.9 (normalized)

## 🔍 **Output Files:**

### **Models:**
- `als_model.pkl` - Trained ALS parameters
- `faiss_index.faiss` - Similarity search index

### **Results:**
- `evaluation_results.pkl` - Performance metrics
- `training_metadata.pkl` - Configuration & results

### **Visualizations:**
- Training loss curves
- Embedding distributions
- Performance analysis

## 🎯 **Next Steps:**

### **Phase 4: Ranking Model**
- **LightGBM** for candidate ranking
- **Feature engineering** from embeddings
- **Business metric** optimization

### **Production Integration**
- **FastAPI** serving layer
- **Real-time** feature store
- **A/B testing** framework

## 🚨 **Troubleshooting:**

### **Common Issues:**
1. **Memory errors**: Reduce `n_factors` or batch size
2. **Slow training**: Use Google Colab with A100 GPU
3. **Poor quality**: Increase `n_iterations` or adjust regularization

### **Performance Tips:**
- **GPU acceleration** for large datasets
- **Batch processing** for memory efficiency
- **Early stopping** based on validation loss

## 📚 **References:**

- **ALS Paper**: "Large-scale Parallel Collaborative Filtering"
- **Faiss Documentation**: Facebook AI Similarity Search
- **Collaborative Filtering**: Netflix Prize approach

---

🎉 **Phase 3 Status: IMPLEMENTED & READY FOR TRAINING!**

Your recommendation system foundation is now complete with:
- ✅ **Data Pipeline** (Phase 1)
- ✅ **Feature Store** (Phase 2)  
- ✅ **Candidate Generation** (Phase 3)

Ready to move to **Phase 4: Ranking Model**! 🚀
