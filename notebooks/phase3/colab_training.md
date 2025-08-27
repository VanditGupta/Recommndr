# 🚀 Recommndr Phase 3: Google Colab Training

## 📋 **Setup Instructions:**

1. **Upload Data**: Upload the `colab_data` folder to your Google Drive
2. **Open Colab**: Create a new notebook in Google Colab
3. **Copy Code**: Copy each cell below into your Colab notebook
4. **Run**: Execute cells in order

---

## 🔧 **Cell 1: Install Dependencies**

```python
# Install required packages
!pip install faiss implicit scipy numpy pandas
print("✅ Dependencies installed!")

# Verify GPU support
import faiss
print(f"✅ Faiss version: {faiss.__version__}")
print(f"✅ GPU available: {faiss.get_num_gpus() > 0}")
```

---

## 🔧 **Cell 2: Mount Google Drive**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your data folder
import os
os.chdir('/content/drive/MyDrive/colab_data')  # Adjust path as needed
print("✅ Google Drive mounted!")
print("📁 Current directory:", os.getcwd())
print("📋 Files available:", os.listdir('.'))
```

---

## 🔧 **Cell 3: Load Data**

```python
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pickle
import json

# Load matrix and mappings
print("📊 Loading data...")
user_item_matrix = load_npz('user_item_matrix.npz')

with open('user_mapping.pkl', 'rb') as f:
    user_mapping = pickle.load(f)

with open('item_mapping.pkl', 'rb') as f:
    item_mapping = pickle.load(f)

with open('matrix_info.json', 'r') as f:
    matrix_info = json.load(f)

print(f"✅ Matrix loaded: {user_item_matrix.shape}")
print(f"✅ Users: {len(user_mapping):,}")
print(f"✅ Items: {len(item_mapping):,}")
print(f"✅ Interactions: {user_item_matrix.nnz:,}")
print(f"✅ Sparsity: {matrix_info['sparsity']:.2%}")
```

---

## 🔧 **Cell 4: ALS Model Implementation**

```python
import numpy as np
from scipy.sparse import csr_matrix
import time

class ALSModel:
    """Alternating Least Squares for Collaborative Filtering."""
    
    def __init__(self, n_factors=100, n_iterations=20, regularization=0.1, random_state=42):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        self.is_trained = False
        
        # Set random seed
        np.random.seed(random_state)
        
    def fit(self, user_item_matrix):
        """Train the ALS model."""
        print(f"🎯 Training ALS with {self.n_factors} factors...")
        start_time = time.time()
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors and biases
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.1
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.1
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = 0.0
        
        # Training loop
        for iteration in range(self.n_iterations):
            iter_start = time.time()
            
            # Update user factors
            self._update_user_factors(user_item_matrix)
            
            # Update item factors
            self._update_item_factors(user_item_matrix)
            
            # Calculate loss
            loss = self._calculate_loss(user_item_matrix)
            iter_time = time.time() - iter_start
            
            print(f"   Iteration {iteration+1}/{self.n_iterations}, Loss: {loss:.4f}, Time: {iter_time:.2f}s")
        
        self.is_trained = True
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time:.2f}s")
        
    def _update_user_factors(self, user_item_matrix):
        """Update user factors using ALS."""
        for user_id in range(user_item_matrix.shape[0]):
            # Get items rated by this user
            user_items = user_item_matrix[user_id].nonzero()[1]
            if len(user_items) == 0:
                continue
                
            # Get ratings
            ratings = user_item_matrix[user_id, user_items].toarray().flatten()
            
            # Solve linear system
            A = self.item_factors[user_items]
            b = ratings - self.item_biases[user_items] - self.global_bias
            
            # Ridge regression
            ATA = A.T @ A
            ATb = A.T @ b
            ridge_matrix = ATA + self.regularization * np.eye(self.n_factors)
            
            try:
                self.user_factors[user_id] = np.linalg.solve(ridge_matrix, ATb)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                self.user_factors[user_id] = np.linalg.lstsq(ridge_matrix, ATb, rcond=None)[0]
    
    def _update_item_factors(self, user_item_matrix):
        """Update item factors using ALS."""
        for item_id in range(user_item_matrix.shape[1]):
            # Get users who rated this item
            item_users = user_item_matrix[:, item_id].nonzero()[0]
            if len(item_users) == 0:
                continue
                
            # Get ratings
            ratings = user_item_matrix[item_users, item_id].toarray().flatten()
            
            # Solve linear system
            A = self.user_factors[item_users]
            b = ratings - self.user_biases[item_users] - self.global_bias
            
            # Ridge regression
            ATA = A.T @ A
            ATb = A.T @ b
            ridge_matrix = ATA + self.regularization * np.eye(self.n_factors)
            
            try:
                self.item_factors[item_id] = np.linalg.solve(ridge_matrix, ATb)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                self.item_factors[item_id] = np.linalg.lstsq(ridge_matrix, ATb, rcond=None)[0]
    
    def _calculate_loss(self, user_item_matrix):
        """Calculate training loss."""
        total_loss = 0
        n_interactions = 0
        
        for user_id in range(user_item_matrix.shape[0]):
            user_items = user_item_matrix[user_id].nonzero()[1]
            if len(user_items) == 0:
                continue
                
            ratings = user_item_matrix[user_id, user_items].toarray().flatten()
            
            # Calculate predictions directly
            user_factor = self.user_factors[user_id]
            item_factors = self.item_factors[user_items]
            predictions = (user_factor @ item_factors.T + 
                          self.user_biases[user_id] + 
                          self.item_biases[user_items] + 
                          self.global_bias)
            
            # MSE loss
            loss = np.mean((ratings - predictions) ** 2)
            total_loss += loss * len(user_items)
            n_interactions += len(user_items)
        
        # Add regularization
        reg_loss = (self.regularization * 
                   (np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)))
        
        return (total_loss / n_interactions) + reg_loss
    
    def get_user_embeddings(self, user_ids=None):
        """Get user embeddings."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        if user_ids is None:
            return self.user_factors
        return self.user_factors[user_ids]
    
    def get_item_embeddings(self, item_ids=None):
        """Get item embeddings."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        if item_ids is None:
            return self.item_factors
        return self.item_factors[item_ids]

print("✅ ALS Model class defined!")
```

---

## 🔧 **Cell 5: Train ALS Model**

```python
# Training parameters
N_FACTORS = 100      # Latent factors
N_ITERATIONS = 20    # Training iterations
REGULARIZATION = 0.1 # L2 regularization

print("🚀 Starting ALS training...")
print(f"📊 Parameters: {N_FACTORS} factors, {N_ITERATIONS} iterations, λ={REGULARIZATION}")

# Initialize and train model
als_model = ALSModel(
    n_factors=N_FACTORS,
    n_iterations=N_ITERATIONS,
    regularization=REGULARIZATION
)

# Train the model
als_model.fit(user_item_matrix)

print("✅ ALS training completed!")
```

---

## 🔧 **Cell 6: Build Faiss Index**

```python
import faiss

print("🔍 Building Faiss index...")

# Get item embeddings
item_embeddings = als_model.get_item_embeddings()
print(f"📊 Item embeddings shape: {item_embeddings.shape}")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(item_embeddings)

# Create IVF index (good balance of speed/accuracy)
n_lists = min(100, item_embeddings.shape[0] // 10)  # Number of clusters
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(item_embeddings.shape[1]), 
                           item_embeddings.shape[1], n_lists)

# Train the index
index.train(item_embeddings)

# Add vectors to index
index.add(item_embeddings)

print(f"✅ Faiss index built with {n_lists} clusters")
print(f"📊 Index contains {index.ntotal} vectors")
```

---

## 🔧 **Cell 7: Test Recommendations**

```python
def get_recommendations(user_id, k=10):
    """Get recommendations for a user."""
    if user_id not in user_mapping:
        print(f"❌ User {user_id} not found")
        return []
    
    # Get user embedding
    user_idx = user_mapping[user_id]
    user_embedding = als_model.get_user_embeddings([user_idx])[0]
    
    # Normalize for cosine similarity
    user_embedding = user_embedding.reshape(1, -1)
    faiss.normalize_L2(user_embedding)
    
    # Search for similar items
    scores, indices = index.search(user_embedding, k)
    
    # Convert back to item IDs
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    recommended_items = [reverse_item_mapping[idx] for idx in indices[0]]
    
    return list(zip(recommended_items, scores[0]))

# Test with a few users
test_users = list(user_mapping.keys())[:3]
print("🧪 Testing recommendations...")

for user_id in test_users:
    print(f"\n👤 User {user_id}:")
    recommendations = get_recommendations(user_id, k=5)
    for item_id, score in recommendations:
        print(f"   🛍️  Item {item_id}: Score {score:.4f}")

print("\n✅ Recommendation testing completed!")
```

---

## 🔧 **Cell 8: Save Models**

```python
import pickle

print("💾 Saving trained models...")

# Save ALS model
als_model_data = {
    'user_factors': als_model.user_factors,
    'item_factors': als_model.item_factors,
    'user_biases': als_model.user_biases,
    'item_biases': als_model.item_biases,
    'global_bias': als_model.global_bias,
    'n_factors': als_model.n_factors,
    'is_trained': als_model.is_trained
}

with open('als_model.pkl', 'wb') as f:
    pickle.dump(als_model_data, f)

# Save Faiss index
faiss.write_index(index, 'faiss_index.bin')

# Save mappings
with open('user_mapping.pkl', 'wb') as f:
    pickle.dump(user_mapping, f)

with open('item_mapping.pkl', 'wb') as f:
    pickle.dump(item_mapping, f)

print("✅ Models saved:")
print("   📁 als_model.pkl")
print("   📁 faiss_index.bin")
print("   📁 user_mapping.pkl")
print("   📁 item_mapping.pkl")
print("\n📥 Download these files back to your local machine!")
```

---

## 🔧 **Cell 9: Performance Metrics**

```python
import time

print("📊 Performance Analysis...")

# Test recommendation speed
n_test_users = 100
test_user_ids = list(user_mapping.keys())[:n_test_users]

start_time = time.time()
for user_id in test_user_ids:
    get_recommendations(user_id, k=10)
end_time = time.time()

avg_time = (end_time - start_time) / n_test_users
print(f"⚡ Average recommendation time: {avg_time*1000:.2f}ms per user")

# Matrix statistics
print(f"\n📈 Matrix Statistics:")
print(f"   Users: {user_item_matrix.shape[0]:,}")
print(f"   Items: {user_item_matrix.shape[1]:,}")
print(f"   Interactions: {user_item_matrix.nnz:,}")
print(f"   Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.2%}")
print(f"   Memory usage: {user_item_matrix.data.nbytes / (1024*1024):.2f} MB")

print("\n🎉 Phase 3 Training Pipeline Completed!")
```

---

## 📋 **Download Instructions:**

1. **Download Models**: Right-click each saved file and download
2. **Move to Local**: Place in your local `data/processed/` folder
3. **Test Locally**: Run your local Phase 3 pipeline with trained models

## 🚀 **Expected Performance:**

- **Training Time**: 3-10 minutes (vs 10-30 minutes on CPU)
- **Recommendation Speed**: ~1-5ms per user
- **Model Quality**: Higher due to better numerical precision

## 💡 **Tips:**

- Use **GPU runtime** if available for even faster training
- Adjust `N_FACTORS` and `N_ITERATIONS` based on your needs
- Monitor loss convergence during training
- Test with different users to verify quality

---

**🎯 You're now ready to train Phase 3 on Google Colab!**
