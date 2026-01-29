# Semantic ID

A Python library for generating semantic IDs from embeddings using RQ-KMeans (Residual Quantization K-Means). This library helps in transforming continuous vector embeddings into discrete, human-readable semantic strings, with support for uniqueness resolution.

## Features

*   **RQ-KMeans Algorithm**: Hierarchical discretization using Residual Quantization.
*   **Constrained Clustering**: Support for `k-means-constrained` to ensure balanced clusters.
*   **Collision Handling**: Mechanism to ensure unique IDs (e.g., `12-5-33-17` vs `12-5-33-17-1`) using in-memory or SQLite storage.
*   **Persistence**: Save and load trained models and codebooks.
*   **Scikit-learn Compatible**: Familiar API style (`fit`, `encode`).

## Installation

```bash
pip install -e .
```

To use the **constrained** implementation (balanced clusters), ensure you have `k-means-constrained` installed (included in default dependencies):

```bash
pip install k-means-constrained
```

To enable **GPU acceleration** (PyTorch backend):

```bash
pip install torch
```

## Usage

### 1. Basic Usage (RQ-KMeans)

```python
import numpy as np
from semantic_id.algorithms.rq_kmeans import RQKMeans

# Generate dummy embeddings (N=100, D=16)
X = np.random.randn(100, 16)

# Initialize RQ-KMeans
# 4 levels, 256 clusters per level
model = RQKMeans(n_levels=4, n_clusters=256, random_state=42)

# Train (CPU by default)
model.fit(X)

# Encode to integer codes
codes = model.encode(X)
print(codes.shape)  # (100, 4)

# Get Semantic IDs (strings)
sids = model.semantic_id(codes)
print(sids[0])  # e.g., "12-45-200-5"
```

### 2. GPU Acceleration (PyTorch)

If you have PyTorch installed and a GPU (CUDA or MPS) available, you can accelerate training and encoding.

```python
# Automatically uses GPU if available
device = "cuda" # or "mps" or "cpu"

model = RQKMeans(n_levels=4, n_clusters=256)
model.fit(X, device=device)
codes = model.encode(X, device=device)
```

### 3. High-Level Engine with Uniqueness

Use `SemanticIdEngine` to automatically handle collisions (ensure every ID is unique).

```python
from semantic_id.engine import SemanticIdEngine
from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.uniqueness.stores import SQLiteCollisionStore

# 1. Setup Algorithm
encoder = RQKMeans(n_levels=3, n_clusters=10)

# 2. Setup Uniqueness Store (SQLite for persistence)
store = SQLiteCollisionStore("collisions.db")
resolver = UniqueIdResolver(store=store)

# 3. Create Engine
engine = SemanticIdEngine(encoder=encoder, unique_resolver=resolver)

# Train and Generate Unique IDs
engine.fit(X)
unique_ids = engine.unique_ids(X)

print(unique_ids[0])
```

### 3. Balanced Clustering (Constrained K-Means)

Standard K-Means can lead to cluster imbalance (some codes used very often, others rarely). Use `implementation="constrained"` to enforce balanced usage of codes.

```python
model = RQKMeans(
    n_levels=4, 
    n_clusters=256, 
    implementation="constrained"
)
model.fit(X)
```

### 4. Saving and Loading

```python
# Save model artifacts
model.save("my_rq_model")

# Load later
loaded_model = RQKMeans.load("my_rq_model")
```

## Roadmap & ToDo

### MVP (Completed)
- [x] RQ-KMeans implementation (CPU/Numpy).
- [x] Standard and Constrained (Balanced) K-Means support.
- [x] Base interfaces (`fit`, `encode`, `save`, `load`).
- [x] Uniqueness resolution (`InMemory`, `SQLite`).
- [x] Basic tests.

### Future Plans (v1.0+)
- [x] **Torch Backend**: Implement `RQKMeans` using PyTorch for GPU acceleration (`torch.cdist`).
- [ ] **RQ-VAE**: Add neural network-based Residual Quantization VAE support.
- [ ] **Experiment Runner**: CLI tool for sweeping hyperparameters (L, K) and comparing collision rates.
- [ ] **Advanced Metrics**: Add reconstruction quality metrics (recall@K for retrieval).
