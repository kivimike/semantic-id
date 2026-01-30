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

## How It Works: The Pipeline

The generation of a Semantic ID involves a pipeline of three main stages:

1.  **Encoder (Quantization)**:
    *   **Input**: Continuous vector embedding (e.g., `[0.1, -0.5, ...]`).
    *   **Process**: The `RQKMeans` algorithm hierarchically assigns the vector to clusters at multiple levels.
    *   **Output**: A sequence of discrete integers (codes), e.g., `(12, 5, 33)`.

2.  **Semantic Formatting**:
    *   **Process**: Codes are joined by a separator.
    *   **Output**: A raw semantic string, e.g., `"12-5-33"`. This represents the *semantic region* of the vector.

3.  **Uniqueness Resolution**:
    *   **Process**: The `UniqueIdResolver` checks a `CollisionStore` (Redis, SQLite, etc.) to see if this string has been assigned.
    *   **Logic**:
        *   First time: Returns `"12-5-33"`.
        *   Collision: Appends a counter, e.g., `"12-5-33-1"`, `"12-5-33-2"`.
    *   **Output**: A globally unique identifier.

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

If you have PyTorch installed and a GPU (CUDA or MPS) available, you can accelerate training and encoding. The PyTorch backend implements `k-means++` initialization, ensuring high-quality centroids similar to the Scikit-Learn backend.

```python
# Automatically uses GPU if available
device = "cuda" # or "mps" or "cpu"

model = RQKMeans(n_levels=4, n_clusters=256)
model.fit(X, device=device)
codes = model.encode(X, device=device)
```

**Note:** `fit()` results may differ slightly between CPU (Scikit-Learn) and GPU (PyTorch) backends due to different random number generators and floating-point precision, even with the same seed. For consistent IDs across environments, see the **Reproducibility** section.

### 3. High-Level Engine with Uniqueness

Use `SemanticIdEngine` to automatically handle collisions (ensure every ID is unique). This is the recommended way to run the full pipeline.

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

### 4. Balanced Clustering (Constrained K-Means)

Standard K-Means can lead to cluster imbalance (some codes used very often, others rarely). Use `implementation="constrained"` to enforce balanced usage of codes.

```python
model = RQKMeans(
    n_levels=4, 
    n_clusters=256, 
    implementation="constrained"
)
model.fit(X)
```

### 5. Advanced Configuration & Custom Stores

You can customize the number of clusters per level and implement custom stores (e.g., for Redis or JSON files).

See **[examples/advanced_usage.py](examples/advanced_usage.py)** for a complete script demonstrating:
*   Defining a custom `JSONCollisionStore`.
*   Using variable cluster sizes (e.g., `n_clusters=[4, 4, 8]`).
*   Step-by-step pipeline execution/debugging.

```python
# Snippet: Variable clusters per level
model = RQKMeans(
    n_levels=3, 
    n_clusters=[10, 20, 50] # Layer 1 has 10 clusters, Layer 2 has 20, etc.
)
```

### 6. Saving and Loading

```python
# Save model artifacts
model.save("my_rq_model")

# Load later
loaded_model = RQKMeans.load("my_rq_model")
```

### 7. Neural Network Support (RQ-VAE)

For complex data distributions, you can use **RQ-VAE** (Residual Quantization Variational AutoEncoder). This trains a neural network to learn the codebooks, often resulting in better reconstruction quality than standard K-Means.

It also supports the **RQ-KMeans+** strategy: initializing the VAE codebooks from a pre-trained RQ-KMeans model for faster convergence.

```python
from semantic_id.algorithms.rq_vae import RQVAE

# Initialize
model = RQVAE(
    in_dim=768,
    num_emb_list=[256, 256, 256],
    e_dim=768,
    layers=[512, 256],
    device="cuda" # or "mps"
)

# Train
model.fit(X_train)

# Generate IDs
ids = model.semantic_id(model.encode(X_test))
```

See **[examples/rq_vae_example.py](examples/rq_vae_example.py)** for a full example of:
*   Training standard RQ-VAE.
*   Using **RQ-KMeans+** warm-start strategy.

### 8. Reproducibility & Cross-Device Consistency

To ensure **identical Semantic IDs** across different machines or devices (e.g., training on a powerful GPU server and inferring on a CPU edge device), follow this workflow:

1.  **Train (`fit`) once**: Train your model on your preferred device.
2.  **Save the model**: Use `model.save()`.
3.  **Load for Inference**: Use `RQKMeans.load()` on the target device.

```python
# Machine A (Training)
model.fit(X_train, device="cuda")
model.save("production_model")

# Machine B (Inference - even on CPU)
prod_model = RQKMeans.load("production_model")
# This will produce the exact same codes as Machine A for the same input
codes = prod_model.encode(X_test, device="cpu")
```

Do **not** retrain (`fit`) on the second machine if you need the IDs to be consistent with the first one, as `fit` involves random initialization which varies across backends (Numpy vs PyTorch) and hardware.

## Roadmap & ToDo

### MVP (Completed)
- [x] RQ-KMeans implementation (CPU/Numpy).
- [x] Standard and Constrained (Balanced) K-Means support.
- [x] Base interfaces (`fit`, `encode`, `save`, `load`).
- [x] Uniqueness resolution (`InMemory`, `SQLite`).
- [x] Basic tests.

### Future Plans (v1.0+)
- [x] **Torch Backend**: Implement `RQKMeans` using PyTorch for GPU acceleration (`torch.cdist`).
- [x] **RQ-VAE**: Add neural network-based Residual Quantization VAE support.
- [ ] **Experiment Runner**: CLI tool for sweeping hyperparameters (L, K) and comparing collision rates.
- [ ] **Advanced Metrics**: Add reconstruction quality metrics (recall@K for retrieval).
