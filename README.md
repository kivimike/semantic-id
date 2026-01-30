# Semantic ID 🌟

**Turn your vectors into meaningful strings.**

Semantic ID is a friendly Python library that helps you transform continuous vector embeddings (like those from OpenAI, BERT, or ResNet) into discrete, human-readable semantic strings. It uses algorithms like **RQ-KMeans** (Residual Quantization K-Means) and **RQ-VAE** (Residual Quantization Variational Autoencoder) to hierarchically cluster your data, giving you IDs that actually mean something!

Imagine turning `[0.12, -0.88, 0.04, ...]` into `"cars-suv-landrover"`. Okay, maybe more like `"12-4-9-1"`, but you get the idea—it preserves semantic similarity!

## 💡 Inspiration

This project is heavily inspired by the incredible work found in:

*   **[Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065)** (Rajput et al., 2023): The paper that lays the groundwork for using semantic IDs in next-gen recommendation systems.
*   **[MiniOneRec](https://github.com/AkaliKong/MiniOneRec)**: A fantastic repository that demonstrates these concepts in action.

We aim to make these powerful techniques accessible and easy to use for everyone.

## 🗺️ Explore Your Embeddings

Before you start clustering, it's super helpful to "see" your data. We love **[Apple's Embedding Atlas](https://github.com/apple/embedding-atlas)** and suggest everyone try it out! It’s a great way to visualize your high-dimensional vectors and understand the landscape of your data. It's also a great way to evaluate your results after training your RQ-model.

---

## ✨ Features

*   **RQ-KMeans**
*   **RQ-VAE**
*   **Uniqueness**: Automatic handling of collisions (e.g., turning duplicates into `12-4-1` and `12-4-2`).

## 📦 Installation

```bash
pip install -e .
```

To enable **GPU acceleration** (recommended!):
```bash
pip install torch
```

To use **balanced clustering**:
```bash
pip install k-means-constrained
```

## 🚀 Quick Start

### 1. The Basics (RQ-KMeans)

Let's generate some simple IDs. We'll use a small number of clusters (10 per level) so the IDs are short and sweet.

```python
import numpy as np
from semantic_id.algorithms.rq_kmeans import RQKMeans

# 1. Generate some dummy data (100 vectors, 16 dimensions)
X = np.random.randn(100, 16)

# 2. Initialize the model
# We'll use 3 levels with 10 clusters each.
# This means our IDs will look like "X-Y-Z" where numbers are 0-9.
model = RQKMeans(n_levels=3, n_clusters=10, random_state=42)

# 3. Train the model
model.fit(X)

# 4. Generate Semantic IDs
# This converts vectors -> codes -> strings
codes = model.encode(X)     # shape (100, 3)
sids = model.semantic_id(codes)

print(f"Vector: {X[0][:3]}...")
print(f"Semantic ID: {sids[0]}")  # Output: e.g., "3-9-1"
```

### 2. Go Fast with GPU 🏎️

Got a GPU? Let's use it! The PyTorch backend is compatible with `cuda` and `mps`.

```python
device = "cuda" # or "mps" for Mac, or "cpu"

model = RQKMeans(n_levels=3, n_clusters=10)
model.fit(X, device=device)
codes = model.encode(X, device=device)
```

### 3. Ensure Uniqueness (The Engine)

In the real world, two different items might end up in the same cluster. The `SemanticIdEngine` handles this gracefully by appending a counter to duplicates.

```python
from semantic_id.engine import SemanticIdEngine
from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.uniqueness.stores import SQLiteCollisionStore

# Setup the algorithm
encoder = RQKMeans(n_levels=3, n_clusters=10)

# Setup the persistence (saves collision counts to a file)
store = SQLiteCollisionStore("collisions.db")
resolver = UniqueIdResolver(store=store)

# Create the engine
engine = SemanticIdEngine(encoder=encoder, unique_resolver=resolver)

# Train and Get Unique IDs
engine.fit(X)
unique_ids = engine.unique_ids(X)

print(unique_ids[0]) # e.g., "3-9-1"
# If another item has code (3, 9, 1), it becomes "3-9-1-1" automatically!
```

### 4. Neural Networks (RQ-VAE) 🧠

For complex data, a simple K-Means might not be enough. **RQ-VAE** uses a neural network to learn the optimal codebooks.

```python
from semantic_id.algorithms.rq_vae import RQVAE

model = RQVAE(
    in_dim=16,                # Input dimension of your vectors
    num_emb_list=[32, 32, 32], # 32 clusters per level
    e_dim=16,                 # Codebook dimension
    layers=[32, 16],          # Hidden layers
    device="cpu"
)

model.fit(X)
ids = model.semantic_id(model.encode(X))
```

## 🔄 Reproducibility

We know how annoying it is when IDs change between machines. To ensure **identical Semantic IDs** across different environments (e.g., Training on GPU -> Inference on CPU):

1.  **Train (`fit`) once** on your training machine.
2.  **Save** the model: `model.save("my_model")`.
3.  **Load** on your production machine: `model = RQKMeans.load("my_model")`.

Do not re-train on the second machine, as random initialization will differ!

## 🗺️ Project Status

We are actively building! Here is what's ready for you today:

-   ✅ **RQ-KMeans**: Core algorithm working on CPU & GPU.
-   ✅ **RQ-VAE**: Neural network based quantization.
-   ✅ **Uniqueness**: Robust handling of ID collisions.
-   ✅ **Persistence**: Save/Load models easily.


