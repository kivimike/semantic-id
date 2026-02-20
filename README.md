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

Before you start clustering, it's super helpful to "see" your data. We love **[Apple's Embedding Atlas](https://github.com/apple/embedding-atlas)** and suggest everyone try it out! It's a great way to visualize your high-dimensional vectors and understand the landscape of your data. It's also a great way to evaluate your results after training your RQ-model.

---

## ✨ Features

*   **RQ-KMeans**: Hierarchical residual quantization with K-Means on CPU & GPU.
*   **RQ-VAE**: Neural network-based quantization with learnable codebooks.
*   **Balanced Clustering**: Constrained K-Means for evenly distributed codes.
*   **Uniqueness**: Automatic collision resolution (suffix-based and Sinkhorn re-encoding).
*   **Custom Formats**: User-defined formatter callbacks for any ID format, plus custom item IDs for collision resolution.
*   **Evaluation**: Built-in metrics — collision rate, recall@K, NDCG@K, distance correlation, code utilization, entropy, quantization MSE.
*   **LLM-Friendly Tokens**: Output IDs in `<a_3><b_9><c_1>` format for language models.
*   **Persistence**: Save/Load models and full engine pipelines.

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
from semantic_id import RQKMeans

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
from semantic_id import SemanticIdEngine, RQKMeans, UniqueIdResolver, SQLiteCollisionStore

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

> **Tip:** For quick experiments, skip the store setup entirely — `SemanticIdEngine` uses an `InMemoryCollisionStore` by default:
> ```python
> engine = SemanticIdEngine(encoder=encoder)  # zero-config uniqueness
> ```

### 4. Neural Networks (RQ-VAE) 🧠

For complex data, a simple K-Means might not be enough. **RQ-VAE** uses a neural network to learn the optimal codebooks.

```python
from semantic_id import RQVAE

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

### 5. Evaluate Your IDs 📊

Use the built-in `evaluate()` function to measure how well your IDs preserve the structure of the original embeddings.

```python
from semantic_id import evaluate

metrics = evaluate(X, codes, encoder=model)
print(metrics)
# {
#   'n_samples': 100,
#   'n_unique_codes': 87,
#   'collision_rate': 0.13,
#   'collision_rate_per_level': [0.9, 0.45, 0.13],
#   'recall_at_10': 0.42,
#   'ndcg_at_10': 0.38,
#   'distance_correlation': 0.65,
#   'code_utilization_per_level': [1.0, 0.95, 0.87],
#   'code_entropy_per_level': [2.30, 2.25, 2.10],
#   'quantization_mse': 0.003
# }
```

| Metric | What it measures |
|---|---|
| `collision_rate` | Fraction of items sharing an ID with another item (lower is better) |
| `collision_rate_per_level` | Collision rate at each prefix depth — shows where uniqueness breaks down |
| `recall_at_10` | How well code-space neighbors match embedding-space neighbors (higher is better) |
| `ndcg_at_10` | Ranking quality of code-space neighbors vs embedding-space (higher is better) |
| `distance_correlation` | Spearman correlation between embedding distances and code distances (higher is better) |
| `code_utilization_per_level` | Fraction of codebook entries used at each level (higher is better) |
| `code_entropy_per_level` | Shannon entropy of code distribution per level (higher = more uniform) |
| `quantization_mse` | Reconstruction error from `decode()` (lower is better; requires an encoder with `decode()`) |

### 6. LLM-Friendly Token Format 🤖

When feeding semantic IDs into a language model, the token format wraps each level in angle brackets with a level letter:

```python
codes = model.encode(X)

# Standard format (default)
plain_ids = model.semantic_id(codes)               # ["3-9-1", "0-5-7", ...]

# Token format for LLMs
token_ids = model.semantic_id(codes, fmt="token")   # ["<a_3><b_9><c_1>", ...]
```

### 7. Custom ID Formats 🎨

Define your own format function for full control over how codes become strings:

```python
# Custom format for your LLM
def my_llm_format(codes):
    return "".join(f"[item_L{i}_{c}]" for i, c in enumerate(codes))

ids = model.semantic_id(codes, formatter=my_llm_format)
# ["[item_L0_3][item_L1_9][item_L2_1]", ...]

# Works through the engine too
engine = SemanticIdEngine(encoder=model)
engine.fit(X)
uids = engine.unique_ids(X, formatter=my_llm_format)
```

### 8. Use Your Own Item IDs 🏷️

Instead of auto-incremented suffixes (`-1`, `-2`), attach your own identifiers:

```python
db_keys = ["SKU001", "SKU002", "SKU003", ...]
uids = engine.unique_ids(X, item_ids=db_keys)
# Collisions become "3-9-1-SKU042" instead of "3-9-1-1"

# Custom separator for the suffix too
uids = engine.unique_ids(X, item_ids=db_keys, sep="/")
# "3/9/1/SKU042"
```

### 9. Balanced Clustering ⚖️

Use `implementation="constrained"` to enforce roughly equal cluster sizes. This reduces collision rates but requires the `k-means-constrained` package.

```python
model = RQKMeans(
    n_levels=3,
    n_clusters=10,
    implementation="constrained",  # balanced clusters
    random_state=42
)
model.fit(X)
```

## 🔄 Reproducibility & Persistence

We know how annoying it is when IDs change between machines. To ensure **identical Semantic IDs** across different environments (e.g., Training on GPU -> Inference on CPU):

1.  **Train (`fit`) once** on your training machine.
2.  **Save** the model.
3.  **Load** on your production machine.

Do not re-train on the second machine, as random initialization will differ!

```python
# Save a single encoder
model.save("my_model")
loaded = RQKMeans.load("my_model")

# Save the full engine (encoder + collision store)
engine.save("my_engine")
loaded_engine = SemanticIdEngine.load("my_engine")
```

Both `RQKMeans` and `RQVAE` support `save()`/`load()`. The engine also persists the collision store so suffix counters are preserved.

## 🗺️ Project Status

We are actively building! Here is what's ready for you today:

-   ✅ **RQ-KMeans**: Core algorithm working on CPU & GPU.
-   ✅ **RQ-VAE**: Neural network based quantization with training history tracking.
-   ✅ **Balanced Clustering**: Constrained K-Means for even code distribution.
-   ✅ **Uniqueness**: Suffix-based and Sinkhorn-based collision resolution.
-   ✅ **Custom Formats**: User-defined formatter callbacks and item IDs for collision resolution.
-   ✅ **Evaluation**: Comprehensive metrics including NDCG, code utilization, entropy, and hierarchical distance.
-   ✅ **Token Format**: LLM-friendly ID output.
-   ✅ **Persistence**: Save/Load models and engines.
