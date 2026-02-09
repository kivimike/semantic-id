"""
Advanced usage example of Semantic ID library.

This script demonstrates:
1.  Defining a custom Collision Store (JSON-based).
2.  Configuring the RQKMeans algorithm with variable clusters per level.
3.  Assembling the full SemanticIdEngine pipeline.
4.  Step-by-step breakdown of what happens inside the engine.
"""

import json
import os
import threading
import numpy as np
from typing import Dict

from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.uniqueness.stores import CollisionStore
from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.engine import SemanticIdEngine

# ==========================================
# 1. Custom Store Implementation
# ==========================================

class JSONCollisionStore(CollisionStore):
    """
    A simple file-based store using JSON.
    Useful for small-scale persistence or debugging.
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}
        
        # Load existing if available
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._counts = json.load(f)

    def next_suffix(self, key: str) -> int:
        with self._lock:
            # Get current count (0 if new)
            current = self._counts.get(key, 0)
            
            # Increment and save
            self._counts[key] = current + 1
            self._save()
            
            return current

    def _save(self):
        # In production, you'd want to write atomically (write temp + rename)
        with open(self.path, "w") as f:
            json.dump(self._counts, f, indent=2)

# ==========================================
# 2. Setup and Training
# ==========================================

def main():
    # Cleanup previous run
    if os.path.exists("my_store.json"):
        os.remove("my_store.json")

    # Generate dummy embeddings (N=50, D=8)
    # Ideally these come from a model like BERT, ResNet, etc.
    X = np.random.randn(50, 8).astype(np.float32)

    print("--- 1. Configuring Algorithm ---")
    # We can specify different K for each level.
    # Level 1: 4 regions (broad categories)
    # Level 2: 4 sub-regions
    # Level 3: 8 sub-regions (finer detail)
    model = RQKMeans(
        n_levels=3, 
        n_clusters=[4, 4, 8], 
        random_state=42
    )

    print("--- 2. Training Encoder ---")
    model.fit(X)
    print("Encoder trained.")

    print("\n--- 3. Setting up Custom Store ---")
    store = JSONCollisionStore("my_store.json")
    resolver = UniqueIdResolver(store=store)

    print("--- 4. Creating Engine ---")
    engine = SemanticIdEngine(encoder=model, unique_resolver=resolver)

    # ==========================================
    # 3. Pipeline Execution
    # ==========================================
    
    print("\n--- 5. Generating IDs (The Pipeline) ---")
    
    # Let's simulate two very similar vectors to force a collision
    # (In RQ-KMeans, nearby vectors often get the same code)
    v1 = X[0:1]
    v2 = X[0:1] + 1e-5 # Tiny perturbation
    
    # Process v1
    id1 = engine.unique_ids(v1)[0]
    print(f"Vector 1 ID: {id1}")
    
    # Process v2 (should collide physically, but get unique suffix)
    id2 = engine.unique_ids(v2)[0]
    print(f"Vector 2 ID: {id2}")
    
    # Verify collision happened in store
    with open("my_store.json", "r") as f:
        data = json.load(f)
        print(f"\nStore content: {data}")
        # Expected: {"<code>-<code>-<code>": 2} indicating it was seen twice

    print("\n--- 6. Manual Pipeline Inspection ---")
    # Breaking down what engine.unique_ids() does internally:
    
    # Step A: Encode to discrete integers
    codes = model.encode(v1)
    print(f"Step A (Codes): {codes[0]}") 
    # e.g., [2, 0, 5]

    # Step B: Convert to Semantic String
    sid_raw = model.semantic_id(codes)[0]
    print(f"Step B (Raw String): {sid_raw}")
    # e.g., "2-0-5"

    # Step C: Resolve Uniqueness
    # (Note: calling this again will increment the counter in our store!)
    suffix_idx = store.next_suffix(sid_raw)
    final_id = sid_raw if suffix_idx == 0 else f"{sid_raw}-{suffix_idx}"
    print(f"Step C (Final Unique ID): {final_id}")

if __name__ == "__main__":
    main()
