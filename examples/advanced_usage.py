"""
Advanced usage example of Semantic ID library.

This script demonstrates:
1.  Defining a custom Collision Store (JSON-based).
2.  Configuring the RQKMeans algorithm with variable clusters per level.
3.  Assembling the full SemanticIdEngine pipeline.
4.  Step-by-step breakdown of what happens inside the engine.
5.  Token format for LLM-friendly IDs.
6.  Evaluating ID quality with built-in metrics.
7.  Decoding (reconstruction) from codes back to vectors.
8.  Save / Load for reproducibility.
9.  RQ-VAE with training history and learning rate scheduling.
10. SinkhornResolver for semantically-aware collision resolution.
11. Batch encoding for large datasets.
"""

import json
import os
import shutil
import threading
import numpy as np
from typing import Dict

from semantic_id import (
    RQKMeans,
    RQVAE,
    SemanticIdEngine,
    UniqueIdResolver,
    SinkhornResolver,
    InMemoryCollisionStore,
    SQLiteCollisionStore,
    evaluate,
)
# CollisionStore base class is not in __all__, import from sub-module
from semantic_id.uniqueness.stores import CollisionStore

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
    # Cleanup previous run artifacts
    for artifact in ["my_store.json", "saved_model", "saved_engine"]:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)

    # Generate dummy embeddings (N=200, D=32)
    # Ideally these come from a model like BERT, ResNet, etc.
    np.random.seed(42)
    X = np.random.randn(200, 32).astype(np.float32)

    print("=" * 60)
    print("--- 1. Configuring Algorithm ---")
    print("=" * 60)
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

    # ==========================================
    # 4. Token Format (LLM-Friendly IDs)
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 7. Token Format (LLM-Friendly) ---")
    print("=" * 60)

    all_codes = model.encode(X)

    # Standard plain format
    plain_ids = model.semantic_id(all_codes)
    print(f"Plain format:  {plain_ids[0]}")  # e.g., "2-0-5"

    # Token format for language models
    token_ids = model.semantic_id(all_codes, fmt="token")
    print(f"Token format:  {token_ids[0]}")  # e.g., "<a_2><b_0><c_5>"

    # Custom separator for plain format
    custom_ids = model.semantic_id(all_codes, sep="/")
    print(f"Custom sep:    {custom_ids[0]}")  # e.g., "2/0/5"

    # ==========================================
    # 5. Evaluate ID Quality
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 8. Evaluate ID Quality ---")
    print("=" * 60)

    # Pass the encoder to also get quantization_mse (requires decode())
    metrics = evaluate(X, all_codes, encoder=model)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>25s}: {value:.4f}")
        else:
            print(f"  {key:>25s}: {value}")
    
    # Metrics explained:
    # - collision_rate: fraction of items sharing a code (lower is better)
    # - recall_at_10: how well code neighbors match embedding neighbors (higher is better)
    # - distance_correlation: Spearman correlation between distances (higher is better)
    # - quantization_mse: reconstruction error (lower is better)

    # ==========================================
    # 6. Decode (Reconstruction)
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 9. Decode / Reconstruction ---")
    print("=" * 60)

    # RQKMeans supports decoding: reconstruct approximate vectors from codes
    X_reconstructed = model.decode(all_codes)
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Original shape:      {X.shape}")
    print(f"Reconstructed shape: {X_reconstructed.shape}")
    print(f"Sample original:     {X[0][:5]}")
    print(f"Sample reconstructed:{X_reconstructed[0][:5]}")

    # ==========================================
    # 7. Save / Load
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 10. Save / Load ---")
    print("=" * 60)

    # Save a single encoder
    model.save("saved_model")
    print("Model saved to 'saved_model/'")

    loaded_model = RQKMeans.load("saved_model")
    codes_after_load = loaded_model.encode(X[:5])
    print(f"Codes before save: {all_codes[:5].tolist()}")
    print(f"Codes after load:  {codes_after_load.tolist()}")
    assert np.array_equal(all_codes[:5], codes_after_load), "Codes should match!"
    print("Save/Load verified: codes match.")

    # Save the full engine (encoder + resolver state)
    engine_with_mem = SemanticIdEngine(
        encoder=model,
        unique_resolver=UniqueIdResolver(store=SQLiteCollisionStore("engine_collisions.db"))
    )
    engine_with_mem.fit(X)
    engine_with_mem.unique_ids(X)  # populate collision counts
    engine_with_mem.save("saved_engine")
    print("\nEngine saved to 'saved_engine/'")

    loaded_engine = SemanticIdEngine.load("saved_engine")
    new_ids = loaded_engine.unique_ids(X[:3])
    print(f"IDs from loaded engine: {new_ids}")

    # ==========================================
    # 8. RQ-VAE with Training History
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 11. RQ-VAE with Training History ---")
    print("=" * 60)

    rqvae = RQVAE(
        in_dim=32,
        num_emb_list=[16, 16, 16],   # 3 levels, 16 codes each
        e_dim=16,                     # codebook embedding dimension
        layers=[64, 32],              # encoder/decoder hidden layers
        lr=1e-3,
        batch_size=64,
        epochs=20,
        lr_scheduler="cosine",        # cosine annealing schedule
        warmup_epochs=2,              # 2 epochs of linear warmup
        kmeans_init=True,             # initialize codebooks with K-Means
        device="cpu",
        verbose=True,
    )

    print("Training RQ-VAE...")
    rqvae.fit(X)

    # Access training history
    print(f"\nTraining history keys: {list(rqvae.history_.keys())}")
    print(f"Final loss:           {rqvae.history_['loss'][-1]:.4f}")
    print(f"Final recon loss:     {rqvae.history_['recon_loss'][-1]:.4f}")
    print(f"Final collision rate: {rqvae.history_['collision_rate'][-1]:.4f}")
    print(f"Final learning rate:  {rqvae.history_['lr'][-1]:.6f}")

    # Encode and evaluate
    vae_codes = rqvae.encode(X)
    vae_ids = rqvae.semantic_id(vae_codes)
    print(f"\nSample RQ-VAE ID (plain): {vae_ids[0]}")
    print(f"Sample RQ-VAE ID (token): {rqvae.semantic_id(vae_codes[:1], fmt='token')[0]}")

    vae_metrics = evaluate(X, vae_codes, encoder=rqvae)
    print("\nRQ-VAE Metrics:")
    for key, value in vae_metrics.items():
        if isinstance(value, float):
            print(f"  {key:>25s}: {value:.4f}")
        else:
            print(f"  {key:>25s}: {value}")

    # RQ-VAE also supports decode()
    X_vae_recon = rqvae.decode(vae_codes)
    print(f"\nRQ-VAE reconstruction MSE: {np.mean((X - X_vae_recon) ** 2):.6f}")

    # RQ-KMeans+ strategy: pretrain codebooks with RQ-KMeans, then use them
    # to warm-start RQ-VAE training for faster convergence.
    # rqvae_plus = RQVAE(in_dim=32, num_emb_list=[16, 16, 16], ...)
    # rqvae_plus.fit(X, pretrained_codebook_path="saved_model/codebooks.npz")

    # ==========================================
    # 9. SinkhornResolver (RQVAE + Engine)
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 12. SinkhornResolver ---")
    print("=" * 60)

    # SinkhornResolver re-encodes colliding items using the Sinkhorn-Knopp
    # algorithm on the last VQ layer, producing more semantically meaningful
    # unique IDs than simple suffix appending.
    # It requires an RQVAE encoder and is used through the SemanticIdEngine.

    sinkhorn_resolver = SinkhornResolver(
        max_iterations=10,     # max re-encoding rounds
        sk_epsilon=0.003,      # Sinkhorn temperature
        fallback_suffix=True,  # fall back to suffix if collisions remain
    )

    sinkhorn_engine = SemanticIdEngine(
        encoder=rqvae,
        unique_resolver=sinkhorn_resolver,
    )

    sinkhorn_ids = sinkhorn_engine.unique_ids(X)
    n_unique_sinkhorn = len(set(sinkhorn_ids))
    print(f"Unique IDs (Sinkhorn): {n_unique_sinkhorn}/{len(X)}")

    # Compare with suffix-based resolution
    suffix_engine = SemanticIdEngine(
        encoder=rqvae,
        unique_resolver=UniqueIdResolver(store=InMemoryCollisionStore()),
    )
    suffix_ids = suffix_engine.unique_ids(X)
    n_unique_suffix = len(set(suffix_ids))
    print(f"Unique IDs (Suffix):   {n_unique_suffix}/{len(X)}")

    # Show some example IDs
    print(f"\nSample Sinkhorn IDs: {sinkhorn_ids[:5]}")
    print(f"Sample Suffix IDs:   {suffix_ids[:5]}")

    # ==========================================
    # 10. Batch Encoding for Large Datasets
    # ==========================================

    print("\n" + "=" * 60)
    print("--- 13. Batch Encoding ---")
    print("=" * 60)

    # For large datasets that don't fit in GPU memory, use batch_size
    # to process data in chunks.
    large_X = np.random.randn(10000, 32).astype(np.float32)

    # RQ-KMeans with batch encoding
    codes_batched = model.encode(large_X, batch_size=1024)
    print(f"Batch-encoded {len(large_X)} vectors -> codes shape: {codes_batched.shape}")

    # RQ-VAE with batch encoding
    vae_codes_batched = rqvae.encode(large_X, batch_size=512)
    print(f"VAE batch-encoded {len(large_X)} vectors -> codes shape: {vae_codes_batched.shape}")

    # Cleanup
    for artifact in ["my_store.json", "saved_model", "saved_engine", "engine_collisions.db"]:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)
    print("\nCleaned up artifacts. Done!")


if __name__ == "__main__":
    main()
