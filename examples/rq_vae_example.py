"""
RQ-VAE and RQ-KMeans+ Example.

This script demonstrates:
1.  Training a standard RQ-VAE model.
2.  Using the RQ-KMeans+ strategy to initialize a new model from pre-trained codebooks.
3.  Generating Semantic IDs using the trained VAE.
"""

import os
import numpy as np
import torch
from semantic_id.algorithms.rq_vae import RQVAE
from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.engine import SemanticIdEngine

def main():
    # 0. Setup
    print("--- 0. Setup & Data Generation ---")
    N, D = 1000, 32
    # Create some structured data (clusters)
    centers = np.random.randn(5, D)
    X = []
    for i in range(N):
        c = centers[i % 5]
        noise = np.random.randn(D) * 0.1
        X.append(c + noise)
    X = np.array(X, dtype=np.float32)
    print(f"Data shape: {X.shape}")

    # ==========================================
    # 1. Standard RQ-VAE Training
    # ==========================================
    print("\n--- 1. Training Standard RQ-VAE ---")
    
    rq_vae = RQVAE(
        in_dim=D,
        num_emb_list=[16, 16], # 2 levels, codebook size 16
        e_dim=16,              # embedding dim for codebooks
        layers=[64, 32],       # Encoder/Decoder hidden layers
        batch_size=256,
        epochs=5,
        lr=1e-3,
        device="cpu",          # Use "cuda" or "mps" if available
        verbose=True
    )
    
    rq_vae.fit(X)
    
    # Check reconstruction quality
    codes = rq_vae.encode(X[:5])
    print(f"Sample Codes:\n{codes}")
    
    # Save the model
    os.makedirs("models/rq_vae", exist_ok=True)
    rq_vae.save("models/rq_vae")
    print("Standard RQ-VAE saved to models/rq_vae")


    # ==========================================
    # 2. RQ-KMeans+ Strategy (Warm Start)
    # ==========================================
    print("\n--- 2. Training with RQ-KMeans+ Strategy ---")
    
    # RQ-KMeans+ initializes the VAE's codebooks from a pre-trained RQ-KMeans model.
    # This often leads to faster convergence and better code utilization.
    
    # A. Train a lightweight RQ-KMeans model first
    print("Step A: Train RQ-KMeans initialization model...")
    rq_kmeans = RQKMeans(
        n_levels=2,
        n_clusters=[16, 16], # Must match VAE config
        random_state=42
    )
    rq_kmeans.fit(X)
    
    # Save RQ-KMeans codebooks
    os.makedirs("models/rq_kmeans_init", exist_ok=True)
    rq_kmeans.save("models/rq_kmeans_init")
    codebook_path = "models/rq_kmeans_init/codebooks.npz"
    print(f"RQ-KMeans codebooks saved to {codebook_path}")
    
    # B. Initialize RQ-VAE with these codebooks
    print("Step B: Initialize RQ-VAE with warm-start...")
    
    # Note: For RQ-KMeans+ residual connection to work, 
    # the codebook embedding dim (e_dim) usually matches the input dim (in_dim),
    # or the encoder must project to e_dim.
    # The implementation adds a residual connection: Z = X + Encoder(X).
    # This requires Encoder output dim == Input dim.
    # And VQ expects input dim == e_dim.
    # So for strict RQ-KMeans+, in_dim must equal e_dim.
    # Or we project X to e_dim before residual? 
    # The implementation: `return x + self.mlp(x)`. So x and mlp(x) must match.
    # And mlp output is fed to VQ. VQ expects e_dim.
    # So X dim must be e_dim.
    
    # If our data is D=32 but we want e_dim=16, we can't use the simple residual connection on X directly
    # unless we project X first.
    # The reference implementation assumes e_dim == in_dim for this strategy.
    
    print("(!) Note: RQ-KMeans+ requires e_dim == in_dim for the residual connection.")
    print("    Retraining simple RQ-KMeans with D=32 to match input...")
    
    rq_kmeans_plus_init = RQKMeans(
        n_levels=2,
        n_clusters=[16, 16], 
        random_state=42
    )
    # RQKMeans natively handles D input and uses it as codebook dim.
    rq_kmeans_plus_init.fit(X)
    rq_kmeans_plus_init.save("models/rq_kmeans_init_plus")
    plus_codebook_path = "models/rq_kmeans_init_plus/codebooks.npz"

    rq_vae_plus = RQVAE(
        in_dim=D,              # 32
        num_emb_list=[16, 16],
        e_dim=D,               # 32 (Matches input for residual)
        layers=[64, 64],
        batch_size=256,
        epochs=5,
        device="cpu",
        verbose=True
    )
    
    # Fit using the pretrained codebooks
    # This wraps the encoder in a Residual connection and zero-inits the MLP,
    # effectively starting as a pure RQ-KMeans model (Identity + Quantization).
    rq_vae_plus.fit(X, pretrained_codebook_path=plus_codebook_path)
    
    print("RQ-KMeans+ model trained.")
    
    # ==========================================
    # 3. Generating Semantic IDs
    # ==========================================
    print("\n--- 3. Generating Semantic IDs ---")
    
    engine = SemanticIdEngine(encoder=rq_vae_plus)
    
    # Generate IDs for new data
    X_new = X[:5]
    ids = engine.unique_ids(X_new)
    
    for i, (vec, uid) in enumerate(zip(X_new, ids)):
        print(f"Vec {i}: {uid}")

if __name__ == "__main__":
    main()
