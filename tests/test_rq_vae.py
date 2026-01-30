import os
import torch
import numpy as np
import pytest
from semantic_id.algorithms.rq_vae import RQVAE
from semantic_id.algorithms.rq_kmeans_plus import ResidualEncoderWrapper, apply_rqkmeans_plus_strategy

def test_rq_vae_initialization():
    in_dim = 32
    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=[16, 16],
        e_dim=16,
        layers=[32, 16],
        batch_size=10,
        epochs=1
    )
    assert model.in_dim == in_dim
    assert len(model.module.rq.vq_layers) == 2

def test_rq_vae_fit_and_encode():
    N, D = 50, 32
    X = np.random.randn(N, D).astype(np.float32)
    
    model = RQVAE(
        in_dim=D,
        num_emb_list=[8, 8],
        e_dim=16,
        layers=[32, 16],
        batch_size=10,
        epochs=2,
        lr=1e-3,
        device="cpu"
    )
    
    # Train
    model.fit(X)
    
    # Encode
    codes = model.encode(X)
    
    assert codes.shape == (N, 2)
    assert codes.dtype == np.int32
    assert np.all(codes >= 0)
    assert np.all(codes < 8)

def test_rq_vae_save_load(tmp_path):
    N, D = 20, 16
    X = np.random.randn(N, D).astype(np.float32)
    
    model = RQVAE(
        in_dim=D,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        epochs=1,
        device="cpu"
    )
    model.fit(X)
    codes_orig = model.encode(X)
    
    save_path = tmp_path / "rq_vae_model"
    model.save(str(save_path))
    
    loaded_model = RQVAE.load(str(save_path))
    codes_loaded = loaded_model.encode(X)
    
    np.testing.assert_array_equal(codes_orig, codes_loaded)
    assert loaded_model.in_dim == model.in_dim

def test_rq_kmeans_plus_strategy(tmp_path):
    # 1. Create a dummy pretrained codebook .npz
    # We need keys 'codebook_0', 'codebook_1'
    # shapes: (K, e_dim). e_dim must match RQVAE e_dim.
    
    K = 4
    e_dim = 16 # Must match in_dim for residual test
    cb0 = np.random.randn(K, e_dim).astype(np.float32)
    cb1 = np.random.randn(K, e_dim).astype(np.float32)
    
    codebook_path = tmp_path / "pretrained_codebooks.npz"
    np.savez(codebook_path, codebook_0=cb0, codebook_1=cb1)
    
    # 2. Init RQVAE
    model = RQVAE(
        in_dim=16,
        num_emb_list=[K, K],
        e_dim=16, # Must match in_dim for residual
        layers=[32],
        epochs=1,
        device="cpu"
    )
    
    # Check that encoder is just MLPLayers initially
    # Actually it's wrapped in MLPLayers class, but not ResidualEncoderWrapper
    assert not isinstance(model.module.encoder, ResidualEncoderWrapper)
    
    # 3. Apply Strategy manually to verify initialization (without training updates)
    apply_rqkmeans_plus_strategy(model.module, str(codebook_path), device="cpu")
    
    # 4. Checks
    # a) Encoder should be wrapped
    assert isinstance(model.module.encoder, ResidualEncoderWrapper)
    
    # b) Last layer should be zero-inited
    last_linear = list(model.module.encoder.mlp.mlp_layers)[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert torch.all(last_linear.weight == 0)
    assert torch.all(last_linear.bias == 0)
    
    # c) Codebooks should match
    w0 = model.module.rq.vq_layers[0].embedding.weight.detach().cpu().numpy()
    np.testing.assert_allclose(w0, cb0, atol=1e-5)
    
    w1 = model.module.rq.vq_layers[1].embedding.weight.detach().cpu().numpy()
    np.testing.assert_allclose(w1, cb1, atol=1e-5)

    # 5. Verify fit runs with the strategy
    # We use dummy data
    X = np.random.randn(20, 16).astype(np.float32)
    # Re-init model or just fit this one (it's already modified, fit should handle it gracefully or double-wrap?)
    # fit calls apply_rqkmeans_plus_strategy if path is provided.
    # apply_rqkmeans_plus_strategy checks: "if not isinstance(model.encoder, ResidualEncoderWrapper):"
    # So it won't double wrap.
    
    model.fit(X, pretrained_codebook_path=str(codebook_path))
    
    # Weights should change now
    last_linear = list(model.module.encoder.mlp.mlp_layers)[-1]
    assert not torch.all(last_linear.weight == 0)
