import numpy as np
import pytest

from semantic_id.utils.metrics import (
    code_entropy_per_level,
    code_utilization_per_level,
    collision_rate_per_level,
    distance_correlation,
    evaluate,
    find_similar,
    hierarchical_distance,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# hierarchical_distance
# ---------------------------------------------------------------------------


class TestHierarchicalDistance:
    def test_identical_codes(self):
        a = np.array([[1, 2, 3]])
        assert hierarchical_distance(a, a).item() == 0

    def test_first_level_differs(self):
        a = np.array([[0, 2, 3]])
        b = np.array([[1, 2, 3]])
        assert hierarchical_distance(a, b).item() == 3

    def test_last_level_differs(self):
        a = np.array([[1, 2, 0]])
        b = np.array([[1, 2, 9]])
        assert hierarchical_distance(a, b).item() == 1

    def test_middle_level_differs(self):
        a = np.array([[1, 0, 3]])
        b = np.array([[1, 9, 3]])
        # Prefix match at level 0 only -> distance = 3 - 1 = 2
        assert hierarchical_distance(a, b).item() == 2

    def test_single_level(self):
        a = np.array([[5]])
        b = np.array([[5]])
        assert hierarchical_distance(a, b).item() == 0
        c = np.array([[3]])
        assert hierarchical_distance(a, c).item() == 1

    def test_batch(self):
        a = np.array([[1, 2, 3], [0, 0, 0]])
        b = np.array([[1, 2, 9], [0, 0, 0]])
        dists = hierarchical_distance(a, b)
        np.testing.assert_array_equal(dists, [1, 0])

    def test_broadcasting(self):
        query = np.array([[1, 2, 3]])  # (1, 3)
        codes = np.array([[1, 2, 3], [1, 2, 0], [1, 0, 0], [0, 0, 0]])  # (4, 3)
        dists = hierarchical_distance(query, codes)
        np.testing.assert_array_equal(dists, [0, 1, 2, 3])

    def test_all_different(self):
        a = np.array([[0, 0, 0, 0]])
        b = np.array([[1, 1, 1, 1]])
        assert hierarchical_distance(a, b).item() == 4


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    @pytest.fixture()
    def clustered_data(self):
        """Six tight clusters far apart — codes match clusters well."""
        rng = np.random.RandomState(0)
        clusters = []
        code_rows = []
        for i in range(6):
            center = np.zeros(8, dtype=np.float32)
            center[i % 8] = 20.0 * (i + 1)
            cluster = rng.randn(10, 8).astype(np.float32) * 0.01 + center
            clusters.append(cluster)
            code_rows.append(np.full((10, 2), [i // 3, i % 3], dtype=np.int32))
        X = np.vstack(clusters)
        codes = np.vstack(code_rows)
        return X, codes

    def test_perfect_codes_have_high_recall(self, clustered_data):
        X, codes = clustered_data
        r = recall_at_k(X, codes, k=5, seed=0)
        assert r > 0.1

    def test_seed_determinism(self, clustered_data):
        X, codes = clustered_data
        r1 = recall_at_k(X, codes, k=5, seed=123)
        r2 = recall_at_k(X, codes, k=5, seed=123)
        assert r1 == r2

    def test_different_seeds_may_differ(self, clustered_data):
        X, codes = clustered_data
        r1 = recall_at_k(X, codes, k=5, seed=0)
        r2 = recall_at_k(X, codes, k=5, seed=999)
        # They *could* be equal, but almost certainly differ on real data
        # Just check both are valid
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0

    def test_k_ge_n_raises(self):
        X = np.random.randn(5, 4).astype(np.float32)
        codes = np.zeros((5, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="k=10 must be less than N=5"):
            recall_at_k(X, codes, k=10)

    def test_hamming_metric(self, clustered_data):
        X, codes = clustered_data
        r = recall_at_k(X, codes, k=5, metric="hamming")
        assert 0.0 <= r <= 1.0

    def test_hierarchical_metric(self, clustered_data):
        X, codes = clustered_data
        r = recall_at_k(X, codes, k=5, metric="hierarchical")
        assert 0.0 <= r <= 1.0

    def test_returns_float(self, clustered_data):
        X, codes = clustered_data
        r = recall_at_k(X, codes, k=5)
        assert isinstance(r, float)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    @pytest.fixture()
    def clustered_data(self):
        rng = np.random.RandomState(0)
        clusters = []
        code_rows = []
        for i in range(6):
            center = np.zeros(8, dtype=np.float32)
            center[i % 8] = 20.0 * (i + 1)
            cluster = rng.randn(10, 8).astype(np.float32) * 0.01 + center
            clusters.append(cluster)
            code_rows.append(np.full((10, 2), [i // 3, i % 3], dtype=np.int32))
        X = np.vstack(clusters)
        codes = np.vstack(code_rows)
        return X, codes

    def test_good_codes_have_high_ndcg(self, clustered_data):
        X, codes = clustered_data
        n = ndcg_at_k(X, codes, k=5, seed=0)
        assert n > 0.1

    def test_k_ge_n_raises(self):
        X = np.random.randn(5, 4).astype(np.float32)
        codes = np.zeros((5, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="k=10 must be less than N=5"):
            ndcg_at_k(X, codes, k=10)

    def test_returns_float_in_range(self, clustered_data):
        X, codes = clustered_data
        n = ndcg_at_k(X, codes, k=5)
        assert isinstance(n, float)
        assert 0.0 <= n <= 1.0

    def test_seed_determinism(self, clustered_data):
        X, codes = clustered_data
        n1 = ndcg_at_k(X, codes, k=5, seed=7)
        n2 = ndcg_at_k(X, codes, k=5, seed=7)
        assert n1 == n2


# ---------------------------------------------------------------------------
# distance_correlation
# ---------------------------------------------------------------------------


class TestDistanceCorrelation:
    def test_positive_correlation_for_good_codes(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 4).astype(np.float32)
        codes = np.zeros((100, 2), dtype=np.int32)
        codes[:, 0] = (X[:, 0] > 0).astype(np.int32)
        codes[:, 1] = (X[:, 1] > 0).astype(np.int32)
        corr = distance_correlation(X, codes, n_pairs=5000, seed=0)
        assert corr > 0.0

    def test_seed_determinism(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 4).astype(np.float32)
        codes = rng.randint(0, 5, (50, 3)).astype(np.int32)
        c1 = distance_correlation(X, codes, seed=42)
        c2 = distance_correlation(X, codes, seed=42)
        assert c1 == c2

    def test_hamming_metric(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 4).astype(np.float32)
        codes = rng.randint(0, 5, (50, 3)).astype(np.int32)
        corr = distance_correlation(X, codes, metric="hamming")
        assert isinstance(corr, float)

    def test_returns_float(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 4).astype(np.float32)
        codes = rng.randint(0, 5, (50, 3)).astype(np.int32)
        assert isinstance(distance_correlation(X, codes), float)


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    @pytest.fixture()
    def codes(self):
        return np.array(
            [
                [0, 0, 0],  # idx 0
                [0, 0, 1],  # idx 1 — differs at level 2 from 0
                [0, 1, 0],  # idx 2 — differs at level 1 from 0
                [1, 0, 0],  # idx 3 — differs at level 0 from 0
                [0, 0, 0],  # idx 4 — identical to 0
            ],
            dtype=np.int32,
        )

    def test_query_by_index(self, codes):
        indices, dists = find_similar(codes, query=0, k=3)
        assert 0 not in indices, "Query index should be excluded"
        assert len(indices) == 3

    def test_query_by_vector(self, codes):
        query_vec = np.array([0, 0, 0], dtype=np.int32)
        indices, dists = find_similar(codes, query=query_vec, k=3)
        assert len(indices) == 3
        assert dists[0] == 0  # item 0 and 4 are identical

    def test_sorted_by_distance(self, codes):
        indices, dists = find_similar(codes, query=0, k=4)
        assert list(dists) == sorted(dists)

    def test_k_larger_than_n(self, codes):
        indices, dists = find_similar(codes, query=0, k=100)
        assert len(indices) == 4  # N-1 since query excluded

    def test_nearest_is_identical(self, codes):
        indices, dists = find_similar(codes, query=0, k=1)
        assert indices[0] == 4  # idx 4 is identical to idx 0
        assert dists[0] == 0

    def test_numpy_integer_query(self, codes):
        idx = np.int64(0)
        indices, dists = find_similar(codes, query=idx, k=2)
        assert 0 not in indices


# ---------------------------------------------------------------------------
# code_utilization_per_level
# ---------------------------------------------------------------------------


class TestCodeUtilizationPerLevel:
    def test_full_utilization(self):
        codes = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        util = code_utilization_per_level(codes, n_clusters=3)
        assert util == [1.0, 1.0]

    def test_partial_utilization(self):
        codes = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int32)
        util = code_utilization_per_level(codes, n_clusters=3)
        assert util[0] == pytest.approx(1.0 / 3)  # only code 0 used at level 0
        assert util[1] == 1.0

    def test_inferred_n_clusters(self):
        codes = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        util = code_utilization_per_level(codes)
        assert util == [1.0, 1.0]

    def test_variable_n_clusters(self):
        codes = np.array([[0, 0], [1, 1]], dtype=np.int32)
        util = code_utilization_per_level(codes, n_clusters=[4, 8])
        assert util[0] == pytest.approx(2.0 / 4)
        assert util[1] == pytest.approx(2.0 / 8)


# ---------------------------------------------------------------------------
# code_entropy_per_level
# ---------------------------------------------------------------------------


class TestCodeEntropyPerLevel:
    def test_uniform_distribution_max_entropy(self):
        codes = np.arange(100).reshape(-1, 1).astype(np.int32)
        codes = np.hstack([codes, codes])
        ent = code_entropy_per_level(codes)
        expected = np.log(100)
        assert ent[0] == pytest.approx(expected, rel=1e-3)

    def test_single_code_zero_entropy(self):
        codes = np.zeros((50, 2), dtype=np.int32)
        ent = code_entropy_per_level(codes)
        assert ent[0] == pytest.approx(0.0, abs=1e-6)
        assert ent[1] == pytest.approx(0.0, abs=1e-6)

    def test_returns_list_of_correct_length(self):
        codes = np.zeros((10, 4), dtype=np.int32)
        ent = code_entropy_per_level(codes)
        assert len(ent) == 4


# ---------------------------------------------------------------------------
# collision_rate_per_level
# ---------------------------------------------------------------------------


class TestCollisionRatePerLevel:
    def test_no_collisions(self):
        codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
        rates = collision_rate_per_level(codes)
        # At depth 1: 2 unique prefixes for 4 items -> 0.5
        assert rates[0] == pytest.approx(0.5)
        # At depth 2: 4 unique prefixes for 4 items -> 0.0
        assert rates[1] == pytest.approx(0.0)

    def test_all_same(self):
        codes = np.zeros((10, 3), dtype=np.int32)
        rates = collision_rate_per_level(codes)
        assert rates[0] == pytest.approx(0.9)
        assert rates[1] == pytest.approx(0.9)
        assert rates[2] == pytest.approx(0.9)

    def test_unique_at_every_level(self):
        codes = np.array([[0, 0], [1, 1]], dtype=np.int32)
        rates = collision_rate_per_level(codes)
        assert rates[0] == 0.0
        assert rates[1] == 0.0


# ---------------------------------------------------------------------------
# evaluate (aggregate)
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.fixture()
    def data(self):
        rng = np.random.RandomState(0)
        X = rng.randn(60, 8).astype(np.float32)
        codes = rng.randint(0, 5, (60, 3)).astype(np.int32)
        return X, codes

    def test_required_keys_present(self, data):
        X, codes = data
        results = evaluate(X, codes)
        assert "n_samples" in results
        assert "n_unique_codes" in results
        assert "collision_rate" in results
        assert "collision_rate_per_level" in results
        assert "recall_at_10" in results
        assert "ndcg_at_10" in results
        assert "distance_correlation" in results
        assert "code_utilization_per_level" in results
        assert "code_entropy_per_level" in results

    def test_n_samples(self, data):
        X, codes = data
        results = evaluate(X, codes)
        assert results["n_samples"] == 60

    def test_collision_rate_in_range(self, data):
        X, codes = data
        results = evaluate(X, codes)
        assert 0.0 <= results["collision_rate"] <= 1.0

    def test_per_level_lists_correct_length(self, data):
        X, codes = data
        results = evaluate(X, codes)
        assert len(results["collision_rate_per_level"]) == 3
        assert len(results["code_utilization_per_level"]) == 3
        assert len(results["code_entropy_per_level"]) == 3

    def test_quantization_mse_with_encoder(self):
        from semantic_id.algorithms.rq_kmeans import RQKMeans

        rng = np.random.RandomState(0)
        X = rng.randn(30, 8).astype(np.float32)
        model = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
        model.fit(X)
        codes = model.encode(X)

        results = evaluate(X, codes, encoder=model, k=5)
        assert "quantization_mse" in results
        assert results["quantization_mse"] >= 0.0

    def test_no_quantization_mse_without_encoder(self, data):
        X, codes = data
        results = evaluate(X, codes)
        assert "quantization_mse" not in results

    def test_custom_k(self, data):
        X, codes = data
        results = evaluate(X, codes, k=3)
        assert "recall_at_3" in results
        assert "ndcg_at_3" in results

    def test_seed_determinism(self, data):
        X, codes = data
        r1 = evaluate(X, codes, seed=7)
        r2 = evaluate(X, codes, seed=7)
        assert r1["recall_at_10"] == r2["recall_at_10"]
        assert r1["distance_correlation"] == r2["distance_correlation"]

    def test_evaluate_quality_alias(self):
        from semantic_id.utils.metrics import evaluate_quality

        assert evaluate_quality is evaluate

    def test_small_n_skips_retrieval(self):
        """When N <= k, recall and ndcg should be skipped gracefully."""
        X = np.random.randn(5, 4).astype(np.float32)
        codes = np.arange(5).reshape(5, 1).astype(np.int32)
        results = evaluate(X, codes, k=10)
        assert "recall_at_10" not in results
        assert "ndcg_at_10" not in results
        assert "collision_rate" in results
