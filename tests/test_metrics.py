import numpy as np
import pytest

from semantic_id.utils.metrics import (
    distance_correlation,
    evaluate,
    find_similar,
    hierarchical_distance,
    recall_at_k,
    _hierarchical_metric,
)


# ---------------------------------------------------------------------------
# hierarchical_distance
# ---------------------------------------------------------------------------


class TestHierarchicalDistance:
    def test_identical_codes(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        assert hierarchical_distance(a, b) == 0

    def test_first_level_differs(self):
        a = np.array([1, 2, 3])
        b = np.array([9, 2, 3])
        # L1 differs -> distance = L = 3
        assert hierarchical_distance(a, b) == 3

    def test_second_level_differs(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 9, 3])
        # L1 matches, L2 differs -> distance = L - 1 = 2
        assert hierarchical_distance(a, b) == 2

    def test_last_level_differs(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 9])
        # Only L3 differs -> distance = 1
        assert hierarchical_distance(a, b) == 1

    def test_hierarchy_property(self):
        """L1 mismatch should always produce a larger distance than L2/L3."""
        base = np.array([1, 2, 3])
        diff_l1 = np.array([9, 2, 3])
        diff_l2 = np.array([1, 9, 3])
        diff_l3 = np.array([1, 2, 9])

        d1 = hierarchical_distance(base, diff_l1)
        d2 = hierarchical_distance(base, diff_l2)
        d3 = hierarchical_distance(base, diff_l3)

        assert d1 > d2 > d3

    def test_later_matches_ignored_after_mismatch(self):
        """If L1 differs, matching L2/L3 should NOT reduce the distance."""
        a = np.array([1, 5, 7])
        b = np.array([2, 5, 7])  # L1 differs, L2 and L3 match
        c = np.array([2, 9, 9])  # L1 differs, L2 and L3 also differ

        assert hierarchical_distance(a, b) == hierarchical_distance(a, c)

    def test_vectorized_batch(self):
        codes_a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        codes_b = np.array([[1, 2, 3], [1, 9, 3], [9, 2, 3]])
        dists = hierarchical_distance(codes_a, codes_b)

        np.testing.assert_array_equal(dists, [0, 2, 3])

    def test_broadcast_single_query(self):
        query = np.array([[1, 2, 3]])
        codes = np.array([[1, 2, 3], [1, 2, 9], [1, 9, 3], [9, 2, 3]])
        dists = hierarchical_distance(query, codes)

        np.testing.assert_array_equal(dists.ravel(), [0, 1, 2, 3])


class TestHierarchicalMetricScalar:
    def test_matches_vectorized(self):
        pairs = [
            ([1, 2, 3], [1, 2, 3]),
            ([1, 2, 3], [9, 2, 3]),
            ([1, 2, 3], [1, 9, 3]),
            ([1, 2, 3], [1, 2, 9]),
        ]
        for a, b in pairs:
            expected = int(hierarchical_distance(np.array(a), np.array(b)))
            got = _hierarchical_metric(np.array(a), np.array(b))
            assert got == expected


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    @pytest.fixture()
    def sample_codes(self):
        return np.array(
            [
                [0, 0, 0],  # idx 0
                [0, 0, 1],  # idx 1 — distance 1 from 0
                [0, 1, 0],  # idx 2 — distance 2 from 0
                [1, 0, 0],  # idx 3 — distance 3 from 0
                [0, 0, 2],  # idx 4 — distance 1 from 0
            ]
        )

    def test_sorted_by_distance(self, sample_codes):
        indices, distances = find_similar(sample_codes, query=0, k=4)

        assert len(indices) == 4
        assert list(distances) == sorted(distances)

    def test_excludes_self(self, sample_codes):
        indices, distances = find_similar(sample_codes, query=0, k=4)
        assert 0 not in indices

    def test_correct_order(self, sample_codes):
        indices, distances = find_similar(sample_codes, query=0, k=4)

        np.testing.assert_array_equal(distances[:2], [1, 1])
        assert set(indices[:2]) == {1, 4}
        assert indices[2] == 2
        assert distances[2] == 2
        assert indices[3] == 3
        assert distances[3] == 3

    def test_query_as_code_vector(self, sample_codes):
        query_code = np.array([0, 0, 0])
        indices, distances = find_similar(sample_codes, query=query_code, k=3)

        assert len(indices) == 3
        # idx 0 is NOT excluded because query is a vector, not an index
        assert distances[0] == 0

    def test_k_larger_than_dataset(self, sample_codes):
        indices, distances = find_similar(sample_codes, query=0, k=100)
        # Should return all items except self
        assert len(indices) == len(sample_codes) - 1


# ---------------------------------------------------------------------------
# recall_at_k / distance_correlation (smoke tests)
# ---------------------------------------------------------------------------


class TestMetricsFunctions:
    @pytest.fixture()
    def data(self):
        rng = np.random.RandomState(0)
        N, D = 200, 16
        X = rng.randn(N, D).astype(np.float32)

        from semantic_id.algorithms.rq_kmeans import RQKMeans

        model = RQKMeans(n_levels=3, n_clusters=8, random_state=0)
        model.fit(X)
        codes = model.encode(X)
        return X, codes, model

    def test_recall_at_k_hierarchical(self, data):
        X, codes, _ = data
        r = recall_at_k(X, codes, k=5, metric="hierarchical")
        assert 0.0 <= r <= 1.0

    def test_recall_at_k_hamming(self, data):
        X, codes, _ = data
        r = recall_at_k(X, codes, k=5, metric="hamming")
        assert 0.0 <= r <= 1.0

    def test_distance_correlation_hierarchical(self, data):
        X, codes, _ = data
        c = distance_correlation(X, codes, metric="hierarchical")
        assert -1.0 <= c <= 1.0

    def test_distance_correlation_hamming(self, data):
        X, codes, _ = data
        c = distance_correlation(X, codes, metric="hamming")
        assert -1.0 <= c <= 1.0

    def test_evaluate_returns_expected_keys(self, data):
        X, codes, model = data
        results = evaluate(X, codes, encoder=model)
        assert "recall_at_10" in results
        assert "distance_correlation" in results
        assert "collision_rate" in results
