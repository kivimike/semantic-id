"""Test that cosine metric raises NotImplementedError."""
import pytest
from semantic_id.algorithms.rq_kmeans import RQKMeans


def test_cosine_metric_raises():
    """Cosine metric is declared but not implemented."""
    with pytest.raises(NotImplementedError, match="Cosine metric is not yet implemented"):
        RQKMeans(n_levels=2, n_clusters=5, metric="cosine")
