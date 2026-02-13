import numpy as np
import pandas as pd
import pytest

from src.pe.distance import CAT_COLS, CAT_QUERY_WEIGHTS, NUMERIC_COLS, WorkloadDistance


def _make_df(n=10, seed=42):
    rng = np.random.RandomState(seed)
    data = {}
    for c in CAT_COLS:
        data[c] = rng.choice(["A", "B", "C"], size=n)
    for c in NUMERIC_COLS:
        data[c] = rng.uniform(0, 100, size=n).astype(np.float32)
    return pd.DataFrame(data)


class TestWorkloadDistance:
    def test_init_sets_weights(self):
        df = _make_df()
        dist = WorkloadDistance(df)
        assert len(dist.cat_cols) == len(CAT_COLS)
        assert len(dist.num_cols) == len(NUMERIC_COLS)
        assert dist.cat_weights.sum() + dist.num_weight * len(dist.num_cols) == pytest.approx(1.0, abs=1e-5)

    def test_self_nearest_neighbor(self):
        df = _make_df(n=5)
        dist = WorkloadDistance(df)
        nn = dist.nearest_neighbors(df, df)
        assert np.array_equal(nn, np.arange(5))

    def test_nn_output_shape(self):
        real = _make_df(n=20, seed=1)
        synth = _make_df(n=30, seed=2)
        dist = WorkloadDistance(real)
        nn = dist.nearest_neighbors(real, synth)
        assert nn.shape == (20,)
        assert nn.min() >= 0
        assert nn.max() < 30

    def test_chunked_same_as_unchunked(self):
        real = _make_df(n=15, seed=3)
        synth = _make_df(n=25, seed=4)
        dist = WorkloadDistance(real)
        nn_small = dist.nearest_neighbors(real, synth, real_chunk=5, synth_chunk=10)
        nn_large = dist.nearest_neighbors(real, synth, real_chunk=100, synth_chunk=100)
        assert np.array_equal(nn_small, nn_large)

    def test_query_weights_applied(self):
        assert CAT_QUERY_WEIGHTS["chassistype"] > CAT_QUERY_WEIGHTS.get("cpuname", 1)
        assert CAT_QUERY_WEIGHTS["countryname_normalized"] > CAT_QUERY_WEIGHTS.get("persona", 1)
