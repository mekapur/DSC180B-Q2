import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import PEApi
from .distance import WorkloadDistance
from .privacy import calibrate_sigma, compute_epsilon


def dp_nn_histogram(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    dist: WorkloadDistance,
    sigma: float,
    real_chunk: int = 5000,
    synth_chunk: int = 10000,
) -> np.ndarray:
    nn_indices = dist.nearest_neighbors(
        real_df, synth_df, real_chunk=real_chunk, synth_chunk=synth_chunk
    )
    n_synth = len(synth_df)
    histogram = np.bincount(nn_indices, minlength=n_synth).astype(np.float64)
    noise = np.random.normal(0, sigma, size=n_synth)
    histogram += noise
    histogram = np.maximum(histogram, 0)
    return histogram


def select_candidates(
    synth_df: pd.DataFrame,
    histogram: np.ndarray,
    n_select: int,
    method: str = "rank",
) -> pd.DataFrame:
    if method == "rank":
        top_indices = np.argsort(histogram)[::-1][:n_select]
    else:
        total = histogram.sum()
        if total <= 0:
            probs = np.ones(len(histogram)) / len(histogram)
        else:
            probs = histogram / total
        top_indices = np.random.choice(
            len(histogram), size=n_select, replace=True, p=probs
        )
    return synth_df.iloc[top_indices].reset_index(drop=True)


async def private_evolution(
    real_df: pd.DataFrame,
    api: PEApi,
    n_synth: int = 50000,
    T: int = 1,
    L: int = 3,
    epsilon: float = 4.0,
    delta: float = 1e-5,
    real_chunk: int = 5000,
    synth_chunk: int = 10000,
    batch_size: int = 20,
    variation_batch_size: int = 5,
    real_subsample: int | None = None,
    use_batch: bool = False,
    work_dir: Path = Path("."),
) -> tuple[pd.DataFrame, dict]:
    if real_subsample is not None and real_subsample < len(real_df):
        voting_df = real_df.sample(real_subsample, random_state=42).reset_index(drop=True)
        print(f"Subsampled {real_subsample:,} from {len(real_df):,} real records for voting")
    else:
        voting_df = real_df

    sigma = calibrate_sigma(epsilon, delta, T)
    mode = "Batch API (50% cheaper)" if use_batch else "Realtime API"
    print(
        f"PE config: N_synth={n_synth}, T={T}, L={L}, "
        f"epsilon={epsilon}, delta={delta}, sigma={sigma:.4f}, "
        f"voting_records={len(voting_df):,}, mode={mode}"
    )

    dist = WorkloadDistance(real_df)
    history = {
        "sigma": sigma,
        "epsilon": epsilon,
        "delta": delta,
        "T": T,
        "L": L,
        "n_synth": n_synth,
        "use_batch": use_batch,
        "iterations": [],
    }

    t0_total = time.time()

    print(f"\n--- Generating initial population (N={n_synth * L}) ---")
    t0 = time.time()
    if use_batch:
        S_t = api.random_api_batch(
            n_synth * L, batch_size=batch_size, work_dir=work_dir
        )
    else:
        S_t = await api.random_api(n_synth * L, batch_size=batch_size)
    print(f"Initial population: {len(S_t)} records ({time.time() - t0:.1f}s)")

    for t in range(T):
        print(f"\n--- Iteration {t + 1}/{T} ---")
        iter_info = {"iteration": t + 1}

        t0 = time.time()
        print(f"Computing DP nearest-neighbor histogram ({len(voting_df)} real x {len(S_t)} synth)...")
        histogram = dp_nn_histogram(
            voting_df, S_t, dist, sigma,
            real_chunk=real_chunk, synth_chunk=synth_chunk,
        )
        iter_info["histogram_time"] = time.time() - t0
        print(f"Histogram computed in {iter_info['histogram_time']:.1f}s")

        nonzero_bins = (histogram > 0).sum()
        iter_info["nonzero_bins"] = int(nonzero_bins)
        print(f"Nonzero bins: {nonzero_bins}/{len(histogram)}")

        t0 = time.time()
        S_prime = select_candidates(S_t, histogram, n_synth, method="rank")
        iter_info["selection_time"] = time.time() - t0
        print(f"Selected top {n_synth} candidates ({iter_info['selection_time']:.1f}s)")

        if t < T - 1:
            t0 = time.time()
            if use_batch:
                variations = api.variation_api_batch(
                    S_prime,
                    n_variations=L - 1,
                    source_batch_size=variation_batch_size,
                    work_dir=work_dir,
                )
            else:
                variations = await api.variation_api(
                    S_prime,
                    n_variations=L - 1,
                    source_batch_size=variation_batch_size,
                )
            S_t = pd.concat([variations, S_prime], ignore_index=True)
            iter_info["variation_time"] = time.time() - t0
            print(
                f"Generated {len(variations)} variations, "
                f"total pool: {len(S_t)} ({iter_info['variation_time']:.1f}s)"
            )
        else:
            S_t = S_prime

        history["iterations"].append(iter_info)

    total_time = time.time() - t0_total
    history["total_time"] = total_time
    actual_eps = compute_epsilon(sigma, delta, T)
    history["actual_epsilon"] = actual_eps
    print(f"\nPE complete: {len(S_t)} synthetic records in {total_time:.1f}s")
    print(f"Actual epsilon: {actual_eps:.4f}")

    S_t.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(S_t))])
    return S_t, history
