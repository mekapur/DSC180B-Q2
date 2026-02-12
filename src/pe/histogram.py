"""DP nearest-neighbor histogram, candidate selection, and PE orchestration.

Implements Algorithm 2 from Lin et al. (2024): each real record votes for its
nearest synthetic candidate, Gaussian noise is added for differential privacy,
and top-ranked candidates are selected for the next iteration. The
PECheckpoint class provides fault-tolerant stage-based resume for long runs.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import PEApi
from .distance import WorkloadDistance
from .privacy import calibrate_sigma, compute_epsilon

logger = logging.getLogger(__name__)


def dp_nn_histogram(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    dist: WorkloadDistance,
    sigma: float,
    real_chunk: int = 5000,
    synth_chunk: int = 10000,
) -> np.ndarray:
    """Compute a DP nearest-neighbor histogram over synthetic candidates.

    Each real record votes for the synthetic candidate closest to it under
    the workload-aware distance. Gaussian noise N(0, sigma) is added to the
    vote counts for (epsilon, delta)-DP. Negative bins are clamped to zero.

    Parameters
    ----------
    real_df : pd.DataFrame
        Private dataset (one row per record).
    synth_df : pd.DataFrame
        Current synthetic population to score.
    dist : WorkloadDistance
        Pre-fitted distance metric with column weights and normalization.
    sigma : float
        Gaussian noise standard deviation (from privacy calibration).
    real_chunk, synth_chunk : int
        Chunk sizes for the blocked nearest-neighbor computation.

    Returns
    -------
    np.ndarray
        Noisy vote histogram of length len(synth_df), clamped to >= 0.
    """
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
    """Select the top synthetic candidates by noisy vote count.

    Two selection modes are supported:
    - "rank" (default, from Xie et al. 2024): deterministically pick the
      n_select candidates with the highest histogram values. Eliminates
      redundancy from probability-based sampling.
    - "sample": sample n_select candidates with replacement, weighted by
      the histogram. Used in the original PE (Lin et al. 2024).

    Parameters
    ----------
    synth_df : pd.DataFrame
        Synthetic population (same length as histogram).
    histogram : np.ndarray
        Noisy vote counts from dp_nn_histogram.
    n_select : int
        Number of candidates to keep.
    method : str
        "rank" for deterministic top-k, "sample" for weighted sampling.

    Returns
    -------
    pd.DataFrame
        Selected candidates, reset-indexed.
    """
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


class PECheckpoint:
    """Stage-based checkpoint system for fault-tolerant PE runs.

    PE generation involves expensive API calls (RANDOM_API for initial
    population, VARIATION_API for each iteration). If the notebook kernel
    restarts mid-run, this class allows resuming from the last completed
    stage rather than re-generating (and re-paying for) data.

    Stages progress linearly:
        initialized -> population_generated -> histogram_computed -> selected -> done

    Each stage saves its artifacts (parquet, npy, json) to checkpoint_dir.
    The metadata file tracks the current stage and completed iteration index.
    """

    STAGE_INIT = "initialized"
    STAGE_POPULATION = "population_generated"
    STAGE_HISTOGRAM = "histogram_computed"
    STAGE_SELECTED = "selected"
    STAGE_DONE = "done"

    def __init__(self, checkpoint_dir: Path):
        self.dir = checkpoint_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "checkpoint.json"

    def save_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def load_meta(self) -> dict | None:
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f)
        return None

    def save_population(self, df: pd.DataFrame, iteration: int = 0) -> None:
        df.to_parquet(self.dir / f"population_iter{iteration}.parquet", index=False)

    def load_population(self, iteration: int = 0) -> pd.DataFrame | None:
        path = self.dir / f"population_iter{iteration}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_histogram(self, histogram: np.ndarray, iteration: int = 0) -> None:
        np.save(self.dir / f"histogram_iter{iteration}.npy", histogram)

    def load_histogram(self, iteration: int = 0) -> np.ndarray | None:
        path = self.dir / f"histogram_iter{iteration}.npy"
        if path.exists():
            return np.load(path)
        return None

    def save_selected(self, df: pd.DataFrame, iteration: int = 0) -> None:
        df.to_parquet(self.dir / f"selected_iter{iteration}.parquet", index=False)

    def load_selected(self, iteration: int = 0) -> pd.DataFrame | None:
        path = self.dir / f"selected_iter{iteration}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


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
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full Private Evolution loop (Lin et al. 2024, adapted for tabular).

    1. Generate initial population of N_synth * L records via RANDOM_API.
    2. For each iteration t = 1..T:
       a. Compute DP nearest-neighbor histogram (real votes for synth).
       b. Select top N_synth candidates by rank.
       c. If not the last iteration, generate L-1 variations per candidate.
    3. Return the final selected population with synthetic guid identifiers.

    For tabular data, T=1 is optimal (Swanberg et al. 2025; confirmed by
    Gonzalez et al. NeurIPS 2025 convergence theory). With T=1, no
    VARIATION_API calls are made and privacy cost is a single Gaussian
    mechanism application.

    Parameters
    ----------
    real_df : pd.DataFrame
        Private dataset with categorical and numeric columns.
    api : PEApi
        Configured API client for RANDOM_API and VARIATION_API calls.
    n_synth : int
        Target synthetic dataset size.
    T : int
        Number of PE iterations (1 recommended for tabular).
    L : int
        Overgeneration factor: generate L * n_synth candidates per iteration.
    epsilon, delta : float
        Privacy parameters for the overall PE run.
    real_chunk, synth_chunk : int
        Chunk sizes for nearest-neighbor computation.
    batch_size : int
        Records per API call for RANDOM_API.
    variation_batch_size : int
        Source records per API call for VARIATION_API.
    real_subsample : int or None
        If set, subsample the real dataset for voting (reduces NN cost).
    use_batch : bool
        If True, use OpenAI Batch API (50% cheaper, async 24h window).
    work_dir : Path
        Directory for batch job files and temporary artifacts.
    checkpoint_dir : Path or None
        Directory for checkpoint files. Defaults to work_dir/pe_checkpoints.

    Returns
    -------
    (pd.DataFrame, dict)
        Synthetic dataset with guid column, and history dict with timing
        and privacy accounting information.
    """
    ckpt = PECheckpoint(checkpoint_dir or (work_dir / "pe_checkpoints"))
    existing_meta = ckpt.load_meta()

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

    resume_stage = existing_meta.get("stage") if existing_meta else None
    resume_iter = existing_meta.get("completed_iteration", -1) if existing_meta else -1

    if resume_stage and resume_stage != PECheckpoint.STAGE_INIT:
        print(f"\nResuming from checkpoint: stage={resume_stage}, iteration={resume_iter}")

    S_t = None
    if resume_stage in (PECheckpoint.STAGE_POPULATION, PECheckpoint.STAGE_HISTOGRAM, PECheckpoint.STAGE_SELECTED, PECheckpoint.STAGE_DONE):
        loaded = ckpt.load_population(max(resume_iter, 0))
        if loaded is not None:
            S_t = loaded
            print(f"Loaded population from checkpoint: {len(S_t)} records")

    if S_t is None:
        print(f"\n--- Generating initial population (N={n_synth * L}) ---")
        t0 = time.time()
        if use_batch:
            S_t = api.random_api_batch(
                n_synth * L, batch_size=batch_size, work_dir=work_dir
            )
        else:
            S_t = await api.random_api(n_synth * L, batch_size=batch_size)
        gen_time = time.time() - t0
        print(f"Initial population: {len(S_t)} records ({gen_time:.1f}s)")
        if len(S_t) == 0:
            raise RuntimeError(
                "Initial population is empty. All API calls failed. "
                "Check the batch job status and API key."
            )
        ckpt.save_population(S_t, iteration=0)
        ckpt.save_meta({
            "stage": PECheckpoint.STAGE_POPULATION,
            "completed_iteration": -1,
            "population_size": len(S_t),
            "generation_time": gen_time,
        })

    for t in range(T):
        if resume_stage == PECheckpoint.STAGE_DONE:
            break

        if t < resume_iter:
            continue

        if t == resume_iter and resume_stage == PECheckpoint.STAGE_SELECTED:
            loaded_sel = ckpt.load_selected(t)
            if loaded_sel is not None:
                print(f"Loaded selected candidates for iteration {t+1} from checkpoint")
                S_t = loaded_sel
                continue

        print(f"\n--- Iteration {t + 1}/{T} ---")
        iter_info = {"iteration": t + 1}

        histogram = None
        if t == resume_iter and resume_stage == PECheckpoint.STAGE_HISTOGRAM:
            histogram = ckpt.load_histogram(t)
            if histogram is not None:
                print(f"Loaded histogram from checkpoint")
                iter_info["histogram_time"] = 0

        if histogram is None:
            if t == resume_iter and resume_stage == PECheckpoint.STAGE_POPULATION:
                loaded_pop = ckpt.load_population(t)
                if loaded_pop is not None:
                    S_t = loaded_pop
                    print(f"Loaded population for iteration {t+1} from checkpoint")

            t0 = time.time()
            print(f"Computing DP nearest-neighbor histogram ({len(voting_df)} real x {len(S_t)} synth)...")
            histogram = dp_nn_histogram(
                voting_df, S_t, dist, sigma,
                real_chunk=real_chunk, synth_chunk=synth_chunk,
            )
            iter_info["histogram_time"] = time.time() - t0
            print(f"Histogram computed in {iter_info['histogram_time']:.1f}s")
            ckpt.save_histogram(histogram, iteration=t)
            ckpt.save_meta({
                "stage": PECheckpoint.STAGE_HISTOGRAM,
                "completed_iteration": t - 1,
                "current_iteration": t,
            })

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
            ckpt.save_population(S_t, iteration=t + 1)
        else:
            S_t = S_prime

        ckpt.save_selected(S_t, iteration=t)
        ckpt.save_meta({
            "stage": PECheckpoint.STAGE_SELECTED,
            "completed_iteration": t,
        })
        history["iterations"].append(iter_info)

    total_time = time.time() - t0_total
    history["total_time"] = total_time
    actual_eps = compute_epsilon(sigma, delta, T)
    history["actual_epsilon"] = actual_eps
    print(f"\nPE complete: {len(S_t)} synthetic records in {total_time:.1f}s")
    print(f"Actual epsilon: {actual_eps:.4f}")

    S_t.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(S_t))])

    ckpt.save_meta({
        "stage": PECheckpoint.STAGE_DONE,
        "completed_iteration": T - 1,
        "total_time": total_time,
        "actual_epsilon": actual_eps,
        "n_synth_final": len(S_t),
    })

    return S_t, history
