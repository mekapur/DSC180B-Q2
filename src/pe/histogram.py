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


class PECheckpoint:
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
