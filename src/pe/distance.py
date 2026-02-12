import numpy as np
import pandas as pd


CAT_COLS = [
    "chassistype",
    "countryname_normalized",
    "modelvendor_normalized",
    "os",
    "cpuname",
    "cpucode",
    "cpu_family",
    "persona",
    "processornumber",
]

NUMERIC_COLS = [
    "ram",
    "net_nrs",
    "net_received_bytes",
    "net_sent_bytes",
    "mem_nrs",
    "mem_avg_pct_used",
    "mem_sysinfo_ram",
    "batt_num_power_ons",
    "batt_duration_mins",
    "web_chrome_duration",
    "web_edge_duration",
    "web_firefox_duration",
    "web_total_duration",
    "web_num_instances",
    "webcat_content_creation_photo_edit_creation",
    "webcat_content_creation_video_audio_edit_creation",
    "webcat_content_creation_web_design_development",
    "webcat_education",
    "webcat_entertainment_music_audio_streaming",
    "webcat_entertainment_other",
    "webcat_entertainment_video_streaming",
    "webcat_finance",
    "webcat_games_other",
    "webcat_games_video_games",
    "webcat_mail",
    "webcat_news",
    "webcat_unclassified",
    "webcat_private",
    "webcat_productivity_crm",
    "webcat_productivity_other",
    "webcat_productivity_presentations",
    "webcat_productivity_programming",
    "webcat_productivity_project_management",
    "webcat_productivity_spreadsheets",
    "webcat_productivity_word_processing",
    "webcat_recreation_travel",
    "webcat_reference",
    "webcat_search",
    "webcat_shopping",
    "webcat_social_social_network",
    "webcat_social_communication",
    "webcat_social_communication_live",
    "onoff_on_time",
    "onoff_off_time",
    "onoff_mods_time",
    "onoff_sleep_time",
    "disp_num_displays",
    "disp_total_duration_ac",
    "disp_total_duration_dc",
    "psys_rap_nrs",
    "psys_rap_avg",
    "pkg_c0_nrs",
    "pkg_c0_avg",
    "avg_freq_nrs",
    "avg_freq_avg",
    "temp_nrs",
    "temp_avg",
    "pkg_power_nrs",
    "pkg_power_avg",
]

CAT_QUERY_WEIGHTS = {
    "chassistype": 6,
    "countryname_normalized": 4,
    "os": 2,
    "persona": 1,
    "cpu_family": 1,
    "processornumber": 1,
    "cpuname": 1,
    "cpucode": 1,
    "modelvendor_normalized": 1,
}


class WorkloadDistance:
    def __init__(self, real_df: pd.DataFrame):
        self.cat_cols = [c for c in CAT_COLS if c in real_df.columns]
        self.num_cols = [c for c in NUMERIC_COLS if c in real_df.columns]

        self.num_min = np.zeros(len(self.num_cols), dtype=np.float32)
        self.num_range = np.ones(len(self.num_cols), dtype=np.float32)
        for j, c in enumerate(self.num_cols):
            cmin = float(real_df[c].min())
            cmax = float(real_df[c].max())
            self.num_min[j] = cmin
            rng = cmax - cmin
            self.num_range[j] = rng if rng > 1e-10 else 1e-10

        raw_cat_w = np.array(
            [CAT_QUERY_WEIGHTS.get(c, 1) for c in self.cat_cols], dtype=np.float32
        )
        total_weight = raw_cat_w.sum() + len(self.num_cols)
        self.cat_weights = raw_cat_w / total_weight
        self.num_weight = 1.0 / total_weight

    def _encode_cat_codes(self, df: pd.DataFrame) -> np.ndarray:
        codes = np.empty((len(df), len(self.cat_cols)), dtype="U64")
        for j, c in enumerate(self.cat_cols):
            codes[:, j] = df[c].astype(str).values
        return codes

    def _encode_num(self, df: pd.DataFrame) -> np.ndarray:
        arr = np.zeros((len(df), len(self.num_cols)), dtype=np.float32)
        for j, c in enumerate(self.num_cols):
            arr[:, j] = df[c].values.astype(np.float32)
        arr = (arr - self.num_min) / self.num_range
        return np.clip(arr, 0, 1)

    def nearest_neighbors(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        real_chunk: int = 5000,
        synth_chunk: int = 10000,
    ) -> np.ndarray:
        from scipy.spatial.distance import cdist

        real_cat = self._encode_cat_codes(real_df)
        real_num = self._encode_num(real_df)
        synth_cat = self._encode_cat_codes(synth_df)
        synth_num = self._encode_num(synth_df)

        real_num_w = real_num * self.num_weight
        synth_num_w = synth_num * self.num_weight

        cat_int_real = np.empty((len(real_df), len(self.cat_cols)), dtype=np.int32)
        cat_int_synth = np.empty((len(synth_df), len(self.cat_cols)), dtype=np.int32)
        vocab = {}
        for j in range(len(self.cat_cols)):
            all_vals = np.unique(np.concatenate([real_cat[:, j], synth_cat[:, j]]))
            mapping = {v: i for i, v in enumerate(all_vals)}
            vocab[j] = mapping
            cat_int_real[:, j] = np.array([mapping[v] for v in real_cat[:, j]], dtype=np.int32)
            cat_int_synth[:, j] = np.array([mapping[v] for v in synth_cat[:, j]], dtype=np.int32)

        n_real = len(real_df)
        n_synth = len(synth_df)
        nn_indices = np.empty(n_real, dtype=np.int64)

        for r_start in range(0, n_real, real_chunk):
            r_end = min(r_start + real_chunk, n_real)
            r_size = r_end - r_start

            best_idx = np.full(r_size, -1, dtype=np.int64)
            best_dist = np.full(r_size, np.inf, dtype=np.float32)

            for s_start in range(0, n_synth, synth_chunk):
                s_end = min(s_start + synth_chunk, n_synth)

                num_dist = cdist(
                    real_num_w[r_start:r_end],
                    synth_num_w[s_start:s_end],
                    metric="cityblock",
                ).astype(np.float32)

                for j in range(len(self.cat_cols)):
                    mismatch = (
                        cat_int_real[r_start:r_end, j:j+1]
                        != cat_int_synth[s_start:s_end, j]
                    )
                    num_dist += self.cat_weights[j] * mismatch.astype(np.float32)

                block_min_idx = num_dist.argmin(axis=1)
                block_min_dist = num_dist[np.arange(r_size), block_min_idx]

                improved = block_min_dist < best_dist
                best_dist[improved] = block_min_dist[improved]
                best_idx[improved] = block_min_idx[improved] + s_start

            nn_indices[r_start:r_end] = best_idx

        return nn_indices
