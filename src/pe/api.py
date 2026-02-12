import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from .distance import CAT_COLS, NUMERIC_COLS


class TelemetryRecord(BaseModel):
    chassistype: str
    countryname_normalized: str
    modelvendor_normalized: str
    os: str
    cpuname: str
    cpucode: str
    cpu_family: str
    persona: str
    processornumber: str
    ram: float
    net_nrs: float
    net_received_bytes: float
    net_sent_bytes: float
    mem_nrs: float
    mem_avg_pct_used: float
    mem_sysinfo_ram: float
    batt_num_power_ons: float
    batt_duration_mins: float
    web_chrome_duration: float
    web_edge_duration: float
    web_firefox_duration: float
    web_total_duration: float
    web_num_instances: float
    webcat_content_creation_photo_edit_creation: float
    webcat_content_creation_video_audio_edit_creation: float
    webcat_content_creation_web_design_development: float
    webcat_education: float
    webcat_entertainment_music_audio_streaming: float
    webcat_entertainment_other: float
    webcat_entertainment_video_streaming: float
    webcat_finance: float
    webcat_games_other: float
    webcat_games_video_games: float
    webcat_mail: float
    webcat_news: float
    webcat_unclassified: float
    webcat_private: float
    webcat_productivity_crm: float
    webcat_productivity_other: float
    webcat_productivity_presentations: float
    webcat_productivity_programming: float
    webcat_productivity_project_management: float
    webcat_productivity_spreadsheets: float
    webcat_productivity_word_processing: float
    webcat_recreation_travel: float
    webcat_reference: float
    webcat_search: float
    webcat_shopping: float
    webcat_social_social_network: float
    webcat_social_communication: float
    webcat_social_communication_live: float
    onoff_on_time: float
    onoff_off_time: float
    onoff_mods_time: float
    onoff_sleep_time: float
    disp_num_displays: float
    disp_total_duration_ac: float
    disp_total_duration_dc: float
    psys_rap_nrs: float
    psys_rap_avg: float
    pkg_c0_nrs: float
    pkg_c0_avg: float
    avg_freq_nrs: float
    avg_freq_avg: float
    temp_nrs: float
    temp_avg: float
    pkg_power_nrs: float
    pkg_power_avg: float


class RecordsBatch(BaseModel):
    records: list[TelemetryRecord]


INSTRUCTIONS = (
    "You generate synthetic tabular data. "
    "Produce realistic, diverse records following the schema and sparsity rules exactly."
)


def _build_schema_description(real_df: pd.DataFrame) -> str:
    cat_info = []
    for c in CAT_COLS:
        if c not in real_df.columns:
            continue
        top_vals = real_df[c].value_counts().head(10).index.tolist()
        cat_info.append(f"{c}: {json.dumps(top_vals)}")

    num_groups = []
    groups = {
        "net": (["net_nrs", "net_received_bytes", "net_sent_bytes"], 4),
        "mem": (["mem_nrs", "mem_avg_pct_used", "mem_sysinfo_ram"], 7),
        "batt": (["batt_num_power_ons", "batt_duration_mins"], 2),
        "browser": (
            [
                "web_chrome_duration",
                "web_edge_duration",
                "web_firefox_duration",
                "web_total_duration",
                "web_num_instances",
            ],
            6,
        ),
        "webcat": ([c for c in NUMERIC_COLS if c.startswith("webcat_")], 5),
        "onoff": (
            [
                "onoff_on_time",
                "onoff_off_time",
                "onoff_mods_time",
                "onoff_sleep_time",
            ],
            4,
        ),
        "disp": (
            ["disp_num_displays", "disp_total_duration_ac", "disp_total_duration_dc"],
            21,
        ),
        "hw": (
            [
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
            ],
            1,
        ),
    }
    for gname, (cols, pct) in groups.items():
        present = [c for c in cols if c in real_df.columns]
        if present:
            num_groups.append(f"{gname} ({pct}% of systems): {', '.join(present)}")

    return (
        "Intel telemetry. Each record represents one client system.\n"
        "Categorical (pick from list): "
        + "; ".join(cat_info)
        + "\nram: integer GB (4/8/16/32/64), always present."
        + "\nNumeric groups (all float>=0): "
        + "; ".join(num_groups)
        + "\nRULE: each system has data in 1-2 groups only. "
        "Set all other groups to 0. hw group is <1% of systems."
    )


def _build_random_prompt(schema_desc: str, batch_size: int) -> str:
    return (
        f"Generate exactly {batch_size} synthetic telemetry records.\n\n"
        f"Schema:\n{schema_desc}\n\n"
        f"Generate diverse, realistic records. Follow the sparsity rules strictly."
    )


def _build_variation_prompt(
    schema_desc: str, records: list[dict], n_variations: int
) -> str:
    return (
        f"Below are {len(records)} telemetry records. For each, generate {n_variations} "
        f"variation(s) that are similar but slightly different.\n\n"
        f"Rules for variations:\n"
        f"- Categorical values: may change to a similar value (e.g., nearby country, same OS family)\n"
        f"- Numeric values: perturb by 10-30% (multiply by a random factor between 0.7 and 1.3)\n"
        f"- Zero values MUST remain zero (preserve sparsity pattern)\n"
        f"- Non-zero values should remain non-zero\n"
        f"- ram should stay at a standard size (4, 8, 16, 32, 64)\n\n"
        f"Schema:\n{schema_desc}\n\n"
        f"Source records:\n{json.dumps(records, indent=None)}\n\n"
        f"Return {len(records) * n_variations} variation records total."
    )


def _make_strict_schema() -> dict:
    schema = RecordsBatch.model_json_schema()

    def enforce_strict(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
            for v in obj.values():
                enforce_strict(v)
        elif isinstance(obj, list):
            for item in obj:
                enforce_strict(item)

    enforce_strict(schema)
    if "$defs" in schema:
        for defn in schema["$defs"].values():
            enforce_strict(defn)
    return schema


class PEApi:
    def __init__(
        self,
        real_df: pd.DataFrame,
        model: str = "gpt-5-nano",
        max_concurrent: int = 50,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=api_key, max_retries=5)
        self.sync_client = OpenAI(api_key=api_key, max_retries=5)
        self.model = model
        self.max_concurrent = max_concurrent
        self.schema_desc = _build_schema_description(real_df)
        self.all_cols = CAT_COLS + NUMERIC_COLS
        self.present_cols = [c for c in self.all_cols if c in real_df.columns]
        self._is_reasoning = model.startswith("gpt-5") or model.startswith("o")
        self._strict_schema = _make_strict_schema()

    # ------------------------------------------------------------------ #
    #  Real-time API (async, for smoke tests and small runs)              #
    # ------------------------------------------------------------------ #

    async def _call_api(self, prompt: str) -> list[dict]:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": INSTRUCTIONS,
                "input": prompt,
                "text_format": RecordsBatch,
                "max_output_tokens": 16000,
            }
            if self._is_reasoning:
                kwargs["reasoning"] = {"effort": "minimal"}
            else:
                kwargs["temperature"] = 1.0
            response = await self.client.responses.parse(**kwargs)
            if response.output_parsed and response.output_parsed.records:
                return [r.model_dump() for r in response.output_parsed.records]
        except LengthFinishReasonError:
            pass
        except Exception as e:
            print(f"  API error: {e}")
        return []

    async def _batch_calls(
        self, prompts: list[str], desc: str = ""
    ) -> list[list[dict]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[list[dict]] = [[] for _ in range(len(prompts))]
        completed = 0
        total = len(prompts)
        t0 = time.time()

        async def run_one(idx: int, prompt: str):
            nonlocal completed
            async with semaphore:
                result = await self._call_api(prompt)
                results[idx] = result
                completed += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  {desc}: {completed}/{total} "
                        f"({elapsed:.0f}s, {rate:.1f}/s)"
                    )

        tasks = [run_one(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    def _records_to_df(self, records: list[dict]) -> pd.DataFrame:
        rows = []
        for r in records:
            row = {}
            for c in self.present_cols:
                val = r.get(c, 0 if c in NUMERIC_COLS else "Unknown")
                row[c] = val
            rows.append(row)

        df = pd.DataFrame(rows)
        for c in [col for col in NUMERIC_COLS if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        for c in [col for col in CAT_COLS if col in df.columns]:
            df[c] = df[c].astype(str).fillna("Unknown")
        return df

    async def random_api(
        self, n_records: int, batch_size: int = 10
    ) -> pd.DataFrame:
        overshoot = int(n_records * 1.25)
        n_batches = (overshoot + batch_size - 1) // batch_size
        prompts = [
            _build_random_prompt(self.schema_desc, batch_size)
            for _ in range(n_batches)
        ]
        print(
            f"RANDOM_API: {n_records} records "
            f"({n_batches} batches of {batch_size}, 25% buffer)"
        )
        all_results = await self._batch_calls(prompts, desc="RANDOM_API")
        all_records = [r for batch in all_results for r in batch]
        raw = len(all_records)
        df = self._records_to_df(all_records[:n_records])
        print(
            f"RANDOM_API: {raw} raw -> {len(df)} returned "
            f"({100 * len(df) / max(raw, 1):.0f}%)"
        )
        return df

    async def variation_api(
        self,
        source_df: pd.DataFrame,
        n_variations: int = 2,
        source_batch_size: int = 5,
    ) -> pd.DataFrame:
        source_records = source_df[self.present_cols].to_dict(orient="records")
        n_batches = (len(source_records) + source_batch_size - 1) // source_batch_size
        prompts = []
        for i in range(n_batches):
            start = i * source_batch_size
            end = min(start + source_batch_size, len(source_records))
            batch = source_records[start:end]
            prompts.append(
                _build_variation_prompt(self.schema_desc, batch, n_variations)
            )
        print(
            f"VARIATION_API: {n_variations} variations for "
            f"{len(source_df)} records ({n_batches} batches)"
        )
        all_results = await self._batch_calls(prompts, desc="VARIATION_API")
        all_records = [r for batch in all_results for r in batch]
        df = self._records_to_df(all_records)
        print(f"VARIATION_API: {len(df)} records")
        return df

    # ------------------------------------------------------------------ #
    #  Batch API (sync, 50% cheaper, for full production runs)            #
    # ------------------------------------------------------------------ #

    def _write_batch_jsonl(self, prompts: list[str], path: Path) -> None:
        with open(path, "w") as f:
            for i, prompt in enumerate(prompts):
                body: dict[str, Any] = {
                    "model": self.model,
                    "instructions": INSTRUCTIONS,
                    "input": prompt,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "RecordsBatch",
                            "schema": self._strict_schema,
                            "strict": True,
                        }
                    },
                    "max_output_tokens": 16000,
                }
                if self._is_reasoning:
                    body["reasoning"] = {"effort": "minimal"}
                else:
                    body["temperature"] = 1.0
                request = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(request) + "\n")

    def _submit_batch(self, jsonl_path: Path, desc: str = "") -> str:
        with open(jsonl_path, "rb") as f:
            uploaded = self.sync_client.files.create(file=f, purpose="batch")
        job = self.sync_client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        total = job.request_counts.total if job.request_counts else "?"
        print(f"  {desc}: batch {job.id} submitted ({total} requests)")
        return job.id

    def _poll_batch(
        self, batch_id: str, desc: str = "", poll_interval: int = 30
    ) -> Any:
        while True:
            status = self.sync_client.batches.retrieve(batch_id)
            rc = status.request_counts
            if rc:
                print(
                    f"  {desc}: {rc.completed}/{rc.total} done, "
                    f"{rc.failed} failed [{status.status}]"
                )
            else:
                print(f"  {desc}: [{status.status}]")
            if status.status in ("completed", "failed", "cancelled", "expired"):
                return status
            time.sleep(poll_interval)

    def _parse_batch_output(self, output_file_id: str) -> list[list[dict]]:
        file_content = self.sync_client.files.content(output_file_id)
        indexed: list[tuple[int, list[dict]]] = []
        for line in file_content.text.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["custom_id"].split("-")[1])
            records: list[dict] = []
            resp = obj.get("response", {})
            if resp.get("status_code") == 200:
                body = resp.get("body", {})
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        for ci in item.get("content", []):
                            if ci.get("type") == "output_text":
                                text = ci.get("text", "")
                                if text:
                                    try:
                                        batch = RecordsBatch.model_validate_json(
                                            text
                                        )
                                        records = [
                                            r.model_dump() for r in batch.records
                                        ]
                                    except Exception:
                                        pass
            indexed.append((idx, records))
        indexed.sort(key=lambda x: x[0])
        return [recs for _, recs in indexed]

    def random_api_batch(
        self,
        n_records: int,
        batch_size: int = 10,
        work_dir: Path = Path("."),
    ) -> pd.DataFrame:
        overshoot = int(n_records * 1.25)
        n_batches = (overshoot + batch_size - 1) // batch_size
        prompts = [
            _build_random_prompt(self.schema_desc, batch_size)
            for _ in range(n_batches)
        ]
        print(
            f"RANDOM_API_BATCH: {n_records} records "
            f"({n_batches} calls, 25% buffer)"
        )
        jsonl_path = work_dir / "batch_random_input.jsonl"
        self._write_batch_jsonl(prompts, jsonl_path)
        batch_id = self._submit_batch(jsonl_path, desc="RANDOM")
        status = self._poll_batch(batch_id, desc="RANDOM")

        if status.status != "completed":
            print(f"RANDOM_API_BATCH: failed ({status.status})")
            return pd.DataFrame()

        all_results = self._parse_batch_output(status.output_file_id)
        all_records = [r for batch in all_results for r in batch]
        raw = len(all_records)
        df = self._records_to_df(all_records[:n_records])
        print(
            f"RANDOM_API_BATCH: {raw} raw -> {len(df)} returned "
            f"({100 * len(df) / max(raw, 1):.0f}%)"
        )
        return df

    def variation_api_batch(
        self,
        source_df: pd.DataFrame,
        n_variations: int = 2,
        source_batch_size: int = 5,
        work_dir: Path = Path("."),
    ) -> pd.DataFrame:
        source_records = source_df[self.present_cols].to_dict(orient="records")
        n_batches = (
            len(source_records) + source_batch_size - 1
        ) // source_batch_size
        prompts = []
        for i in range(n_batches):
            start = i * source_batch_size
            end = min(start + source_batch_size, len(source_records))
            prompts.append(
                _build_variation_prompt(
                    self.schema_desc, source_records[start:end], n_variations
                )
            )
        print(
            f"VARIATION_API_BATCH: {n_variations} variations for "
            f"{len(source_df)} records ({n_batches} calls)"
        )
        jsonl_path = work_dir / "batch_variation_input.jsonl"
        self._write_batch_jsonl(prompts, jsonl_path)
        batch_id = self._submit_batch(jsonl_path, desc="VARIATION")
        status = self._poll_batch(batch_id, desc="VARIATION")

        if status.status != "completed":
            print(f"VARIATION_API_BATCH: failed ({status.status})")
            return pd.DataFrame()

        all_results = self._parse_batch_output(status.output_file_id)
        all_records = [r for batch in all_results for r in batch]
        df = self._records_to_df(all_records)
        print(f"VARIATION_API_BATCH: {len(df)} records")
        return df
