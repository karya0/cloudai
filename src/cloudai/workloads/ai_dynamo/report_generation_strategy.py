# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import shutil
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.lazy_imports import lazy


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    metrics: ClassVar[list[str]] = [
        "default",
        "output-token-throughput-per-gpu",
        *[
            f"{base}:{suffix}"
            for base in [
                "output-token-throughput",
                "request-throughput",
                "time-to-first-token",
                "time-to-second-token",
                "request-latency",
                "inter-token-latency",
            ]
            for suffix in [
                "avg", "min", "max", "p99", "p95", "p90", "p75", "p50", "p25", "p10", "p5", "p1"
            ]
        ],
    ]

    metric_mapping = {
        "default": "Output Token Throughput (tokens/sec)",
        "output-token-throughput": "Output Token Throughput (tokens/sec)",
        "request-throughput": "Request Throughput (per sec)",
        "time-to-first-token": "Time To First Token (ms)",
        "time-to-second-token": "Time To Second Token (ms)",
        "request-latency": "Request Latency (ms)",
        "inter-token-latency": "Inter Token Latency (ms)",
    }

    metric_mapping_per_gpu = {
        "output-token-throughput-per-gpu": "Overall Output Tokens per Second per GPU",
    }

    def can_handle_directory(self) -> bool:
        output_path = self.test_run.output_path
        csv_files = list(output_path.rglob("profile_genai_perf.csv"))
        json_files = list(output_path.rglob("profile_genai_perf.json"))
        return len(csv_files) > 0 and len(json_files) > 0

    def _read_metric_from_csv(self, metric_name: str, metric_type: str) -> float:
        output_path = self.test_run.output_path
        if not output_path.exists() or not output_path.is_dir():
            return METRIC_ERROR

        csv_file = list(output_path.rglob("profile_genai_perf.csv"))
        if len(csv_file) == 0:
            return METRIC_ERROR

        source_csv = csv_file[0]
        if source_csv.stat().st_size == 0:
            return METRIC_ERROR

        # TODO: Repplace the following read_csv with something that handles
        # blocks of csv data in a single file. Alternatively, split the csv file
        # into multiple files based on different blocks of metrics.
        df = lazy.pd.read_csv(source_csv, nrows=11)

        if metric_name not in df["Metric"].values:
            logging.warning(f"Metric {metric_name} not found in the CSV file.")
            return METRIC_ERROR

        if metric_type not in df.columns:
            logging.warning(f"Metric type {metric_type} not found in the CSV file.")
            return METRIC_ERROR

        metric_row = df[df["Metric"] == metric_name]

        if metric_row.empty:
            return METRIC_ERROR

        result = float(metric_row[metric_type].iloc[0].replace(",", ""))
        if lazy.np.isnan(result):
            return METRIC_ERROR

        return result

    def get_total_gpus(self) -> int:
        gpus_per_node = None
        if isinstance(self.system, SlurmSystem):
            gpus_per_node = self.system.gpus_per_node

        if gpus_per_node is None or gpus_per_node == 0:
            logging.warning("gpus_per_node is None or 0, skipping Overall Output Tokens per Second per GPU calculation.")
            return 0

        num_prefill_nodes = self.test_run.test.test_definition.cmd_args.dynamo.num_prefill_nodes
        num_decode_nodes = self.test_run.test.test_definition.cmd_args.dynamo.num_decode_nodes

        logging.info(f"num_prefill_nodes: {num_prefill_nodes}, num_decode_nodes: {num_decode_nodes}, gpus_per_node: {gpus_per_node}")   
        return (num_prefill_nodes + num_decode_nodes) * gpus_per_node

    def get_output_token_throughput_per_gpu(self) -> float:
        output_token_throughput = self._read_metric_from_csv(self.metric_mapping["output-token-throughput"], "avg")
        if output_token_throughput == METRIC_ERROR:
            return METRIC_ERROR

        total_gpus = self.get_total_gpus()
        if total_gpus == 0:
            return METRIC_ERROR

        return output_token_throughput / total_gpus

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

        if metric == "output-token-throughput-per-gpu":
            return self.get_output_token_throughput_per_gpu();

        metric_name = metric
        metric_type = 'avg'

        if ":" in metric:
            metric_name, metric_type = metric.split(":")

        mapped_metric = self.metric_mapping.get(metric_name)
        if not mapped_metric:
            return METRIC_ERROR

        return self._read_metric_from_csv(mapped_metric, metric_type)

    def generate_report(self) -> None:
        output_path = self.test_run.output_path
        source_csv = next(output_path.rglob("profile_genai_perf.csv"))
        target_csv = output_path / "report.csv"

        shutil.copy2(source_csv, target_csv)

        output_token_throughput_per_gpu = self.get_output_token_throughput_per_gpu()
        if output_token_throughput_per_gpu != METRIC_ERROR:
            with open(target_csv, "a") as f:
                f.write(f"{self.metric_mapping_per_gpu['output-token-throughput-per-gpu']},{output_token_throughput_per_gpu}\n")
