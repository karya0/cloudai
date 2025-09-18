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

import csv
import logging
import shutil
from pathlib import Path
from typing import ClassVar
import pandas as pd

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.systems.slurm.slurm_system import SlurmSystem


class AIDynamoReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from AI Dynamo run directories."""

    def _find_csv_file(self) -> Path | None:
        output_path = self.test_run.output_path
        if not output_path.exists() or not output_path.is_dir():
            return None

        csv_files = list(output_path.rglob(self.test_run.test.test_definition.cmd_args.dynamo.report_csv))
        if not csv_files or csv_files[0].stat().st_size == 0:
            return None

        return csv_files[0]

    def get_metric(self, metric: str) -> float:
        metric_name = metric
        metric_type = 'avg'

        if ":" in metric:
            metric_name, metric_type = metric.split(":")

        source_csv = self._find_csv_file()
        if not source_csv:
            return METRIC_ERROR

        df = pd.read_csv(source_csv)
        if metric_type not in df.columns:
            return METRIC_ERROR

        if not metric_name in df["Metric"].values:
            return METRIC_ERROR

        return float(df[df["Metric"] == metric_name][metric_type].values[0])

    def generate_report(self) -> None:
        pass
