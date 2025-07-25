# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal, Optional, Union

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class NCCLCmdArgs(CmdArgs):
    """NCCL test command arguments."""

    docker_image_url: str
    subtest_name: Union[
        Literal[
            "all_reduce_perf_mpi",
            "all_gather_perf_mpi",
            "alltoall_perf_mpi",
            "broadcast_perf_mpi",
            "gather_perf_mpi",
            "hypercube_perf_mpi",
            "reduce_perf_mpi",
            "reduce_scatter_perf_mpi",
            "scatter_perf_mpi",
            "sendrecv_perf_mpi",
            "bisection_perf_mpi",
            # K8s tests
            "all_reduce_perf",
            "all_gather_perf",
            "alltoall_perf",
            "broadcast_perf",
            "gather_perf",
            "hypercube_perf",
            "reduce_perf",
            "reduce_scatter_perf",
            "scatter_perf",
            "sendrecv_perf",
            "bisection_perf",
        ],
        list[
            Literal[
                "all_reduce_perf_mpi",
                "all_gather_perf_mpi",
                "alltoall_perf_mpi",
                "broadcast_perf_mpi",
                "gather_perf_mpi",
                "hypercube_perf_mpi",
                "reduce_perf_mpi",
                "reduce_scatter_perf_mpi",
                "scatter_perf_mpi",
                "sendrecv_perf_mpi",
                "bisection_perf_mpi",
                # K8s tests
                "all_reduce_perf",
                "all_gather_perf",
                "alltoall_perf",
                "broadcast_perf",
                "gather_perf",
                "hypercube_perf",
                "reduce_perf",
                "reduce_scatter_perf",
                "scatter_perf",
                "sendrecv_perf",
                "bisection_perf",
            ]
        ],
    ] = "all_reduce_perf_mpi"
    nthreads: Union[int, list[int]] = 1
    ngpus: Union[int, list[int]] = 1
    minbytes: Union[str, list[str]] = "32M"
    maxbytes: Union[str, list[str]] = "32M"
    stepbytes: Union[str, list[str]] = "1M"
    op: Union[
        Literal["sum", "prod", "min", "max", "avg", "all"], list[Literal["sum", "prod", "min", "max", "avg", "all"]]
    ] = "sum"
    datatype: Union[Literal["uint8", "float"], list[Literal["uint8", "float"]]] = "float"
    root: Union[int, list[int]] = 0
    iters: Union[int, list[int]] = 20
    warmup_iters: Union[int, list[int]] = 5
    agg_iters: Union[int, list[int]] = 1
    average: Union[int, list[int]] = 1
    parallel_init: Union[int, list[int]] = 0
    check: Union[int, list[int]] = 1
    blocking: Union[int, list[int]] = 0
    cudagraph: Union[int, list[int]] = 0
    stepfactor: Optional[Union[int, list[int]]] = None


class NCCLTestDefinition(TestDefinition):
    """Test object for NCCL."""

    cmd_args: NCCLCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.predictor] if self.predictor else [self.docker_image]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stdout_path = tr.output_path / "stdout.txt"
        if stdout_path.is_file():
            with stdout_path.open("r") as file:
                content = file.read()

                # Check for specific error patterns
                if "Test NCCL failure" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"NCCL test failure detected in {stdout_path}. "
                            "Possible reasons include network errors or remote process exits. "
                            "Please review the NCCL test output and errors in the file first. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )
                if "Test failure" in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"Test failure detected in {stdout_path}. "
                            "Please review the specific test failure messages in the file. "
                            "Ensure that the NCCL test environment is correctly set up and configured. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )

                # Check for success indicators
                if "# Out of bounds values" in content and "# Avg bus bandwidth" in content:
                    return JobStatusResult(is_successful=True)

                # Identify missing success indicators
                missing_indicators = []
                if "# Out of bounds values" not in content:
                    missing_indicators.append("'# Out of bounds values'")
                if "# Avg bus bandwidth" not in content:
                    missing_indicators.append("'# Avg bus bandwidth'")

                error_message = (
                    f"Missing success indicators in {stdout_path}: {', '.join(missing_indicators)}. "
                    "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
                    "Please review the NCCL test output and errors in the file. "
                    "Ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
                    f"and check if {stdout_path} is created and contains the expected keywords. "
                    "If the issue persists, contact the system administrator."
                )
                return JobStatusResult(is_successful=False, error_message=error_message)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stdout.txt file not found in the specified output directory {tr.output_path}. "
                "This file is expected to be created as a result of the NCCL test run. "
                "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
                f"You can run the generated NCCL test command manually and verify the creation of {stdout_path}. "
                "If the issue persists, contact the system administrator."
            ),
        )
