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

from typing import Any, Dict, List, Union
from unittest.mock import Mock

import pytest

from cloudai.core import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestSlurmCommandGenStrategy


class TestNcclTestSlurmCommandGenStrategy:
    @pytest.mark.parametrize(
        "env_vars, num_nodes, nodes, expected_result",
        [
            (
                {"NCCL_TOPO_FILE": "/path/to/topo"},
                2,
                ["node1", "node2"],
                {
                    "container_mounts": "/path/to/topo:/path/to/topo",
                },
            ),
            (
                {"NCCL_TOPO_FILE": "/path/to/topo"},
                1,
                ["node1"],
                {
                    "container_mounts": "/path/to/topo:/path/to/topo",
                },
            ),
        ],
    )
    def test_parse_slurm_args(
        self,
        slurm_system: SlurmSystem,
        env_vars: Dict[str, Union[str, List[str]]],
        num_nodes: int,
        nodes: List[str],
        expected_result: Dict[str, Any],
    ) -> None:
        nccl = NCCLTestDefinition(
            name="name",
            description="desc",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
            extra_env_vars=env_vars,
        )
        t = Test(test_definition=nccl, test_template=Mock())
        tr = TestRun(name="t1", test=t, nodes=nodes, num_nodes=num_nodes)
        cmd_gen_strategy = NcclTestSlurmCommandGenStrategy(slurm_system, tr)
        assert expected_result["container_mounts"] in cmd_gen_strategy.container_mounts()

    @pytest.mark.parametrize(
        "args",
        [
            {"nthreads": "4", "ngpus": "2"},
            {"R": "1", "G": "0"},
            {"stepfactor": None},
        ],
    )
    def test_generate_test_command(self, slurm_system: SlurmSystem, args: dict[str, Union[str, list[str]]]) -> None:
        cmd_args: NCCLCmdArgs = NCCLCmdArgs.model_validate({**{"docker_image_url": "fake_image_url"}, **args})
        nccl = NCCLTestDefinition(name="name", description="desc", test_template_name="NcclTest", cmd_args=cmd_args)
        t = Test(test_definition=nccl, test_template=Mock())
        tr = TestRun(name="t1", test=t, nodes=[], num_nodes=1)

        cmd_gen_strategy = NcclTestSlurmCommandGenStrategy(slurm_system, tr)
        cmd = " ".join(cmd_gen_strategy.generate_test_command())

        for arg in args:
            if args[arg] is None:
                assert arg not in cmd
            else:
                cli_key = f"--{arg}" if len(arg) > 1 else f"-{arg}"
                assert f"{cli_key} {args[arg]}" in cmd
