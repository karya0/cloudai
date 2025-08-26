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

import logging
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, FieldValidationInfo, field_serializer, field_validator

from cloudai.core import DockerImage, File, GitRepo, Installable, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class WorkerBaseArgs(BaseModel):
    """Base arguments for VLLM workers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    num_nodes: Union[int, list[int]] = Field(alias="num-nodes")
    nodes: Optional[str] = Field(default=None, alias="nodes")


class PrefillWorkerArgs(WorkerBaseArgs):
    """Arguments for prefill worker."""

    pass


class DecodeWorkerArgs(WorkerBaseArgs):
    """Arguments for decode worker."""

    pass


class AIDynamoArgs(BaseModel):
    """Arguments for AI Dynamo setup."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    backend: str = "vllm"
    prefill_worker: PrefillWorkerArgs
    decode_worker: DecodeWorkerArgs
    #num_prefill_nodes: Union[int, list[int]] = Field(alias="num-prefill-nodes")
    #num_decode_nodes: Union[int, list[int]] = Field(alias="num-decode-nodes")


class GenAIPerfArgs(BaseModel):
    """Arguments for GenAI performance profiling."""

    model_config = ConfigDict(extra="allow")


class AIDynamoCmdArgs(CmdArgs):
    """Arguments for AI Dynamo."""

    docker_image_url: str
    huggingface_home_host_path: Path = Path.home() / ".cache/huggingface"
    huggingface_home_container_path: Path = Path("/root/.cache/huggingface")
    skip_huggingface_home_host_path_validation: bool = False
    dynamo: AIDynamoArgs
    genai_perf: GenAIPerfArgs
    gpus_per_node: int

    @field_serializer("huggingface_home_host_path", "huggingface_home_container_path")
    def _path_serializer(self, v: Path) -> str:
        return str(v.absolute())


class AIDynamoTestDefinition(TestDefinition):
    """Test definition for AI Dynamo."""

    cmd_args: AIDynamoCmdArgs
    _docker_image: Optional[DockerImage] = None
    script: File = File(Path(__file__).parent.parent / "ai_dynamo/ai_dynamo.sh")
    dynamo_repo: GitRepo = GitRepo(
        url="https://github.com/ai-dynamo/dynamo.git", commit="f7e468c7e8ff0d1426db987564e60572167e8464"
    )
    constraints: list[str] = []

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.script, self.dynamo_repo]

    @property
    def huggingface_home_host_path(self) -> Path:
        path = Path(self.cmd_args.huggingface_home_host_path)
        if not self.cmd_args.skip_huggingface_home_host_path_validation and not path.is_dir():
            raise FileNotFoundError(f"HuggingFace home path not found at {path}")
        return path

    @property
    def get_total_gpus(self) -> int:
        gpus_per_node = self.cmd_args.gpus_per_node

        if gpus_per_node is None or gpus_per_node == 0:
            logging.warning("gpus_per_node is None or 0, skipping Overall Output Tokens per Second per GPU calculation.")
            return 0

        num_prefill_nodes = self.cmd_args.dynamo.prefill_worker.num_nodes
        num_decode_nodes = self.cmd_args.dynamo.decode_worker.num_nodes

        return (num_prefill_nodes + num_decode_nodes) * gpus_per_node


    def constraint_check(self, tr: TestRun) -> bool:

        dynamo_args = tr.test.test_definition.cmd_args.dynamo.model_dump(by_alias=True)
        prefill_args = tr.test.test_definition.cmd_args.dynamo.prefill_worker.model_dump(by_alias=True)
        decode_args = tr.test.test_definition.cmd_args.dynamo.decode_worker.model_dump(by_alias=True)

        for constraint in self.constraints:
            resolved = constraint.lower()
            resolved = resolved.replace('%dynamo%', "dynamo_args")
            resolved = resolved.replace('%prefill%', "prefill_args")
            resolved = resolved.replace('%decode%', "decode_args")
            resolved = resolved.replace('%gpus_per_node%', str(self.cmd_args.gpus_per_node))

            if eval(resolved) == False:
                logging.info(f"constraint_check failed for: {resolved}")
                return False

            logging.info(f"constraint_check passed for: {resolved}")
        return True
