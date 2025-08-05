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

from typing import List


def inverse_reward(observation: List[float]) -> float:
    if observation and observation[0] != 0:
        return 1.0 / observation[0]
    return 0.0


def negative_reward(observation: List[float]) -> float:
    if observation:
        return -observation[0]
    return 0.0


def identity_reward(observation: List[float]) -> float:
    if observation:
        return observation[0]
    return 0.0


def custom_reward(observation: List[float]) -> float:
    """Custom reward function for the AI Dynamo."""
    ttft_idx = 0
    itl_idx = 1
    throughput_idx = 2

    # Normalization
    ttft_baseline = 0.3  # seconds
    itl_baseline = 0.02  # seconds
    throughput_baseline = 50.0  # tokens/s

    # Weighting between metrics - equal focus on TTFT and throughput
    ttft_weight = 0.45
    itl_weight = 0.1
    throughput_weight = 0.45

    if len(observation) < 3:
        return -1.0

    ttft = observation[ttft_idx]
    itl = observation[itl_idx]
    throughput = observation[throughput_idx]

    ttft_reward = ttft_baseline / ttft
    itl_reward = itl_baseline / itl

    throughput_reward = throughput / throughput_baseline

    # Weighted combined reward
    reward = ttft_weight * ttft_reward + itl_weight * itl_reward + throughput_weight * throughput_reward

    return reward
