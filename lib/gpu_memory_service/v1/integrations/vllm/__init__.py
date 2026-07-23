# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM workspace-growth routing for experimental GMS V1."""

from .patches import install_vllm_integration

__all__ = ["install_vllm_integration"]
