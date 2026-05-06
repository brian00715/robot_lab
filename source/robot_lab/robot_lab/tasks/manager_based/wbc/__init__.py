# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Manager-based whole-body-control tasks."""

from .wbc_env import WbcEnv

from .config.go2_x5 import *  # noqa: F401, F403

__all__ = ["WbcEnv"]
