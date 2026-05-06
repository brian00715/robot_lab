# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Manager-based whole-body-control tasks."""

from .wbc_env import WbcEnv

# Importing `learning` registers VWBCActorCritic / VWBCPPO into the rsl_rl
# namespaces so OnPolicyRunner can resolve them by name from the runner cfg.
from . import learning  # noqa: F401

from .config.go2_x5 import *  # noqa: F401, F403

__all__ = ["WbcEnv"]
