# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Manager-based whole-body locomotion-manipulation environment."""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg


class WbcEnv(ManagerBasedRLEnv):
    """Environment for velocity/pose tracking with visual_wholebody-style arm IK."""

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the velocity pose environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Arm motion is handled by the action term through an EE goal and Jacobian IK.
