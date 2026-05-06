# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Velocity and Pose Tracking Environment for Quadruped Locomotion

This environment extends velocity tracking with height and orientation control.
For GO2+ARX5, it integrates the arm trajectory controller to handle composite actions.
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg


class WbcEnv(ManagerBasedRLEnv):
    """Environment for quadruped locomotion with velocity and pose tracking.

    For robots with manipulators (e.g., GO2+ARX5), the composite action space 
    is handled by DogArmCompositeAction in the ActionManager:
    - Policy outputs actions for dog joints only (12D for GO2)
    - Arm trajectory controller generates arm actions (6D for ARX5)
    - DogArmCompositeAction combines them and applies to robot (18D total)
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the velocity pose environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for the environment.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # print("[WbcEnv] Initialized successfully")  # (DISABLED for performance)
        # print("[WbcEnv] NOTE: Arm motion is handled by DogArmCompositeAction in ActionManager")  # (DISABLED for performance)

