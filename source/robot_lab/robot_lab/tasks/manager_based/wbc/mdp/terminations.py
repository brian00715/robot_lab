# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for the VWBC port.

Mirrors ``manip_loco.check_termination`` (lines 257-277):

    r_term = |roll|  > 0.8
    p_term = |pitch| > 0.8
    z_term = root_z  < 0.1
    timeout
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_roll_pitch_too_large(
    env: ManagerBasedRLEnv,
    roll_threshold: float = 0.8,
    pitch_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when ``|roll| > roll_threshold`` or ``|pitch| > pitch_threshold``."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return (torch.abs(roll) > roll_threshold) | (torch.abs(pitch) > pitch_threshold)


__all__ = ["base_roll_pitch_too_large"]
