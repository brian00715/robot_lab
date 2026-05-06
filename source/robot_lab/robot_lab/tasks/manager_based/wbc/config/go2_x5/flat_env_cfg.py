# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat-terrain variant of the GO2 + ARX5 VWBC env."""

from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import ArxX5WbcRoughEnvCfg


@configclass
class ArxX5WbcFlatEnvCfg(ArxX5WbcRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None


@configclass
class ArxX5WbcFlatPlayEnvCfg(ArxX5WbcFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Disable training-time perturbations during play
        self.events.push_robot = None
