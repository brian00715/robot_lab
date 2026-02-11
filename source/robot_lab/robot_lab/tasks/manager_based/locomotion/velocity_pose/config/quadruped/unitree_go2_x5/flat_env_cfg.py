# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat terrain configuration for Unitree GO2 + ARX5."""

from isaaclab.utils import configclass
from .rough_env_cfg import UnitreeGo2X5VelocityPoseRoughEnvCfg


@configclass
class UnitreeGo2X5VelocityPoseFlatEnvCfg(UnitreeGo2X5VelocityPoseRoughEnvCfg):
    """Configuration for flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()

                
        # -------------------- Terrain Configuration --------------------
        # Change terrain to flat plane (critical for flat terrain!)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # Flat terrain - no height scanner needed
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        
        # Disable height scanner observations
        if hasattr(self.observations.policy, 'height_scan'):
            self.observations.policy.height_scan = None
        
        # Disable height tracking reward (requires height scanner)
        if hasattr(self.rewards, 'track_height_exp'):
            self.rewards.track_height_exp = None
        
        # Fix base_height_l2 reward (body_names="" causes error)
        if hasattr(self.rewards, 'base_height_l2'):
            self.rewards.base_height_l2.params["sensor_cfg"] = None
            # Set body_names to base_link_name to avoid empty string error
            self.rewards.base_height_l2.params["asset_cfg"].body_names = self.base_link_name
            # Set weight to 0 since we don't need height reward on flat terrain
            self.rewards.base_height_l2.weight = 0.0
        
        # Disable terrain curriculum
        if hasattr(self, 'curriculum') and hasattr(self.curriculum, 'terrain_levels'):
            self.curriculum.terrain_levels = None
