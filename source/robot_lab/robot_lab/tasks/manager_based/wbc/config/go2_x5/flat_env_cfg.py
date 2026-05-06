# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat terrain configuration for Unitree GO2 + ARX5."""

from isaaclab.utils import configclass
from .rough_env_cfg import ArxX5WbcRoughEnvCfg


@configclass
class ArxX5WbcFlatEnvCfg(ArxX5WbcRoughEnvCfg):
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
        
        if hasattr(self.rewards, 'track_height_exp') and self.rewards.track_height_exp is not None:
            self.rewards.track_height_exp.params["sensor_cfg"] = None
        
        if hasattr(self.rewards, 'base_height_l2') and self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
            self.rewards.base_height_l2.params["asset_cfg"].body_names = self.base_link_name
            self.rewards.base_height_l2.weight = 0.0
        
        # Disable terrain curriculum
        if hasattr(self, 'curriculum') and hasattr(self.curriculum, 'terrain_levels'):
            self.curriculum.terrain_levels = None


@configclass
class ArxX5WbcFlatPlayEnvCfg(ArxX5WbcFlatEnvCfg):
    """Flat play configuration with visible ARX5 arm motion."""

    inference_stage = 4
    fixed_arm_mode_idx = 0

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False

        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot = None

        self.curriculum.command_levels = None
        self.curriculum.command_curriculum_height_pose = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity_pose.debug_vis = False
        self.commands.base_velocity_pose.ranges.height = (0.18, 0.43)
        self.commands.base_velocity_pose.ranges.roll = (-0.785, 0.785)
        self.commands.base_velocity_pose.ranges.pitch = (-0.436, 0.436)
