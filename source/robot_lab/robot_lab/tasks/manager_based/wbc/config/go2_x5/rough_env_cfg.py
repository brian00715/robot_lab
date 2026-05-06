# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unitree GO2 + ARX5 — VWBC port (rough terrain)."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.wbc.wbc_env_cfg import WbcRoughEnvCfg
from robot_lab.assets.go2_x5 import GO2_X5_CFG


DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@configclass
class ArxX5WbcRoughEnvCfg(WbcRoughEnvCfg):
    """GO2 + ARX5 rough-terrain env."""

    base_link_name: str = "base"
    foot_link_name: str = ".*_foot"
    ee_body_name: str = "ee"

    def __post_init__(self):
        super().__post_init__()

        # ---- scene ----
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ---- actions: pin joint patterns + per-joint scales (b1z1: 0.4 hip, 0.45 thigh/calf) ----
        self.actions.joint_pos.dog_joint_names = DOG_JOINT_NAMES
        self.actions.joint_pos.arm_joint_names = ARM_JOINT_NAMES
        self.actions.joint_pos.ee_body_name = self.ee_body_name
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.4,
            ".*_thigh_joint": 0.45,
            ".*_calf_joint": 0.45,
        }

        # ---- events: pin asset_cfg body names ----
        self.events.randomize_friction.params["asset_cfg"] = SceneEntityCfg("robot")
        self.events.randomize_base_mass_and_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=self.base_link_name
        )
        self.events.randomize_base_mass_and_com.params["gripper_body_name"] = self.ee_body_name

        # ---- rewards: pin contact body names ----
        self.rewards.collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[f"^(?!.*{self.foot_link_name}).*_(thigh|calf)$", "base"],
        )
        # b1z1 penalises contacts on thigh/trunk/calf — on GO2+X5 the trunk
        # body is named ``base`` (no separate ``trunk`` link).
