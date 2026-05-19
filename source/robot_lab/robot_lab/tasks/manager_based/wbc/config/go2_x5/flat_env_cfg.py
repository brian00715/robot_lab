# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat-terrain variant of the GO2 + ARX5 VWBC env."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .rough_env_cfg import ArxX5WbcRoughEnvCfg
import robot_lab.tasks.manager_based.wbc.mdp as mdp

_VWBC_REW_NORM = 1.0 / 100.0


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


# ---------------------------------------------------------------------------
# H2: FPG-style IK feasibility reward experiment variants
# ---------------------------------------------------------------------------


@configclass
class ArxX5WbcFlatH2PFGEnvCfg(ArxX5WbcFlatEnvCfg):
    """H2 experiment — adds FPG-style ``ik_feasibility_dls`` reward on top of
    the unmodified baseline.  Baseline task ID can be used as the control run;
    this task ID adds only the geometric prior reward term."""

    def __post_init__(self):
        super().__post_init__()
        # IK feasibility reward (geometric prior, weight starts at 1.0 / 100)
        self.rewards.ik_feasibility = RewTerm(
            func=mdp.ik_feasibility_dls,
            weight=1.0 * _VWBC_REW_NORM,
            params={
                "ee_goal_command_name": "ee_goal",
                "ee_body_name": "ee",
                "arm_joint_pattern": "joint[1-6]",
                "ik_damping": 0.05,
                "sigma_q": 0.35,
                "residual_tol": 0.08,
                "pos_tol": 0.20,
            },
        )


@configclass
class ArxX5WbcFlatH2PFGPlayEnvCfg(ArxX5WbcFlatH2PFGEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.push_robot = None
