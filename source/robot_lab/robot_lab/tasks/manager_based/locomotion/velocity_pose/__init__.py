# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with velocity and pose tracking commands.

These environments extend velocity tracking with height and orientation control.
"""

##
# Register Gym environments for all robots
##

# Import all robot-specific registrations
from .config.quadruped.unitree_go2 import *  # noqa: F401, F403
from .config.quadruped.unitree_go2_x5 import *  # noqa: F401, F403

# Import environment class
from .velocity_pose_env import VelocityPoseEnv

__all__ = ["VelocityPoseEnv"]
