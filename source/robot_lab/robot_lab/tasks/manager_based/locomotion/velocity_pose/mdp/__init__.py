# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical MDP components for velocity_pose locomotion environments.

This module is organized into three sub-modules:
- low_level: Low-level locomotion control (without IK)
- high_level: High-level IK-based control
- shared: Shared components (actions, commands, observations, etc.)
"""

# Import all velocity mdp functions first
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# Import shared components (actions, commands, observations, etc.)
from .shared import *  # noqa: F401, F403

# Import low-level components
from .low_level import *  # noqa: F401, F403

# Import high-level components
from .high_level import *  # noqa: F401, F403

# Backward compatibility: re-export commonly used classes at top level
from .low_level import (  # noqa: F401
    DogArmCompositeAction,
    ARX5TrajectoryController,
    create_arm_controller,
)

from .high_level import (  # noqa: F401
    DogArmIKCompositeAction,
    ARX5IKController,
    create_ik_arm_controller,
    extract_yaw_quat,
)

