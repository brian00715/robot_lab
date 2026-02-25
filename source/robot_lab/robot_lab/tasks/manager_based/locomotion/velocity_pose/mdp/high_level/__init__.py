# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""High-Level IK-based components."""

# Import IK-based components
from .composite_actions_ik import DogArmIKCompositeAction  # noqa: F401
from .arm_controller_ik import (  # noqa: F401
    ARX5IKController,
    create_ik_arm_controller,
    extract_yaw_quat,
)
from .ik_rewards import *  # noqa: F401, F403

# Keep arm_ik_controller for backward compatibility if needed
try:
    from .arm_ik_controller import *  # noqa: F401, F403
except ImportError:
    pass
