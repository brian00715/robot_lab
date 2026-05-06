# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""VWBC learning components.

Importing this package registers :class:`VWBCActorCritic` and :class:`VWBCPPO`
into the namespace used by :class:`OnPolicyRunner._construct_algorithm`. The
runner resolves classes via ``eval(class_name)``, which sees only the globals
of the ``rsl_rl.runners.on_policy_runner`` module — so we inject the classes
*there* (along with the public ``rsl_rl.modules`` / ``rsl_rl.algorithms``
namespaces for completeness).
"""

from __future__ import annotations

import rsl_rl.algorithms
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _opr

from .actor_critic import VWBCActorCritic, StateHistoryEncoder
from .ppo import VWBCPPO

# Inject into the runner's eval() namespace.
_opr.VWBCActorCritic = VWBCActorCritic
_opr.VWBCPPO = VWBCPPO

# Also expose on the public namespaces for code that imports them directly.
rsl_rl.modules.VWBCActorCritic = VWBCActorCritic
rsl_rl.algorithms.VWBCPPO = VWBCPPO

__all__ = ["VWBCActorCritic", "VWBCPPO", "StateHistoryEncoder"]
