# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Curriculum terms for the VWBC port.

VWBC's "curriculum" (``manip_loco._resample_commands`` and ``manip_loco.step``)
is implemented as two global-step gates that live where they are read:

* "Sample positive ``vx`` only before 5000*24 global steps" lives inside
  :class:`VWBCVelocityCommand._resample_command`.
* "Switch action delay buffer index from -1 to -2 at 10000*24 global steps"
  lives inside :class:`VisualWholeBodyAction.process_actions`.

There are no manager-driven curriculum terms in VWBC, so this module
intentionally ships no terms — keep it for the ``mdp/`` import boilerplate
and as a place to stash future curricula.
"""

from __future__ import annotations

__all__ = []
