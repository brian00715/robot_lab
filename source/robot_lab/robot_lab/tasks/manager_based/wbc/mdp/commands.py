# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Command terms for the visual_wholebody (VWBC) port.

Two command generators are provided to match the IsaacGym `manip_loco` MDP:

1. ``VWBCVelocityCommand`` — 3D base-velocity command ``[vx, vy=0, wz]`` in the
   robot's yaw-aligned (Point B) frame. Resampled every 3 s. ``vy`` is forced
   to zero. Below ``lin_vel_x_clip`` / ``ang_vel_yaw_clip`` thresholds the whole
   command is zeroed (matches ``b1z1_config.commands.lin_vel_x_clip / ang_vel_yaw_clip``).
   No heading control. Includes the VWBC global-step curriculum: before
   ``positive_only_until_steps`` global env steps only positive ``vx`` is
   sampled.

2. ``EEGoalSphereCommand`` — End-effector goal in the base-yaw frame, sampled
   in spherical coordinates with linear interpolation between consecutive
   targets. Mirrors ``manip_loco._resample_ee_goal`` / ``_update_curr_ee_goal``.
   Exposes ``curr_ee_goal_cart_world``, ``ee_goal_orn_quat``,
   ``ee_goal_local_cart``, ``curr_ee_goal_sphere`` for the action term, the
   observation terms and the ``tracking_ee_world`` reward.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Helpers (sphere <-> cart)
# ---------------------------------------------------------------------------


def sphere2cart(sphere: torch.Tensor) -> torch.Tensor:
    """Spherical (l, pitch, yaw) -> Cartesian (x, y, z) — VWBC convention.

    Matches ``isaacgym.torch_utils.sphere2cart``: x = l*cos(p)*cos(y),
    y = l*cos(p)*sin(y), z = l*sin(p).
    """
    l = sphere[..., 0]
    p = sphere[..., 1]
    y = sphere[..., 2]
    x_c = l * torch.cos(p) * torch.cos(y)
    y_c = l * torch.cos(p) * torch.sin(y)
    z_c = l * torch.sin(p)
    return torch.stack([x_c, y_c, z_c], dim=-1)


def cart2sphere(cart: torch.Tensor) -> torch.Tensor:
    x = cart[..., 0]
    y = cart[..., 1]
    z = cart[..., 2]
    l = torch.sqrt(x * x + y * y + z * z + 1e-12)
    p = torch.asin(torch.clamp(z / l, -1.0, 1.0))
    yw = torch.atan2(y, x)
    return torch.stack([l, p, yw], dim=-1)


# ---------------------------------------------------------------------------
# 1. VWBC velocity command
# ---------------------------------------------------------------------------


class VWBCVelocityCommand(CommandTerm):
    """3D base-velocity command following ``manip_loco._resample_commands``.

    Command is in the base-yaw (Point-B) frame: ``[vx, vy=0, wz]``. Below
    ``lin_vel_x_clip`` / ``ang_vel_yaw_clip`` the command is zeroed (matches
    ``b1z1_config``). Implements the VWBC curriculum: while
    ``env.common_step_counter < positive_only_until_steps`` only positive
    ``vx`` is sampled.
    """

    cfg: VWBCVelocityCommandCfg

    def __init__(self, cfg: VWBCVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        # global-step curriculum: positive vx only at the start of training
        global_steps = int(getattr(self._env, "common_step_counter", 0))
        lin_lo, lin_hi = self.cfg.ranges.lin_vel_x
        if global_steps < self.cfg.positive_only_until_steps:
            lin_lo = max(lin_lo, 0.0)

        self.vel_command_b[env_ids, 0] = r.uniform_(lin_lo, lin_hi)
        self.vel_command_b[env_ids, 1] = 0.0  # VWBC forces vy=0
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # threshold clipping: zero entire command when both axes are below thresholds
        # (matches manip_loco._resample_commands lines 950-951)
        x_clip = self.cfg.lin_vel_x_clip
        yaw_clip = self.cfg.ang_vel_yaw_clip
        keep = (torch.abs(self.vel_command_b[env_ids, 0]) > x_clip) | (
            torch.abs(self.vel_command_b[env_ids, 2]) > yaw_clip
        )
        self.vel_command_b[env_ids] *= keep.unsqueeze(1).float()

    def _update_command(self):
        # No heading control / no standing-env override in VWBC.
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        return


@configclass
class VWBCVelocityCommandCfg(CommandTermCfg):
    """Configuration for the VWBC base-velocity command."""

    class_type: type = VWBCVelocityCommand

    asset_name: str = "robot"
    """Name of the articulation in the scene."""

    resampling_time_range: tuple[float, float] = (3.0, 3.0)
    """Resampling time in seconds (VWBC uses 3 s)."""

    lin_vel_x_clip: float = 0.2
    """Below ``|vx| < lin_vel_x_clip`` AND ``|wz| < ang_vel_yaw_clip`` the command is zeroed."""

    ang_vel_yaw_clip: float = 0.5

    positive_only_until_steps: int = 5000 * 24
    """During the first ``positive_only_until_steps`` global env steps only ``vx >= 0`` is sampled."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-0.8, 0.8)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    ranges: Ranges = Ranges()


# ---------------------------------------------------------------------------
# 2. EE goal sphere command
# ---------------------------------------------------------------------------


class EEGoalSphereCommand(CommandTerm):
    """Environment-driven end-effector goal in spherical coordinates.

    Mirrors ``manip_loco._resample_ee_goal`` and ``_update_curr_ee_goal``:

    * On reset (``is_init=True``) the goal is set to the configured
      ``init_pos_start`` / ``init_pos_end`` pair, with the EE delta orn cleared.
    * Otherwise a new target is sampled in spherical coordinates with up to 10
      collision-rejection retries against the configured AABB and underground
      limit. The orientation delta is sampled uniformly in
      ``[delta_orn_*]``.
    * Each step ``curr_ee_goal_sphere`` linearly interpolates from
      ``ee_start_sphere`` towards ``ee_goal_sphere`` over ``traj_time``,
      then holds for ``hold_time``. After ``traj_time + hold_time`` a new goal
      is resampled.

    The ``command`` tensor returned by this term is the EE goal expressed in
    the **yaw-aligned base frame** (Point-B Cartesian), shape ``(N, 3)``. This
    matches the obs term ``ee_goal_local_cart`` from ``compute_observations``.
    Other consumers (the action term, the ``tracking_ee_world`` reward, the
    EE goal orientation obs) read additional buffers off this term directly:

    * ``self.curr_ee_goal_cart_world``  ``(N, 3)`` — world-frame goal position
    * ``self.ee_goal_orn_quat``         ``(N, 4)`` — world-frame goal orn (wxyz)
    * ``self.curr_ee_goal_sphere``      ``(N, 3)``
    * ``self.ee_goal_orn_delta_rpy``    ``(N, 3)``
    """

    cfg: EEGoalSphereCommandCfg

    def __init__(self, cfg: EEGoalSphereCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        N = self.num_envs
        d = self.device
        dt = float(self._env.step_dt)

        # constants
        self._dt = dt
        self._arm_induced_pitch = float(cfg.arm_induced_pitch)

        self._init_start = torch.tensor(cfg.ranges.init_pos_start, device=d, dtype=torch.float32)
        self._init_end = torch.tensor(cfg.ranges.init_pos_end, device=d, dtype=torch.float32)

        self._collision_lower = torch.tensor(cfg.collision_lower_limits, device=d, dtype=torch.float32)
        self._collision_upper = torch.tensor(cfg.collision_upper_limits, device=d, dtype=torch.float32)
        self._underground_limit = float(cfg.underground_limit)
        n_check = int(cfg.num_collision_check_samples)
        self._n_check = n_check
        self._collision_check_t = torch.linspace(0.0, 1.0, n_check, device=d).view(1, 1, n_check)

        self._sphere_center_offset = torch.tensor(
            [cfg.sphere_center_x_offset, cfg.sphere_center_y_offset, cfg.sphere_center_z_invariant_offset],
            device=d,
            dtype=torch.float32,
        ).repeat(N, 1)

        # buffers (timesteps, sampled in __init__ once like VWBC does)
        traj_lo, traj_hi = cfg.traj_time
        hold_lo, hold_hi = cfg.hold_time
        traj_steps = (torch.rand(N, device=d) * (traj_hi - traj_lo) + traj_lo) / dt
        hold_steps = (torch.rand(N, device=d) * (hold_hi - hold_lo) + hold_lo) / dt
        self.traj_timesteps = traj_steps
        self.traj_total_timesteps = traj_steps + hold_steps
        self.goal_timer = torch.zeros(N, device=d)

        self.ee_start_sphere = self._init_start.unsqueeze(0).repeat(N, 1).clone()
        self.ee_goal_sphere = self._init_end.unsqueeze(0).repeat(N, 1).clone()
        self.curr_ee_goal_sphere = torch.zeros(N, 3, device=d)
        self.curr_ee_goal_cart = torch.zeros(N, 3, device=d)
        self.curr_ee_goal_cart_world = torch.zeros(N, 3, device=d)

        self.ee_goal_orn_delta_rpy = torch.zeros(N, 3, device=d)
        self.ee_goal_orn_quat = torch.zeros(N, 4, device=d)
        self.ee_goal_orn_quat[:, 0] = 1.0  # identity

        # 3D command buffer returned to the obs/manager layer (yaw-frame Cart)
        self._cmd_buf = torch.zeros(N, 3, device=d)

        # metrics (kept for parity with manager-based introspection)
        self.metrics["ee_goal_cart_world_norm"] = torch.zeros(N, device=d)

    # ----- public CommandTerm API ----------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return self._cmd_buf

    def _update_metrics(self):
        self.metrics["ee_goal_cart_world_norm"][:] = torch.norm(self.curr_ee_goal_cart_world, dim=-1)

    # ----- core mechanics -------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        """Manager calls this on reset and at the resampling interval.

        VWBC samples a fresh EE goal on every reset (``is_init=True`` path),
        and continually resamples through ``_update_curr_ee_goal`` whenever
        ``goal_timer > traj_total_timesteps`` — that path is invoked from
        ``_update_command``. Because the manager's resampling interval
        machinery would interfere with VWBC's per-env trajectory timer, we
        treat every ``_resample_command`` call as the ``is_init=True`` reset
        case (it is only invoked by the manager on env reset thanks to the
        very long ``resampling_time_range``).
        """
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # On reset: clear delta orn, snap to init start / init end pair.
        self.ee_goal_orn_delta_rpy[env_ids_t] = 0.0
        self.ee_start_sphere[env_ids_t] = self._init_start
        self.ee_goal_sphere[env_ids_t] = self._init_end
        self.goal_timer[env_ids_t] = 0.0
        self.curr_ee_goal_sphere[env_ids_t] = self.ee_start_sphere[env_ids_t]
        self.curr_ee_goal_cart[env_ids_t] = sphere2cart(self.curr_ee_goal_sphere[env_ids_t])

        # also resample traj/hold durations for the new episode
        traj_lo, traj_hi = self.cfg.traj_time
        hold_lo, hold_hi = self.cfg.hold_time
        n = len(env_ids_t)
        traj_steps = (torch.rand(n, device=self.device) * (traj_hi - traj_lo) + traj_lo) / self._dt
        hold_steps = (torch.rand(n, device=self.device) * (hold_hi - hold_lo) + hold_lo) / self._dt
        self.traj_timesteps[env_ids_t] = traj_steps
        self.traj_total_timesteps[env_ids_t] = traj_steps + hold_steps

    def _update_command(self):
        """Per-step trajectory update (VWBC ``_update_curr_ee_goal``)."""
        # interpolate t in [0, 1] from goal_timer / traj_timesteps
        t = torch.clamp(self.goal_timer / self.traj_timesteps, 0.0, 1.0).unsqueeze(-1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t)
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)

        # world-frame goal position via base-yaw frame
        base_quat_w = self.robot.data.root_quat_w  # (N, 4) wxyz
        base_yaw_quat = self._yaw_quat_from_quat(base_quat_w)
        ee_goal_cart_yaw_global = math_utils.quat_apply(base_yaw_quat, self.curr_ee_goal_cart)
        spherical_center = self._get_ee_goal_spherical_center(base_yaw_quat)
        self.curr_ee_goal_cart_world[:] = spherical_center + ee_goal_cart_yaw_global

        # default eef orientation (matches manip_loco._update_curr_ee_goal lines 1270-1272)
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self._arm_induced_pitch
        self.ee_goal_orn_quat[:] = math_utils.quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + math.pi / 2.0,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            default_yaw + self.ee_goal_orn_delta_rpy[:, 2],
        )

        # advance goal timer & resample for envs whose hold expired
        self.goal_timer += 1.0
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if resample_id.numel() > 0:
            self._inflight_resample(resample_id)

        # also publish the obs-friendly EE goal (yaw-frame Cartesian)
        self._cmd_buf[:] = self.curr_ee_goal_cart

    def _inflight_resample(self, env_ids: torch.Tensor):
        """In-trajectory resample: pick a new orn delta + sphere goal.

        Mirrors the ``is_init=False`` branch in ``manip_loco._resample_ee_goal``:
        delta orn is uniformly resampled, then up to 10 spherical-goal candidates
        are drawn until none collide with the configured AABB / underground.
        """
        rng = self.cfg.ranges
        n = env_ids.numel()
        d = self.device

        delta = torch.empty(n, 3, device=d).uniform_(-1.0, 1.0)
        delta[:, 0] = delta[:, 0] * (rng.delta_orn_r[1] - rng.delta_orn_r[0]) / 2.0 + (
            rng.delta_orn_r[0] + rng.delta_orn_r[1]
        ) / 2.0
        delta[:, 1] = delta[:, 1] * (rng.delta_orn_p[1] - rng.delta_orn_p[0]) / 2.0 + (
            rng.delta_orn_p[0] + rng.delta_orn_p[1]
        ) / 2.0
        delta[:, 2] = delta[:, 2] * (rng.delta_orn_y[1] - rng.delta_orn_y[0]) / 2.0 + (
            rng.delta_orn_y[0] + rng.delta_orn_y[1]
        ) / 2.0
        self.ee_goal_orn_delta_rpy[env_ids] = delta

        # start = current goal (continuity)
        self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()

        active = env_ids.clone()
        for _ in range(10):
            if active.numel() == 0:
                break
            m = active.numel()
            new_sphere = torch.empty(m, 3, device=d)
            new_sphere[:, 0] = torch.empty(m, device=d).uniform_(*rng.pos_l)
            new_sphere[:, 1] = torch.empty(m, device=d).uniform_(*rng.pos_p)
            new_sphere[:, 2] = torch.empty(m, device=d).uniform_(*rng.pos_y)
            self.ee_goal_sphere[active] = new_sphere

            collide = self._collision_check(active)
            active = active[collide]

        # resample traj/hold timer
        traj_lo, traj_hi = self.cfg.traj_time
        hold_lo, hold_hi = self.cfg.hold_time
        traj_steps = (torch.rand(n, device=d) * (traj_hi - traj_lo) + traj_lo) / self._dt
        hold_steps = (torch.rand(n, device=d) * (hold_hi - hold_lo) + hold_lo) / self._dt
        self.traj_timesteps[env_ids] = traj_steps
        self.traj_total_timesteps[env_ids] = traj_steps + hold_steps
        self.goal_timer[env_ids] = 0.0

    def _collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return mask of envs whose current start->goal segment hits the bbox.

        Same as ``manip_loco._collision_check``: linearly interpolate from
        start to goal in spherical coordinates, convert to cart, and reject
        if any sample lands inside the configured AABB or below the
        underground limit.
        """
        s = self.ee_start_sphere[env_ids]  # (n, 3)
        g = self.ee_goal_sphere[env_ids]   # (n, 3)
        # broadcast: (n, 3, 1) lerp (1, 1, T) -> (n, 3, T) -> (T, n, 3)
        ee_target_sphere = torch.lerp(s.unsqueeze(-1), g.unsqueeze(-1), self._collision_check_t)
        ee_target_sphere = ee_target_sphere.permute(2, 0, 1)
        flat = ee_target_sphere.reshape(-1, 3)
        cart = sphere2cart(flat).reshape(self._n_check, -1, 3)
        in_bbox = torch.logical_and(
            torch.all(cart < self._collision_upper, dim=-1),
            torch.all(cart > self._collision_lower, dim=-1),
        )
        underground = cart[..., 2] < self._underground_limit
        return torch.any(in_bbox | underground, dim=0)

    # ----- frame helpers --------------------------------------------------

    def _yaw_quat_from_quat(self, quat_w: torch.Tensor) -> torch.Tensor:
        """Pure yaw quaternion (wxyz) extracted from a base quaternion."""
        w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        half = yaw / 2.0
        out = torch.zeros_like(quat_w)
        out[:, 0] = torch.cos(half)
        out[:, 3] = torch.sin(half)
        return out

    def _get_ee_goal_spherical_center(self, base_yaw_quat: torch.Tensor) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w
        center = torch.cat([root_pos[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        return center + math_utils.quat_apply(base_yaw_quat, self._sphere_center_offset)

    def _set_debug_vis_impl(self, debug_vis: bool):
        return


@configclass
class EEGoalSphereCommandCfg(CommandTermCfg):
    """Configuration for ``EEGoalSphereCommand``."""

    class_type: type = EEGoalSphereCommand

    asset_name: str = "robot"

    # very long resampling_time so the manager only "resets" us on env reset.
    # In-flight goal resampling is driven by the per-env traj+hold timer.
    resampling_time_range: tuple[float, float] = (1.0e8, 1.0e8)

    traj_time: tuple[float, float] = (1.0, 3.0)
    hold_time: tuple[float, float] = (0.5, 2.0)

    # AABB used for collision rejection of EE goal trajectories (in yaw-frame)
    collision_lower_limits: tuple[float, float, float] = (-0.8, -0.2, -0.7)
    collision_upper_limits: tuple[float, float, float] = (0.1, 0.2, -0.05)
    underground_limit: float = -0.7
    num_collision_check_samples: int = 10

    # spherical center offset relative to the dog base (yaw-frame)
    sphere_center_x_offset: float = 0.3
    sphere_center_y_offset: float = 0.0
    sphere_center_z_invariant_offset: float = 0.7

    # Default eef pitch added to -pos_p to get the natural arm orientation
    arm_induced_pitch: float = 0.38

    @configclass
    class Ranges:
        init_pos_start: tuple[float, float, float] = (0.5, math.pi / 8.0, 0.0)
        init_pos_end: tuple[float, float, float] = (0.7, 0.0, 0.0)
        pos_l: tuple[float, float] = (0.4, 0.95)
        pos_p: tuple[float, float] = (-math.pi / 2.5, math.pi / 3.0)
        pos_y: tuple[float, float] = (-1.2, 1.2)

        delta_orn_r: tuple[float, float] = (-0.5, 0.5)
        delta_orn_p: tuple[float, float] = (-0.5, 0.5)
        delta_orn_y: tuple[float, float] = (-0.5, 0.5)

    ranges: Ranges = Ranges()
