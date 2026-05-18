# SPDX-License-Identifier: Apache-2.0
"""Benchmark Go2 Walk-These-Ways policies with fixed play commands.

The script runs a deterministic suite of command segments and records quantitative
tracking/gait metrics. It is intended for comparing training experiments that
look similar in TensorBoard but differ in play quality.

Example:
    python scripts/reinforcement_learning/rsl_rl/play_wtw_benchmark.py \
        --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
        --num_envs 8 --headless \
        --checkpoint logs/rsl_rl/go2_walk_these_ways/<run>/model_20000.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Benchmark Go2 WalkTheseWays play behavior with fixed commands.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--suite",
    type=str,
    default="standard",
    choices=["standard", "gaits", "height", "height_full", "posture", "trot_params"],
)
parser.add_argument("--cases_json", type=str, default=None, help="Optional JSON file overriding the built-in suite.")
parser.add_argument("--duration_s", type=float, default=6.0, help="Measured duration per case.")
parser.add_argument("--warmup_s", type=float, default=2.0, help="Unmeasured settling duration per case.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Foot contact force threshold in N.")
parser.add_argument(
    "--min_contact_state_s",
    type=float,
    default=0.06,
    help="Minimum contact-state duration used to debounce contact frequency metrics.",
)
parser.add_argument("--output_dir", type=str, default=None, help="Directory for summary.csv/json.")
parser.add_argument("--no_reset_between_cases", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_yaw, quat_conjugate
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

FOOT_NAMES = ("FL", "FR", "RL", "RR")
GAIT_PRESETS = {
    # command columns [phase, offset, bound], matching _resample_commands() after binary phases.
    "pronk": (0.0, 0.0, 0.0),
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.5, 0.0),
    "bound": (0.0, 0.0, 0.5),
}


@dataclass
class BenchmarkCase:
    name: str
    gait: str = "trot"
    vx: float = 0.4
    vy: float = 0.0
    yaw_rate: float = 0.0
    height: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    freq: float = 3.0
    duration: float = 0.5
    swing_height: float = 0.08
    stance_width: float = 0.25
    stance_length: float = 0.40
    warmup_s: float | None = None
    duration_s: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkCase":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: value for key, value in data.items() if key in known}
        extras = {key: value for key, value in data.items() if key not in known}
        kwargs["extras"] = extras
        return cls(**kwargs)


class CaseAccumulator:
    def __init__(
        self,
        name: str,
        gait: str,
        command: torch.Tensor,
        num_envs: int,
        device: torch.device,
        base_height_target: float,
        debounce_steps: int,
    ):
        self.name = name
        self.gait = gait
        self.command = command.detach().clone()
        self.num_envs = num_envs
        self.device = device
        self.base_height_target = base_height_target
        self.debounce_steps = max(1, debounce_steps)
        self.samples = 0
        self.done_count = 0
        self.prev_contact: torch.Tensor | None = None
        self.debounced_contact: torch.Tensor | None = None
        self.pending_contact: torch.Tensor | None = None
        self.pending_counts = torch.zeros(num_envs, 4, dtype=torch.long, device=device)
        self.rising_edges = torch.zeros(4, device=device)

        self.sum_values: dict[str, torch.Tensor] = {}
        self.sqerr_values: dict[str, torch.Tensor] = {}
        self.contact_duty = torch.zeros(4, device=device)
        self.desired_duty = torch.zeros(4, device=device)
        self.contact_match = torch.zeros(4, device=device)
        self.contact_prob_match = torch.zeros(4, device=device)
        self.contact_phase_sin = torch.zeros(4, device=device)
        self.contact_phase_cos = torch.zeros(4, device=device)
        self.contact_phase_count = torch.zeros(4, device=device)
        self.swing_height_sum = torch.tensor(0.0, device=device)
        self.swing_height_sqerr = torch.tensor(0.0, device=device)
        self.swing_height_samples = torch.tensor(0.0, device=device)
        self.foot_height_profile_sqerr = torch.tensor(0.0, device=device)
        self.foot_height_profile_samples = torch.tensor(0.0, device=device)
        self.stance_width_sum = torch.tensor(0.0, device=device)
        self.stance_width_sqerr = torch.tensor(0.0, device=device)
        self.stance_length_sum = torch.tensor(0.0, device=device)
        self.stance_length_sqerr = torch.tensor(0.0, device=device)
        self.pair_sums = {
            "fl_fr_same": torch.tensor(0.0, device=device),
            "fl_rl_same": torch.tensor(0.0, device=device),
            "fl_rr_same": torch.tensor(0.0, device=device),
            "fr_rl_same": torch.tensor(0.0, device=device),
            "fr_rr_same": torch.tensor(0.0, device=device),
            "rl_rr_same": torch.tensor(0.0, device=device),
            "fl_fr_opp": torch.tensor(0.0, device=device),
            "fl_rl_opp": torch.tensor(0.0, device=device),
            "fl_rr_opp": torch.tensor(0.0, device=device),
            "fr_rl_opp": torch.tensor(0.0, device=device),
            "fr_rr_opp": torch.tensor(0.0, device=device),
            "rl_rr_opp": torch.tensor(0.0, device=device),
        }

    def add(self, env, dones: torch.Tensor | None, contact_threshold: float):
        cmd = self.command
        n = env.num_envs
        if dones is not None:
            self.done_count += int(dones.sum().item())

        roll, pitch, _ = euler_xyz_from_quat(env.base_quat)
        if hasattr(env, "contact_force_norms"):
            raw_contact = env.contact_force_norms[:, env.feet_indices] > contact_threshold
        else:
            raw_contact = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) > contact_threshold
        actual_contact = self._debounce_contact(raw_contact)
        desired_contact = env.desired_contact_states > 0.5
        desired_prob = torch.clamp(env.desired_contact_states, 0.0, 1.0)
        swing_weight = torch.clamp(1.0 - env.desired_contact_states, 0.0, 1.0)

        if self.prev_contact is not None:
            self.rising_edges += torch.logical_and(~self.prev_contact, actual_contact).sum(dim=0).float()
        self.prev_contact = actual_contact.detach().clone()

        self._add_mean("vx", env.base_lin_vel[:, 0])
        self._add_mean("vy", env.base_lin_vel[:, 1])
        self._add_mean("yaw_rate", env.base_ang_vel[:, 2])
        self._add_mean("height", env.base_pos[:, 2])
        self._add_mean("roll", roll)
        self._add_mean("pitch", pitch)
        self._add_mean("lin_vel_z_abs", torch.abs(env.base_lin_vel[:, 2]))
        self._add_mean("ang_vel_xy_norm", torch.norm(env.base_ang_vel[:, :2], dim=1))
        self._add_mean("action_abs", torch.mean(torch.abs(env.actions), dim=1))
        self._add_mean("dof_vel_abs", torch.mean(torch.abs(env.dof_vel), dim=1))

        self._add_sqerr("vx", env.base_lin_vel[:, 0], cmd[0])
        self._add_sqerr("vy", env.base_lin_vel[:, 1], cmd[1])
        self._add_sqerr("yaw_rate", env.base_ang_vel[:, 2], cmd[2])
        self._add_sqerr("height", env.base_pos[:, 2], cmd[3] + env.cfg.base_height_target)
        # The environment reward/debug convention targets -command for roll/pitch.
        self._add_sqerr("pitch", pitch, -cmd[10])
        self._add_sqerr("roll", roll, -cmd[11])

        foot_height = env.foot_positions[:, :, 2]
        phases = 1 - torch.abs(1.0 - torch.clip((env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        target_foot_height = cmd[9] * phases + 0.02
        self.swing_height_sum += (foot_height * swing_weight).sum()
        self.swing_height_sqerr += (torch.square(foot_height - (cmd[9] + 0.02)) * swing_weight).sum()
        self.swing_height_samples += swing_weight.sum()
        self.foot_height_profile_sqerr += (torch.square(foot_height - target_foot_height) * swing_weight).sum()
        self.foot_height_profile_samples += swing_weight.sum()

        foot_positions_body = torch.zeros_like(env.foot_positions)
        translated_feet = env.foot_positions - env.base_pos.unsqueeze(1)
        for i in range(4):
            foot_positions_body[:, i, :] = quat_apply_yaw(quat_conjugate(env.base_quat), translated_feet[:, i, :])
        stance_width = 2.0 * torch.mean(torch.abs(foot_positions_body[:, :, 1]), dim=1)
        stance_length = 2.0 * torch.mean(torch.abs(foot_positions_body[:, :, 0]), dim=1)
        self.stance_width_sum += stance_width.sum()
        self.stance_width_sqerr += torch.square(stance_width - cmd[12]).sum()
        self.stance_length_sum += stance_length.sum()
        self.stance_length_sqerr += torch.square(stance_length - cmd[13]).sum()

        self.contact_duty += actual_contact.float().sum(dim=0)
        self.desired_duty += desired_contact.float().sum(dim=0)
        self.contact_match += (actual_contact == desired_contact).float().sum(dim=0)
        self.contact_prob_match += (1.0 - torch.abs(actual_contact.float() - desired_prob)).sum(dim=0)
        phase = env.foot_indices * (2.0 * math.pi)
        self.contact_phase_sin += (actual_contact.float() * torch.sin(phase)).sum(dim=0)
        self.contact_phase_cos += (actual_contact.float() * torch.cos(phase)).sum(dim=0)
        self.contact_phase_count += actual_contact.float().sum(dim=0)
        self._add_pair_stats(actual_contact)
        self.samples += n

    def summary(self, measured_duration_s: float) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "case": self.name,
            "gait": self.gait,
            "samples": float(self.samples),
            "done_count": float(self.done_count),
            "done_rate": self.done_count / max(1.0, self.samples),
            "cmd_vx": float(self.command[0].item()),
            "cmd_vy": float(self.command[1].item()),
            "cmd_yaw_rate": float(self.command[2].item()),
            "cmd_height": float(self.command[3].item()),
            "target_height": float((self.command[3] + self.base_height_target).item()),
            "cmd_freq": float(self.command[4].item()),
            "cmd_phase": float(self.command[5].item()),
            "cmd_offset": float(self.command[6].item()),
            "cmd_bound": float(self.command[7].item()),
            "cmd_gait_duration": float(self.command[8].item()),
            "cmd_swing_height": float(self.command[9].item()),
            "cmd_pitch": float(self.command[10].item()),
            "cmd_roll": float(self.command[11].item()),
            "cmd_stance_width": float(self.command[12].item()),
            "cmd_stance_length": float(self.command[13].item()),
        }
        for key, value in self.sum_values.items():
            row[f"{key}_mean"] = float((value / max(1, self.samples)).item())
        for key, value in self.sqerr_values.items():
            row[f"{key}_rmse"] = float(torch.sqrt(value / max(1, self.samples)).item())
        for i, foot in enumerate(FOOT_NAMES):
            contact_duty = self.contact_duty[i] / max(1, self.samples)
            desired_duty = self.desired_duty[i] / max(1, self.samples)
            contact_phase_count = torch.clamp(self.contact_phase_count[i], min=1.0)
            contact_phase_r = (
                torch.sqrt(self.contact_phase_sin[i] ** 2 + self.contact_phase_cos[i] ** 2) / contact_phase_count
            )
            contact_phase = torch.atan2(self.contact_phase_sin[i], self.contact_phase_cos[i]) / (2.0 * math.pi)
            contact_phase = torch.remainder(contact_phase, 1.0)
            row[f"contact_duty_{foot}"] = float(contact_duty.item())
            row[f"desired_duty_{foot}"] = float(desired_duty.item())
            row[f"contact_duty_error_{foot}"] = float(torch.abs(contact_duty - desired_duty).item())
            row[f"contact_match_{foot}"] = float((self.contact_match[i] / max(1, self.samples)).item())
            row[f"contact_prob_match_{foot}"] = float((self.contact_prob_match[i] / max(1, self.samples)).item())
            row[f"contact_freq_{foot}"] = float(
                (self.rising_edges[i] / max(1.0, self.num_envs * measured_duration_s)).item()
            )
            row[f"contact_phase_r_{foot}"] = float(contact_phase_r.item())
            row[f"contact_phase_{foot}"] = float(contact_phase.item())
        for key, value in self.pair_sums.items():
            row[key] = float((value / max(1, self.samples)).item())
        row["contact_match_mean"] = float((self.contact_match.sum() / max(1, self.samples * 4)).item())
        row["contact_prob_match_mean"] = float((self.contact_prob_match.sum() / max(1, self.samples * 4)).item())
        row["contact_duty_error_mean"] = float(
            sum(float(row[f"contact_duty_error_{foot}"]) for foot in FOOT_NAMES) / len(FOOT_NAMES)
        )
        row["contact_freq_mean"] = float(
            sum(float(row[f"contact_freq_{foot}"]) for foot in FOOT_NAMES) / len(FOOT_NAMES)
        )
        cmd_freq = max(1e-6, abs(float(self.command[4].item())))
        row["contact_freq_ratio"] = float(row["contact_freq_mean"] / cmd_freq)
        row["contact_freq_error"] = abs(float(row["contact_freq_ratio"]) - 1.0)
        swing_samples = torch.clamp(self.swing_height_samples, min=1.0)
        profile_samples = torch.clamp(self.foot_height_profile_samples, min=1.0)
        row["swing_height_mean"] = float((self.swing_height_sum / swing_samples).item())
        row["swing_height_peak_rmse"] = float(torch.sqrt(self.swing_height_sqerr / swing_samples).item())
        row["foot_height_profile_rmse"] = float(torch.sqrt(self.foot_height_profile_sqerr / profile_samples).item())
        row["stance_width_mean"] = float((self.stance_width_sum / max(1, self.samples)).item())
        row["stance_width_rmse"] = float(torch.sqrt(self.stance_width_sqerr / max(1, self.samples)).item())
        row["stance_length_mean"] = float((self.stance_length_sum / max(1, self.samples)).item())
        row["stance_length_rmse"] = float(torch.sqrt(self.stance_length_sqerr / max(1, self.samples)).item())
        row["contact_phase_r_mean"] = float(
            sum(float(row[f"contact_phase_r_{foot}"]) for foot in FOOT_NAMES) / len(FOOT_NAMES)
        )
        row["gait_score"] = self._gait_score(row)
        row["gait_score_v2"] = self._gait_score_v2(row)
        row["tracking_score"] = math.exp(
            -(
                float(row.get("vx_rmse", 0.0)) ** 2
                + float(row.get("vy_rmse", 0.0)) ** 2
                + 0.25 * float(row.get("yaw_rate_rmse", 0.0)) ** 2
            )
        )
        row["height_score"] = math.exp(-(float(row.get("height_rmse", 0.0)) ** 2) / 0.01)
        row["overall_score"] = (
            0.45 * float(row["tracking_score"]) + 0.35 * float(row["gait_score_v2"]) + 0.20 * float(row["height_score"])
        )
        return row

    def _debounce_contact(self, raw_contact: torch.Tensor) -> torch.Tensor:
        if self.debounced_contact is None:
            self.debounced_contact = raw_contact.detach().clone()
            self.pending_contact = raw_contact.detach().clone()
            self.pending_counts.zero_()
            return self.debounced_contact
        if self.debounce_steps <= 1:
            self.debounced_contact = raw_contact.detach().clone()
            return self.debounced_contact

        assert self.pending_contact is not None
        differs = raw_contact != self.debounced_contact
        same_pending = raw_contact == self.pending_contact
        self.pending_counts = torch.where(differs & same_pending, self.pending_counts + 1, self.pending_counts)
        self.pending_counts = torch.where(
            differs & ~same_pending, torch.ones_like(self.pending_counts), self.pending_counts
        )
        self.pending_counts = torch.where(differs, self.pending_counts, torch.zeros_like(self.pending_counts))
        self.pending_contact = torch.where(differs, raw_contact, self.debounced_contact)
        accept = differs & (self.pending_counts >= self.debounce_steps)
        self.debounced_contact = torch.where(accept, raw_contact, self.debounced_contact)
        self.pending_counts = torch.where(accept, torch.zeros_like(self.pending_counts), self.pending_counts)
        return self.debounced_contact

    def _add_mean(self, key: str, value: torch.Tensor):
        self.sum_values[key] = self.sum_values.get(key, torch.tensor(0.0, device=self.device)) + value.sum()

    def _add_sqerr(self, key: str, value: torch.Tensor, target: torch.Tensor | float):
        self.sqerr_values[key] = (
            self.sqerr_values.get(key, torch.tensor(0.0, device=self.device)) + torch.square(value - target).sum()
        )

    def _add_pair_stats(self, contact: torch.Tensor):
        pairs = {
            "fl_fr": (0, 1),
            "fl_rl": (0, 2),
            "fl_rr": (0, 3),
            "fr_rl": (1, 2),
            "fr_rr": (1, 3),
            "rl_rr": (2, 3),
        }
        for name, (i, j) in pairs.items():
            same = contact[:, i] == contact[:, j]
            self.pair_sums[f"{name}_same"] += same.float().sum()
            self.pair_sums[f"{name}_opp"] += (~same).float().sum()

    def _gait_score(self, row: dict[str, float | str]) -> float:
        if self.gait == "trot":
            values = [row["fl_rr_same"], row["fr_rl_same"], row["fl_fr_opp"], row["fl_rl_opp"]]
        elif self.gait == "pace":
            values = [row["fl_rl_same"], row["fr_rr_same"], row["fl_fr_opp"], row["fl_rr_opp"]]
        elif self.gait == "bound":
            values = [row["fl_fr_same"], row["rl_rr_same"], row["fl_rl_opp"], row["fr_rr_opp"]]
        elif self.gait == "pronk":
            values = [row["fl_fr_same"], row["fl_rl_same"], row["fl_rr_same"], row["fr_rl_same"], row["fr_rr_same"]]
        else:
            values = [row["contact_match_mean"]]
        return float(sum(float(v) for v in values) / len(values))

    def _gait_score_v2(self, row: dict[str, float | str]) -> float:
        pair_score = self._gait_score(row)
        prob_match = float(row["contact_prob_match_mean"])
        duty_score = max(0.0, 1.0 - 2.0 * float(row["contact_duty_error_mean"]))
        freq_score = math.exp(-(float(row["contact_freq_error"]) ** 2) / 0.25)
        phase_score = float(row["contact_phase_r_mean"])
        return 0.30 * pair_score + 0.30 * prob_match + 0.20 * duty_score + 0.15 * freq_score + 0.05 * phase_score


def _register_go2_wtw_rma_classes():
    import rsl_rl.runners.on_policy_runner as on_policy_runner
    from robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_rma import Go2WTWActorCritic, Go2WTWPPO

    on_policy_runner.Go2WTWActorCritic = Go2WTWActorCritic
    on_policy_runner.Go2WTWPPO = Go2WTWPPO


def _builtin_cases(suite: str) -> list[BenchmarkCase]:
    gait_cases = [
        BenchmarkCase("trot_v04", gait="trot", vx=0.4),
        BenchmarkCase("trot_v08", gait="trot", vx=0.8),
        BenchmarkCase("pace_v04", gait="pace", vx=0.4),
        BenchmarkCase("bound_v04", gait="bound", vx=0.4),
        BenchmarkCase("pronk_v04", gait="pronk", vx=0.4),
        BenchmarkCase("trot_yaw_left", gait="trot", vx=0.3, yaw_rate=0.5),
    ]
    height_cases = [
        BenchmarkCase("trot_height_low", gait="trot", vx=0.4, height=-0.10),
        BenchmarkCase("trot_height_mid", gait="trot", vx=0.4, height=0.0),
        BenchmarkCase("trot_height_high", gait="trot", vx=0.4, height=0.10),
    ]
    height_full_cases = [
        BenchmarkCase("trot_height_full_low", gait="trot", vx=0.4, height=-0.25),
        BenchmarkCase("trot_height_full_mid", gait="trot", vx=0.4, height=0.0),
        BenchmarkCase("trot_height_full_high", gait="trot", vx=0.4, height=0.15),
    ]
    posture_cases = [
        BenchmarkCase("trot_pitch_pos", gait="trot", vx=0.3, pitch=0.25),
        BenchmarkCase("trot_pitch_neg", gait="trot", vx=0.3, pitch=-0.25),
        BenchmarkCase("trot_roll_pos", gait="trot", vx=0.3, roll=0.20),
        BenchmarkCase("trot_roll_neg", gait="trot", vx=0.3, roll=-0.20),
    ]
    trot_param_cases = [
        BenchmarkCase("trot_freq_2p0", gait="trot", vx=0.4, freq=2.0),
        BenchmarkCase("trot_freq_3p0", gait="trot", vx=0.4, freq=3.0),
        BenchmarkCase("trot_freq_4p0", gait="trot", vx=0.4, freq=4.0),
        BenchmarkCase("trot_swing_0p05", gait="trot", vx=0.4, swing_height=0.05),
        BenchmarkCase("trot_swing_0p12", gait="trot", vx=0.4, swing_height=0.12),
        BenchmarkCase("trot_swing_0p20", gait="trot", vx=0.4, swing_height=0.20),
        BenchmarkCase("trot_width_0p16", gait="trot", vx=0.4, stance_width=0.16),
        BenchmarkCase("trot_width_0p28", gait="trot", vx=0.4, stance_width=0.28),
        BenchmarkCase("trot_width_0p40", gait="trot", vx=0.4, stance_width=0.40),
        BenchmarkCase("trot_length_0p36", gait="trot", vx=0.4, stance_length=0.36),
        BenchmarkCase("trot_length_0p45", gait="trot", vx=0.4, stance_length=0.45),
    ]
    if suite == "gaits":
        return gait_cases
    if suite == "height":
        return height_cases
    if suite == "height_full":
        return height_full_cases
    if suite == "posture":
        return posture_cases
    if suite == "trot_params":
        return trot_param_cases
    return gait_cases + height_cases + posture_cases


def _load_cases(path: str | None, suite: str) -> list[BenchmarkCase]:
    if path is None:
        return _builtin_cases(suite)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [BenchmarkCase.from_dict(item) for item in data]


def _case_to_command(case: BenchmarkCase, env_cfg: DirectRLEnvCfg, device: torch.device) -> torch.Tensor:
    cmd = torch.zeros(env_cfg.num_commands, dtype=torch.float32, device=device)
    phase, offset, bound = GAIT_PRESETS[case.gait]
    cmd[0] = case.vx
    cmd[1] = case.vy
    cmd[2] = case.yaw_rate
    cmd[3] = case.height
    cmd[4] = case.freq
    cmd[5] = phase
    cmd[6] = offset
    cmd[7] = bound
    cmd[8] = case.duration
    cmd[9] = case.swing_height
    cmd[10] = case.pitch
    cmd[11] = case.roll
    cmd[12] = case.stance_width
    cmd[13] = case.stance_length
    return cmd


def _apply_command(env, command: torch.Tensor):
    env.commands[:, : command.numel()] = command.unsqueeze(0).expand(env.num_envs, -1)


def _write_outputs(rows: list[dict[str, Any]], output_dir: str, metadata: dict[str, Any]):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary.json")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "case",
        "gait",
        "overall_score",
        "tracking_score",
        "gait_score_v2",
        "gait_score",
        "height_score",
        "done_rate",
        "cmd_vx",
        "vx_mean",
        "vx_rmse",
        "cmd_freq",
        "contact_freq_mean",
        "contact_freq_ratio",
        "contact_freq_error",
        "cmd_height",
        "height_mean",
        "height_rmse",
        "cmd_swing_height",
        "swing_height_mean",
        "swing_height_peak_rmse",
        "foot_height_profile_rmse",
        "cmd_stance_width",
        "stance_width_mean",
        "stance_width_rmse",
        "cmd_stance_length",
        "stance_length_mean",
        "stance_length_rmse",
        "cmd_pitch",
        "pitch_mean",
        "pitch_rmse",
        "contact_match_mean",
        "contact_prob_match_mean",
        "contact_duty_error_mean",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "cases": rows}, f, indent=2)
    print(f"[INFO] Wrote benchmark CSV: {csv_path}")
    print(f"[INFO] Wrote benchmark JSON: {json_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.resampling_time = 1e9
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _register_go2_wtw_rma_classes()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cases = _load_cases(args_cli.cases_json, args_cli.suite)
    rows: list[dict[str, Any]] = []
    obs = env.get_observations()
    print(f"[INFO] Starting benchmark with {len(cases)} cases, {env.unwrapped.num_envs} envs each.")
    st = time.time()
    for case in cases:
        if not args_cli.no_reset_between_cases:
            obs, _ = env.reset()
        if hasattr(env.unwrapped, "gait_indices"):
            env.unwrapped.gait_indices.zero_()
        command = _case_to_command(case, env_cfg, env.unwrapped.device)
        _apply_command(env.unwrapped, command)

        warmup_s = args_cli.warmup_s if case.warmup_s is None else case.warmup_s
        duration_s = args_cli.duration_s if case.duration_s is None else case.duration_s
        warmup_steps = max(0, round(warmup_s / env.unwrapped.step_dt))
        measure_steps = max(1, round(duration_s / env.unwrapped.step_dt))
        debounce_steps = max(1, round(args_cli.min_contact_state_s / env.unwrapped.step_dt))
        accumulator = CaseAccumulator(
            case.name,
            case.gait,
            command,
            env.unwrapped.num_envs,
            env.unwrapped.device,
            env.unwrapped.cfg.base_height_target,
            debounce_steps,
        )
        print(
            f"[CASE] {case.name}: gait={case.gait} vx={case.vx:.2f} yaw={case.yaw_rate:.2f} "
            f"h={case.height:.2f} pitch={case.pitch:.2f} roll={case.roll:.2f}"
        )

        for step in range(warmup_steps + measure_steps):
            _apply_command(env.unwrapped, command)
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            _apply_command(env.unwrapped, command)
            if step >= warmup_steps:
                accumulator.add(env.unwrapped, dones, args_cli.contact_threshold)
        row = accumulator.summary(measure_steps * env.unwrapped.step_dt)
        rows.append(row)
        print(
            f"[RESULT] {case.name}: overall={row['overall_score']:.3f} track={row['tracking_score']:.3f} "
            f"gait_v2={row['gait_score_v2']:.3f} gait_old={row['gait_score']:.3f} "
            f"freq_ratio={row['contact_freq_ratio']:.2f} duty_err={row['contact_duty_error_mean']:.3f} "
            f"swing_rmse={row['foot_height_profile_rmse']:.3f} width_rmse={row['stance_width_rmse']:.3f} "
            f"height={row['height_score']:.3f} done_rate={row['done_rate']:.4f}"
        )
    print(f"[INFO] Completed benchmark in {time.time() - st:.2f} seconds.")

    if args_cli.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(log_dir, "benchmark_play", stamp)
    else:
        output_dir = os.path.abspath(args_cli.output_dir)
    metadata = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "suite": args_cli.suite,
        "duration_s": args_cli.duration_s,
        "warmup_s": args_cli.warmup_s,
        "contact_threshold": args_cli.contact_threshold,
        "min_contact_state_s": args_cli.min_contact_state_s,
        "foot_order": FOOT_NAMES,
        "gait_presets": GAIT_PRESETS,
    }
    _write_outputs(rows, output_dir, metadata)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
