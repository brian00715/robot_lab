# SPDX-License-Identifier: Apache-2.0
"""Joystick play script for Go2 Walk-These-Ways Direct RL task.

Uses JoyLink JoystickClient for gamepad input. Supports ZMQ, ROS2, and DDS backends.
Usage:
    python play_wtw_joystick.py --joy_config JoyLink/config/loco_ctrl.yaml

cmd[0] = vx
cmd[1] = vy
cmd[2] = yaw_rate
cmd[3] = height
cmd[4] = gait_freq
cmd[5] = gait_phase
cmd[6] = gait_offset
cmd[7] = gait_bound
cmd[8] = gait_duration
cmd[9] = swing_height
cmd[10] = pitch
cmd[11] = roll
cmd[12] = stance_width
cmd[13] = stance_length
cmd[14] = aux_reward_coef
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play WTW policy with joystick input.")
parser.add_argument("--task", default="RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--joy_config", default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from joystick_client import JoystickClient
from rsl_rl.runners import OnPolicyRunner


class _OnnxActorWrapper(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module | None):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward(self, obs: torch.Tensor):
        return self.actor(self.normalizer(obs))


def _export_policy(policy_nn, normalizer, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    except Exception as exc:
        print(f"[WARN] Failed to export TorchScript policy; continuing play: {exc}")
    try:
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    except Exception as exc:
        actor = getattr(policy_nn, "actor", None)
        adaptation_module = getattr(actor, "adaptation_module", None)
        if adaptation_module is None or not hasattr(adaptation_module[0], "in_features"):
            print(f"[WARN] Failed to export ONNX policy; continuing play: {exc}")
            return
        try:
            wrapper = _OnnxActorWrapper(actor, normalizer).to("cpu").eval()
            obs = torch.zeros(1, adaptation_module[0].in_features)
            torch.onnx.export(
                wrapper,
                obs,
                os.path.join(export_dir, "policy.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
            print(f"[WARN] IsaacLab ONNX exporter does not support {type(actor).__name__}: {exc}")
        except Exception as fallback_exc:
            print(f"[WARN] Failed to export CSE ONNX policy; continuing play: {fallback_exc}")


class WtwJoystick:
    """Joystick controller for the Walk-These-Ways Direct RL task."""

    def __init__(self, env_cfg: DirectRLEnvCfg, device: torch.device, config_path: str):
        self._sim_device = device
        self._client = JoystickClient(config_path)
        self._client.connect()

        self._cmd = np.zeros(15, dtype=np.float32)
        self._cmd_limits = self._build_cmd_limits(env_cfg)
        self._axis_scales = self._build_axis_scales()
        self._set_default_gait_cmd()

        self._stand_when_still = False

        # Button edge-detection state
        self._prev_buttons: dict[str, int] = {}
        # Trigger/dpad threshold-crossing state
        self._prev_dpad_x = 0.0
        self._prev_dpad_y = 0.0
        self._prev_lt = 0.0
        self._prev_rt = 0.0

        # Increment step sizes (mirror keyboard sensitivities)
        self._height_step = 0.02
        self._roll_step = 0.05
        self._freq_step = 0.2
        self._swing_step = 0.02

        self._gait_presets = {
            "pronk": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "trot": np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "pace": np.array([0.0, 0.5, 0.0], dtype=np.float32),
            "bound": np.array([0.0, 0.0, 0.5], dtype=np.float32),
        }
        self._gait_buttons = {
            "a": "pronk",
            "b": "trot",
            "x": "pace",
            "y": "bound",
        }

        self._no_data_count = 0
        self._printed_no_data = False

    def __del__(self):
        self._client.close()

    def reset(self):
        self._cmd.fill(0.0)
        self._set_default_gait_cmd()
        self._print_state()

    def advance(self) -> torch.Tensor:
        # JoystickClient._recv_loop continuously pushes to its internal queue.
        # Drain to get the freshest sample without blocking.
        data = None
        while True:
            msg = self._client.receive(timeout_ms=0)
            if msg is None:
                break
            data = msg

        if data is not None:
            self._no_data_count = 0
            self._printed_no_data = False
            self._process(data)
        else:
            self._no_data_count += 1
            if self._no_data_count > 100 and not self._printed_no_data:
                print("[JOY] No joystick data received. Is the JoyLink server running?")
                self._printed_no_data = True

        cmd = self._cmd

        if self._stand_when_still and np.linalg.norm(cmd[:3]) < 1e-4:
            cmd[4] = 0.0
            cmd[5:8] = 0.0
            cmd[8] = 0.5

        cmd = np.clip(cmd, self._cmd_limits[:, 0], self._cmd_limits[:, 1])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    # ── Processing ──────────────────────────────────────────────────────

    def _process(self, data: dict):
        axes: dict[str, float] = data.get("axes", {})
        buttons: dict[str, int] = data.get("buttons", {})

        self._cmd[0] = axes["left_stick_x"] * self._axis_scales[0]  # vx
        self._cmd[1] = axes["left_stick_y"] * self._axis_scales[1]  # vy
        self._cmd[2] = axes["right_stick_y"] * self._axis_scales[2]  # yaw
        self._cmd[10] = -axes["right_stick_x"] * self._axis_scales[4]  # pitch

        # body height (dpad_x, incremental)
        dpad_x = axes.get("dpad_x", 0.0)
        if abs(dpad_x) > 0.5 and abs(self._prev_dpad_x) <= 0.5:
            self._cmd[3] = np.clip(
                self._cmd[3] + np.sign(dpad_x) * self._height_step,
                self._cmd_limits[3, 0],
                self._cmd_limits[3, 1],
            )
        self._prev_dpad_x = dpad_x

        # gait frequency (dpad_y, incremental)
        dpad_y = axes.get("dpad_y", 0.0)
        if abs(dpad_y) > 0.5 and abs(self._prev_dpad_y) <= 0.5:
            self._cmd[4] = np.clip(
                self._cmd[4] + np.sign(dpad_y) * self._freq_step,
                self._cmd_limits[4, 0],
                self._cmd_limits[4, 1],
            )
        self._prev_dpad_y = dpad_y

        # swing height (lt/rt, incremental)
        lt = axes.get("left_trigger", 0.0)
        if lt > 0.5 and self._prev_lt <= 0.5:
            self._cmd[9] = np.clip(
                self._cmd[9] - self._swing_step,
                self._cmd_limits[9, 0],
                self._cmd_limits[9, 1],
            )
        self._prev_lt = lt
        rt = axes.get("right_trigger", 0.0)
        if rt > 0.5 and self._prev_rt <= 0.5:
            self._cmd[9] = np.clip(
                self._cmd[9] + self._swing_step,
                self._cmd_limits[9, 0],
                self._cmd_limits[9, 1],
            )
        self._prev_rt = rt

        # roll (lb/rb, incremental)
        if self._rising("lb", buttons):
            self._cmd[11] = np.clip(
                self._cmd[11] + self._roll_step,
                self._cmd_limits[11, 0],
                self._cmd_limits[11, 1],
            )
        if self._rising("rb", buttons):
            self._cmd[11] = np.clip(
                self._cmd[11] - self._roll_step,
                self._cmd_limits[11, 0],
                self._cmd_limits[11, 1],
            )

        # Face buttons → gait presets
        for button, gait_name in self._gait_buttons.items():
            if self._rising(button, buttons):
                self._cmd[5:8] = self._gait_presets[gait_name]
                self._print_state()

        # Start → toggle auto-stand
        if self._rising("start", buttons):
            self._stand_when_still = not self._stand_when_still
            self._print_state()

        # Back → reset all commands
        if self._rising("back", buttons):
            self.reset()

        self._store_button_state(buttons)
        self._print_state()

    def _rising(self, name: str, buttons: dict[str, int]) -> bool:
        val = buttons.get(name, 0)
        prev = self._prev_buttons.get(name, 0)
        return bool(val and not prev)

    def _store_button_state(self, buttons: dict[str, int]):
        for name, val in buttons.items():
            self._prev_buttons[name] = val

    # ── Defaults & limits ───────────────────────────────────────────────

    def _default_gait_cmd(self) -> np.ndarray:
        gait_cmd = self._cmd_limits[4:12].mean(axis=1).astype(np.float32)
        gait_cmd[0] = 3.0
        gait_cmd[1:4] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        gait_cmd[4] = 0.5
        gait_cmd[5] = 0.08
        gait_cmd[6] = 0.25
        return gait_cmd

    def _set_default_gait_cmd(self):
        self._cmd[4:12] = self._default_gait_cmd()

    def _build_cmd_limits(self, env_cfg: DirectRLEnvCfg) -> np.ndarray:
        def _to_limits(rng):
            low, high = float(rng[0]), float(rng[1])
            return (min(low, high), max(low, high))

        cmd_limits = np.tile(np.array([-np.inf, np.inf], dtype=np.float32), (15, 1))
        cmd_limits[0] = _to_limits(env_cfg.lin_vel_x)
        cmd_limits[1] = _to_limits(env_cfg.lin_vel_y)
        cmd_limits[2] = _to_limits(env_cfg.ang_vel_yaw)
        cmd_limits[3] = _to_limits(env_cfg.body_height_cmd)
        cmd_limits[10] = _to_limits(env_cfg.body_pitch_range)
        cmd_limits[11] = _to_limits(env_cfg.body_roll_range)
        cmd_limits[4] = _to_limits(env_cfg.gait_frequency_cmd_range)
        cmd_limits[5] = _to_limits(env_cfg.gait_phase_cmd_range)
        cmd_limits[6] = _to_limits(env_cfg.gait_offset_cmd_range)
        cmd_limits[7] = _to_limits(env_cfg.gait_bound_cmd_range)
        cmd_limits[8] = _to_limits(env_cfg.gait_duration_cmd_range)
        cmd_limits[9] = _to_limits(env_cfg.footswing_height_range)
        cmd_limits[12] = _to_limits(env_cfg.stance_width_range)
        cmd_limits[13] = _to_limits(env_cfg.stance_length_range)
        return cmd_limits

    def _build_axis_scales(self) -> np.ndarray:
        """Maximum absolute value per base-command channel, for scaling [-1, 1] → cmd."""
        scales = np.zeros(6, dtype=np.float32)
        base_indices = [0, 1, 2, 3, 10, 11]
        for i, cmd_idx in enumerate(base_indices):
            scales[i] = max(abs(self._cmd_limits[cmd_idx, 0]), abs(self._cmd_limits[cmd_idx, 1]))
        return scales

    # ── Display ─────────────────────────────────────────────────────────

    def _print_state(self):
        cmd = np.clip(self._cmd, self._cmd_limits[:, 0], self._cmd_limits[:, 1])
        base_cmd = cmd[[0, 1, 2, 3, 10, 11]]
        gait_cmd = cmd[4:12]
        msg = (
            "[JOY] "
            f"vx={base_cmd[0]:.2f} vy={base_cmd[1]:.2f} yaw={base_cmd[2]:.2f} "
            f"h={base_cmd[3]:.2f} pitch={base_cmd[4]:.2f} roll={base_cmd[5]:.2f} | "
            f"freq={gait_cmd[0]:.2f} phase={gait_cmd[1]:.2f} off={gait_cmd[2]:.2f} "
            f"bound={gait_cmd[3]:.2f} dur={gait_cmd[4]:.2f} swing={gait_cmd[5]:.2f} "
            f"sw={gait_cmd[6]:.2f} sl={gait_cmd[7]:.2f} stand0={self._stand_when_still}"
        )
        print(msg)


def _register_go2_wtw_rma_classes():
    import rsl_rl.runners.on_policy_runner as on_policy_runner
    from robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_rma import Go2WTWActorCritic, Go2WTWPPO

    on_policy_runner.Go2WTWActorCritic = Go2WTWActorCritic
    on_policy_runner.Go2WTWPPO = Go2WTWPPO


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.resampling_time = 1e9
    env_cfg.episode_length_s = 1e9
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
            env_cfg.sim.device = args_cli.device
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    if not args_cli.headless:
        env.unwrapped.set_debug_vis(True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _register_go2_wtw_rma_classes()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = os.path.join(log_dir, "exported")
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    _export_policy(policy_nn, normalizer, export_dir)
    print(f"[INFO] Policy export attempted in {export_dir}")

    print(f"[INFO] Joystick config: {args_cli.joy_config}")
    joystick = WtwJoystick(env_cfg, env.unwrapped.device, args_cli.joy_config)

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        cmd = joystick.advance()
        env.unwrapped.commands[0] = cmd
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if args_cli.video and timestep == args_cli.video_length - 1:
            break
        timestep += 1
        elapsed = time.time() - start_time
        dt = env.unwrapped.step_dt
        if elapsed < dt:
            time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
