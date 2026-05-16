# SPDX-License-Identifier: Apache-2.0
"""Keyboard play script for Go2 Walk-These-Ways Direct RL task.

Keys:
    WASDQE: vx, vy, yaw
    UJ: height up/down
    IK: pitch up/down
    OL: roll up/down
    Z/X/C/V: gait presets (pronking/trotting/bounding/pacing)
    B/N: gait frequency +/-
    M/, : swing height +/-
    T: toggle auto-stand when vx/vy/yaw are zero (default off)
    P: reset all commands
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play WTW policy with keyboard input.")
parser.add_argument("--task", default="RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
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


class WtwKeyboard:
    def __init__(self, env_cfg: DirectRLEnvCfg, device: torch.device):
        import carb
        import omni

        self._sim_device = device
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=self: obj._on_keyboard_event(event, *args),
        )

        self._base_cmd = np.zeros(6, dtype=np.float32)  # vx, vy, yaw, height, pitch, roll
        self._base_limits = self._build_base_limits(env_cfg)
        self._base_key_map = self._build_base_key_map()

        self._gait_limits = self._build_gait_limits(env_cfg)
        self._gait_cmd = self._default_gait_cmd()
        self._stand_when_still = False
        self._gait_key_map = self._build_gait_key_map()
        self._gait_presets = {
            "Z": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "X": np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "C": np.array([0.0, 0.5, 0.0], dtype=np.float32),
            "V": np.array([0.0, 0.0, 0.5], dtype=np.float32),
        }

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def reset(self):
        self._base_cmd.fill(0.0)
        self._gait_cmd = self._default_gait_cmd()
        self._print_state()

    def advance(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_cmd = np.clip(self._base_cmd, self._base_limits[:, 0], self._base_limits[:, 1])
        gait_cmd = np.clip(self._gait_cmd, self._gait_limits[:, 0], self._gait_limits[:, 1])
        if self._stand_when_still and np.linalg.norm(base_cmd[:3]) < 1e-4:
            gait_cmd[0] = 0.0
            gait_cmd[1:4] = 0.0
            gait_cmd[4] = 0.5
        base_cmd_t = torch.tensor(base_cmd, dtype=torch.float32, device=self._sim_device)
        gait_cmd_t = torch.tensor(gait_cmd, dtype=torch.float32, device=self._sim_device)
        return base_cmd_t, gait_cmd_t

    def _default_gait_cmd(self) -> np.ndarray:
        gait_cmd = self._gait_limits.mean(axis=1).astype(np.float32)
        gait_cmd[0] = 3.0
        gait_cmd[1:4] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        gait_cmd[4] = 0.5
        gait_cmd[5] = 0.08
        gait_cmd[6] = 0.25
        return gait_cmd

    def _build_base_limits(self, env_cfg: DirectRLEnvCfg) -> np.ndarray:
        def _to_limits(rng):
            low, high = float(rng[0]), float(rng[1])
            return (min(low, high), max(low, high))

        base_limits = [
            _to_limits(env_cfg.lin_vel_x),
            _to_limits(env_cfg.lin_vel_y),
            _to_limits(env_cfg.ang_vel_yaw),
            _to_limits(env_cfg.body_height_cmd),
            _to_limits(env_cfg.body_pitch_range),
            _to_limits(env_cfg.body_roll_range),
        ]
        return np.asarray(base_limits, dtype=np.float32)

    def _build_gait_limits(self, env_cfg: DirectRLEnvCfg) -> np.ndarray:
        def _to_limits(rng):
            low, high = float(rng[0]), float(rng[1])
            return (min(low, high), max(low, high))

        gait_limits = [
            _to_limits(env_cfg.gait_frequency_cmd_range),
            _to_limits(env_cfg.gait_phase_cmd_range),
            _to_limits(env_cfg.gait_offset_cmd_range),
            _to_limits(env_cfg.gait_bound_cmd_range),
            _to_limits(env_cfg.gait_duration_cmd_range),
            _to_limits(env_cfg.footswing_height_range),
            _to_limits(env_cfg.stance_width_range),
            _to_limits(env_cfg.stance_length_range),
        ]
        return np.asarray(gait_limits, dtype=np.float32)

    def _build_base_key_map(self) -> dict[str, np.ndarray]:
        sens = np.array([0.2, 0.2, 0.2, 0.02, 0.05, 0.05], dtype=np.float32)
        return {
            "W": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * sens[0],
            "S": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * sens[0],
            "A": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * sens[1],
            "D": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * sens[1],
            "Q": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * sens[2],
            "E": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * sens[2],
            "U": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * sens[3],
            "J": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * sens[3],
            "I": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * sens[4],
            "K": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * sens[4],
            "O": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * sens[5],
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * sens[5],
        }

    def _build_gait_key_map(self) -> dict[str, np.ndarray]:
        steps = np.array([0.2, 0.02], dtype=np.float32)
        return {
            "B": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * steps[0],
            "N": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * steps[0],
            "M": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * steps[1],
            "COMMA": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * steps[1],
            ",": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * steps[1],
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "P":
                self.reset()
            elif event.input.name == "T":
                self._stand_when_still = not self._stand_when_still
                self._print_state()
            elif event.input.name in self._base_key_map:
                self._base_cmd += self._base_key_map[event.input.name]
                self._base_cmd = np.clip(self._base_cmd, self._base_limits[:, 0], self._base_limits[:, 1])
                self._print_state()
            elif event.input.name in self._gait_presets:
                gait = self._gait_presets[event.input.name]
                self._gait_cmd[1:4] = gait
                self._gait_cmd = np.clip(self._gait_cmd, self._gait_limits[:, 0], self._gait_limits[:, 1])
                self._print_state()
            elif event.input.name in self._gait_key_map:
                self._gait_cmd += self._gait_key_map[event.input.name]
                self._gait_cmd = np.clip(self._gait_cmd, self._gait_limits[:, 0], self._gait_limits[:, 1])
                self._print_state()
        return True

    def _print_state(self):
        base_cmd = np.clip(self._base_cmd, self._base_limits[:, 0], self._base_limits[:, 1])
        gait_cmd = np.clip(self._gait_cmd, self._gait_limits[:, 0], self._gait_limits[:, 1])
        msg = (
            "[KEY] "
            f"vx={base_cmd[0]:.2f} vy={base_cmd[1]:.2f} yaw={base_cmd[2]:.2f} "
            f"h={base_cmd[3]:.2f} pitch={base_cmd[4]:.2f} roll={base_cmd[5]:.2f} | "
            f"freq={gait_cmd[0]:.2f} phase={gait_cmd[1]:.2f} off={gait_cmd[2]:.2f} "
            f"bound={gait_cmd[3]:.2f} dur={gait_cmd[4]:.2f} swing={gait_cmd[5]:.2f} "
            f"sw={gait_cmd[6]:.2f} sl={gait_cmd[7]:.2f} stand0={self._stand_when_still}"
        )
        print(msg)


def _register_go2_wtw_rma_classes():
    from robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_rma import Go2WTWActorCritic, Go2WTWPPO
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.Go2WTWActorCritic = Go2WTWActorCritic
    on_policy_runner.Go2WTWPPO = Go2WTWPPO


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
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

    keyboard = WtwKeyboard(env_cfg, env.unwrapped.device)

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        base_cmd, gait_cmd = keyboard.advance()
        cmd = env.unwrapped.commands[0].clone()
        cmd[0] = base_cmd[0]
        cmd[1] = base_cmd[1]
        cmd[2] = base_cmd[2]
        cmd[3] = base_cmd[3]
        cmd[10] = base_cmd[4]
        cmd[11] = base_cmd[5]
        if cmd.shape[0] > 4:
            cmd[4] = gait_cmd[0]
        if cmd.shape[0] > 5:
            cmd[5] = gait_cmd[1]
        if cmd.shape[0] > 6:
            cmd[6] = gait_cmd[2]
        if cmd.shape[0] > 7:
            cmd[7] = gait_cmd[3]
        if cmd.shape[0] > 8:
            cmd[8] = gait_cmd[4]
        if cmd.shape[0] > 9:
            cmd[9] = gait_cmd[5]
        if cmd.shape[0] > 12:
            cmd[12] = gait_cmd[6]
        if cmd.shape[0] > 13:
            cmd[13] = gait_cmd[7]
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
