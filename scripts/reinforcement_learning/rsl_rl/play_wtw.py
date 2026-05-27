# SPDX-License-Identifier: Apache-2.0
"""Play script for Go2 Walk-These-Ways Direct RL task.

Usage:
    python scripts/reinforcement_learning/rsl_rl/play_wtw.py \
        --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
        --num_envs 16

    # With specific checkpoint:
    python scripts/reinforcement_learning/rsl_rl/play_wtw.py \
        --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
        --num_envs 1 --checkpoint <path/to/model_199.pt>

    # Record video:
    python scripts/reinforcement_learning/rsl_rl/play_wtw.py \
        --task RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0 \
        --num_envs 4 --video --video_length 500
"""

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play Go2 WalkTheseWays policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Go2-WalkTheseWays-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
import robot_lab.tasks.direct.go2_wtw  # noqa: F401 – ensure WTW/X5 gym env registration


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


def _register_go2_wtw_rma_classes(task_name: str | None = None):
    if task_name is not None and "WalkTheseWays" not in task_name and "ArmDisturbance" not in task_name:
        return
    from robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_rma import Go2WTWActorCritic, Go2WTWPPO
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.Go2WTWActorCritic = Go2WTWActorCritic
    on_policy_runner.Go2WTWPPO = Go2WTWPPO


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
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
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "play"),
                                       step_trigger=lambda step: step == 0,
                                       video_length=args_cli.video_length, disable_logger=True)

    # Enable command/pose debug visualization (GUI mode only; no-op in headless)
    if not args_cli.headless:
        env.unwrapped.set_debug_vis(True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _register_go2_wtw_rma_classes(task_name)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy
    export_dir = os.path.join(log_dir, "exported")
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    _export_policy(policy_nn, normalizer, export_dir)
    print(f"[INFO] Policy export attempted in {export_dir}")

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
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
