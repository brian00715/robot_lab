# SPDX-License-Identifier: Apache-2.0
"""Play script for Go2X5 Walk-These-Ways with annotated video recording.

Records a video with per-frame text overlays showing:
  - Gait mode (trot / pace / bound / pronk)
  - Arm disturbance mode (circular / figure_eight / ... / static)
  - Arm curriculum stage and motion scale
  - Linear / angular velocity commands
  - Step counter

Usage:
    python scripts/reinforcement_learning/rsl_rl/play_wtw_annotated.py \
        --task RobotLab-Isaac-Go2X5-ArmDisturbance-Direct-v0 \
        --num_envs 1 \
        --checkpoint logs/rsl_rl/go2_x5_arm_disturbance/<run>/model_XXXXX.pt \
        --video --video_length 500 --env_id 0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play Go2X5 WTW with annotated video.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_id", type=int, default=0, help="Which env index to annotate (0-indexed).")
parser.add_argument("--output_name", type=str, default=None, help="Output video filename (without .mp4). Defaults to <ckpt>_annotated.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Go2X5-ArmDisturbance-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True  # headless for recording: avoids GUI event-loop freeze
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import math
import numpy as np
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
import robot_lab.tasks.direct.go2_wtw  # noqa: F401


# ---------------------------------------------------------------------------
# PIL-based frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw text lines onto a numpy HWC uint8 frame using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to load a monospaced font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    x, y = 10, 10
    line_height = 22
    for line in lines:
        # Draw black shadow for readability
        draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 0))
        y += line_height

    return np.array(img)


def _get_gait_name(env_raw, env_id: int) -> str:
    """Return gait category name for the given env index."""
    try:
        cat_idx = int(env_raw.env_command_categories[env_id])
        return env_raw.category_names[cat_idx]
    except Exception:
        return "unknown"


def _get_arm_mode(env_raw, env_id: int) -> str:
    """Return arm motion mode name for the given env index (X5 env only)."""
    try:
        ctrl = env_raw._arm_controller
        return ctrl.motion_modes[env_id]
    except Exception:
        return "N/A"


def _get_arm_info(env_raw) -> tuple[int, float]:
    """Return (arm_stage, arm_motion_scale) from X5 env."""
    try:
        return env_raw._arm_stage + 1, env_raw._arm_motion_scale
    except Exception:
        return 0, 0.0


# ---------------------------------------------------------------------------
# RSL-RL class registration (same as play_wtw.py)
# ---------------------------------------------------------------------------

def _register_go2_wtw_rma_classes(task_name: str | None = None):
    if task_name is not None and "WalkTheseWays" not in task_name and "ArmDisturbance" not in task_name:
        return
    from robot_lab.tasks.direct.go2_wtw.agents.rsl_rl_rma import Go2WTWActorCritic, Go2WTWPPO
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.Go2WTWActorCritic = Go2WTWActorCritic
    on_policy_runner.Go2WTWPPO = Go2WTWPPO


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # Play config: resample gait/pose commands every 10 s; never force-reset by timeout
    env_cfg.resampling_time = 10.0
    env_cfg.episode_length_s = 1e6
    env_cfg.play_resample_arm = True  # also resample arm mode/stage every 10 s

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    # Always render as rgb_array so we can grab frames
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env_raw = env.unwrapped  # Direct access to our env

    # Enable debug visualization so pose/velocity markers appear in rendered frames
    env_raw.set_debug_vis(True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _register_go2_wtw_rma_classes(task_name)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env_raw.device)
    print("[INFO] Policy loaded successfully.")

    # ---- Video writer setup -----------------------------------------------
    frames: list[np.ndarray] = []
    env_id = min(args_cli.env_id, args_cli.num_envs - 1)

    # ---- Run loop --------------------------------------------------------
    obs = env.get_observations()
    timestep = 0
    total_steps = args_cli.video_length if args_cli.video else int(1e9)

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        # ---- Capture and annotate frame -----------------------------------
        if args_cli.video:
            frame = env_raw.render()  # HWC uint8 numpy
            if frame is not None:
                # --- gather info ---
                gait = _get_gait_name(env_raw, env_id)
                arm_mode = _get_arm_mode(env_raw, env_id)
                arm_stage, arm_scale = _get_arm_info(env_raw)

                cmds = env_raw.commands[env_id].cpu()
                vx    = cmds[0].item()
                vy    = cmds[1].item()
                vyaw  = cmds[2].item()
                freq  = cmds[4].item() if len(cmds) > 4 else 0.0
                # cmd height offset (idx 3) + base_height_target = absolute target height
                cmd_h_offset = cmds[3].item() if len(cmds) > 3 else 0.0
                cmd_pitch_deg = math.degrees(-cmds[10].item()) if len(cmds) > 10 else 0.0
                cmd_roll_deg  = math.degrees(-cmds[11].item()) if len(cmds) > 11 else 0.0
                target_h = cmd_h_offset + env_raw.cfg.base_height_target

                # current pose angles from quaternion
                from isaaclab.utils.math import euler_xyz_from_quat
                base_quat = env_raw.robot.data.root_quat_w[env_id].unsqueeze(0)
                cur_roll_t, cur_pitch_t, _ = euler_xyz_from_quat(base_quat)
                cur_roll_deg  = math.degrees(cur_roll_t[0].item())
                cur_pitch_deg = math.degrees(cur_pitch_t[0].item())

                base_h = env_raw.base_pos[env_id, 2].item()

                lines = [
                    f"Step: {timestep:04d}",
                    f"Gait:     {gait}  ({freq:.2f} Hz)",
                    f"Arm mode: {arm_mode}",
                    f"Arm stage: {arm_stage}  scale: {arm_scale:.2f}",
                    f"Vel cmd: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}",
                    f"--- Height ---",
                    f"  CMD h: {target_h:.3f} m",
                    f"  CUR h: {base_h:.3f} m  (Δ={base_h - target_h:+.3f} m)",
                    f"--- Pose ---",
                    f"  CMD pitch={cmd_pitch_deg:+.1f}°  roll={cmd_roll_deg:+.1f}°",
                    f"  CUR pitch={cur_pitch_deg:+.1f}°  roll={cur_roll_deg:+.1f}°",
                    f"  (R/G/B=cmd | orange/cyan/white=cur)",
                ]
                annotated = _annotate_frame(frame, lines)
                frames.append(annotated)

        timestep += 1
        if timestep >= total_steps:
            break

        if not args_cli.video:
            elapsed = time.time() - start_time
            dt = env_raw.step_dt
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ---- Save video -------------------------------------------------------
    if args_cli.video and frames:
        import imageio
        video_dir = os.path.join(log_dir, "videos", "play_annotated")
        os.makedirs(video_dir, exist_ok=True)
        ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
        out_name = args_cli.output_name if args_cli.output_name else f"{ckpt_name}_annotated"
        video_path = os.path.join(video_dir, f"{out_name}.mp4")
        fps = max(1, int(round(1.0 / env_raw.step_dt)))
        imageio.mimwrite(video_path, frames, fps=fps, codec="libx264", quality=8)
        print(f"[INFO] Annotated video saved to: {video_path}")

    env.close()


if __name__ == "__main__":
    main()
