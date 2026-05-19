"""Standalone script to print joint names from GO2_X5 USD.

Usage:
    /home/user/Apps/isaac-lab-2.3.0/isaaclab.sh -p scripts/tools/print_joint_names.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print GO2_X5 joint names from USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from robot_lab.assets.go2_x5 import GO2_X5_CFG

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    cfg = GO2_X5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)

    sim.reset()
    robot.update(0.0)

    print("\n=== GO2_X5 Joint Names (simulator natural order) ===")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")
    print(f"\nTotal joints: {robot.num_joints}")
    print("=====================================================\n")

    simulation_app.close()

if __name__ == "__main__":
    main()
