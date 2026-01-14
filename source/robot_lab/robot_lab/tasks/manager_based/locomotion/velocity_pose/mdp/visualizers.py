# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for VelocityPose commands."""

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VelocityPoseCommandVisualizer:
    """可视化VelocityPose的6D指令: [vx, vy, w_z, height, roll, pitch]
    
    为每个机器人在质心正上方显示：
    - 目标姿态坐标轴（原点高度 = 目标高度指令）
    - RGB坐标轴根据roll/pitch旋转
    """
    
    def __init__(self, env: "ManagerBasedEnv", num_envs: int):
        """初始化可视化器
        
        Args:
            env: Manager-based RL环境
            num_envs: 环境数量
        """
        self.env = env
        self.num_envs = num_envs
        self.device = env.device
        
        # 创建markers用于可视化
        self._create_markers()
        
        print(f"[VelocityPoseVisualizer] 初始化完成，为{num_envs}个机器人创建可视化")
    
    def _create_markers(self):
        """创建可视化markers - 为每个元素创建独立的marker
        
        创建两组坐标轴：
        1. 目标姿态（亮色）- 来自命令指令
        2. 当前姿态（暗色）- 来自重力投影计算
        """
        # 坐标轴尺寸
        axis_length = 0.3   # 30cm长的圆柱
        axis_radius = 0.01  # 1cm粗
        sphere_radius = 0.025  # 2.5cm球体
        
        # ============================================================
        # 目标姿态可视化（亮色，来自指令）
        # ============================================================
        
        # 目标原点球 - 黄色
        target_origin_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_origin",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=sphere_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # 亮黄色
                ),
            },
        )
        self.target_origin_marker = VisualizationMarkers(target_origin_cfg)
        
        # 目标X轴 - 亮红色
        target_x_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_x_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 亮红
                ),
            },
        )
        self.target_x_axis_marker = VisualizationMarkers(target_x_axis_cfg)
        
        # 目标Y轴 - 亮绿色
        target_y_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_y_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # 亮绿
                ),
            },
        )
        self.target_y_axis_marker = VisualizationMarkers(target_y_axis_cfg)
        
        # 目标Z轴 - 亮蓝色
        target_z_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/target_z_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 亮蓝
                ),
            },
        )
        self.target_z_axis_marker = VisualizationMarkers(target_z_axis_cfg)
        
        # ============================================================
        # 当前姿态可视化（暗色，来自重力投影）
        # ============================================================
        
        # 当前原点球 - 橙色
        current_origin_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_origin",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=sphere_radius * 0.8,  # 稍小一点
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # 橙色
                ),
            },
        )
        self.current_origin_marker = VisualizationMarkers(current_origin_cfg)
        
        # 当前X轴 - 暗红色
        current_x_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_x_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.0, 0.0)),  # 暗红
                ),
            },
        )
        self.current_x_axis_marker = VisualizationMarkers(current_x_axis_cfg)
        
        # 当前Y轴 - 暗绿色
        current_y_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_y_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 0.0)),  # 暗绿
                ),
            },
        )
        self.current_y_axis_marker = VisualizationMarkers(current_y_axis_cfg)
        
        # 当前Z轴 - 暗蓝色
        current_z_axis_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/VelocityPoseCommand/current_z_axis",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=axis_radius * 0.8,
                    height=axis_length,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.6)),  # 暗蓝
                ),
            },
        )
        self.current_z_axis_marker = VisualizationMarkers(current_z_axis_cfg)
    
    def update(self, commands: torch.Tensor, robot: Articulation):
        """更新可视化 - 同时显示目标姿态和当前姿态
        
        Args:
            commands: (N, 6) [vx, vy, w_z, height, roll, pitch]
            robot: 机器人articulation
        """
        # 获取当前base位置和姿态
        base_pos_w = robot.data.root_pos_w  # (N, 3)
        base_quat_w = robot.data.root_quat_w  # (N, 4) [w, x, y, z]
        
        # 解析目标命令
        target_height = commands[:, 3]  # 目标高度
        target_roll = commands[:, 4]    # 目标roll角
        target_pitch = commands[:, 5]   # 目标pitch角
        
        # ============================================================
        # 步骤1: 从重力投影计算当前实际姿态（roll和pitch）
        # ============================================================
        current_roll, current_pitch = self._compute_orientation_from_gravity(robot)
        current_height = base_pos_w[:, 2]  # 当前实际高度
        
        # ============================================================
        # 步骤2: 计算目标姿态的可视化
        # ============================================================
        # 目标原点位置：质心XY + 目标高度Z
        target_origin_pos = base_pos_w.clone()
        target_origin_pos[:, 2] = target_height
        
        # 计算目标姿态四元数
        target_quat = quat_from_euler_xyz(target_roll, target_pitch, torch.zeros_like(target_roll))
        
        # 计算目标坐标轴（圆柱一端在原点）
        target_axes = self._compute_axis_markers(target_origin_pos, target_quat, axis_length=0.3)
        
        # ============================================================
        # 步骤3: 计算当前姿态的可视化
        # ============================================================
        # 当前原点位置：质心实际位置
        current_origin_pos = base_pos_w.clone()
        
        # 计算当前姿态四元数
        current_quat = quat_from_euler_xyz(current_roll, current_pitch, torch.zeros_like(current_roll))
        
        # 计算当前坐标轴（圆柱一端在原点）
        current_axes = self._compute_axis_markers(current_origin_pos, current_quat, axis_length=0.3)
        
        # ============================================================
        # 步骤4: 更新所有markers
        # ============================================================
        identity_quat = torch.zeros((self.num_envs, 4), device=self.device)
        identity_quat[:, 0] = 1.0
        
        # 目标姿态（亮色）
        self.target_origin_marker.visualize(target_origin_pos, identity_quat)
        self.target_x_axis_marker.visualize(target_axes['x_pos'], target_axes['x_quat'])
        self.target_y_axis_marker.visualize(target_axes['y_pos'], target_axes['y_quat'])
        self.target_z_axis_marker.visualize(target_axes['z_pos'], target_axes['z_quat'])
        
        # 当前姿态（暗色）
        self.current_origin_marker.visualize(current_origin_pos, identity_quat)
        self.current_x_axis_marker.visualize(current_axes['x_pos'], current_axes['x_quat'])
        self.current_y_axis_marker.visualize(current_axes['y_pos'], current_axes['y_quat'])
        self.current_z_axis_marker.visualize(current_axes['z_pos'], current_axes['z_quat'])
    
    def _compute_orientation_from_gravity(self, robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        """从重力投影计算当前base的roll和pitch角
        
        原理：
        - 重力在base坐标系中的投影 = R^T * g_world
        - 其中 g_world = [0, 0, -1] （世界坐标系中重力向下）
        - projected_gravity 就是重力在base坐标系中的表示
        
        从projected_gravity反推姿态：
        - gx, gy, gz = projected_gravity的三个分量
        - roll = atan2(gy, gz)  # 左右倾斜
        - pitch = atan2(-gx, sqrt(gy^2 + gz^2))  # 前后俯仰
        
        Args:
            robot: 机器人articulation
            
        Returns:
            roll: (N,) 当前roll角（弧度）
            pitch: (N,) 当前pitch角（弧度）
        """
        # 获取重力投影（base坐标系中的重力方向）
        # 这个已经由环境计算好了
        projected_gravity = robot.data.projected_gravity_b  # (N, 3) [gx, gy, gz]
        
        gx = projected_gravity[:, 0]
        gy = projected_gravity[:, 1]
        gz = projected_gravity[:, 2]
        
        # 从重力投影计算roll和pitch
        # roll: 绕X轴旋转（左右倾斜）
        roll = torch.atan2(gy, gz)
        
        # pitch: 绕Y轴旋转（前后俯仰）
        pitch = torch.atan2(-gx, torch.sqrt(gy**2 + gz**2))
        
        return roll, pitch
    
    def _compute_axis_markers(self, origin_pos: torch.Tensor, orientation_quat: torch.Tensor, 
                             axis_length: float) -> dict:
        """计算坐标轴markers的位置和方向（圆柱一端在原点）
        
        修改为：圆柱的一端在原点处相交，而不是中心在原点
        
        Args:
            origin_pos: (N, 3) 原点位置
            orientation_quat: (N, 4) 姿态四元数
            axis_length: 圆柱长度
            
        Returns:
            dict with keys: x_pos, x_quat, y_pos, y_quat, z_pos, z_quat
        """
        # 圆柱一端在原点，另一端延伸出去
        # 所以圆柱中心应该在 origin + axis_direction * (length/2)
        half_length = axis_length / 2.0
        
        # --- X轴（红色）---
        # 方向：沿X轴正方向
        x_direction = self._rotate_vector(
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        x_pos = origin_pos + x_direction * half_length  # 圆柱中心
        
        # 圆柱姿态：从Z轴旋转到X轴，再应用整体姿态
        x_quat = self._quat_multiply(
            orientation_quat,
            quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device)
            )
        )
        
        # --- Y轴（绿色）---
        y_direction = self._rotate_vector(
            torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        y_pos = origin_pos + y_direction * half_length
        
        y_quat = self._quat_multiply(
            orientation_quat,
            quat_from_euler_xyz(
                torch.ones(self.num_envs, device=self.device) * 1.5708,
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
        )
        
        # --- Z轴（蓝色）---
        z_direction = self._rotate_vector(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3),
            orientation_quat
        )
        z_pos = origin_pos + z_direction * half_length
        
        z_quat = orientation_quat.clone()
        
        return {
            'x_pos': x_pos,
            'x_quat': x_quat,
            'y_pos': y_pos,
            'y_quat': y_quat,
            'z_pos': z_pos,
            'z_quat': z_quat,
        }
    
    def _rotate_vector(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """用四元数旋转向量
        
        使用Isaac Lab的内置函数
        
        Args:
            vec: (N, 3) 向量
            quat: (N, 4) 四元数 [w, x, y, z]
            
        Returns:
            rotated_vec: (N, 3) 旋转后的向量
        """
        from isaaclab.utils.math import quat_apply
        return quat_apply(quat, vec)
    
    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数乘法 q1 * q2
        
        使用Isaac Lab的内置函数
        
        Args:
            q1, q2: (N, 4) [w, x, y, z]
            
        Returns:
            result: (N, 4) q1 * q2
        """
        return quat_mul(q1, q2)
    
    def reset(self, env_ids: torch.Tensor):
        """重置指定环境的可视化"""
        # Markers会自动处理重置，无需额外操作
        pass
