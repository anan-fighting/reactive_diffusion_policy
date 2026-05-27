"""
eval_real_robot_ur5e.py
=======================
Universal Robots UR5e 真机部署推理脚本。

参考 eval_real_robot_nova5.py 实现，支持：
  - LDP（Latent Diffusion Policy）推理部署（快慢双系统）
  - AT（Asymmetric Tokenizer）仅作为 LDP 的子模块使用

使用方法：
----------
## 1. 启动 UR5e 机器人 HTTP 服务器（在另一个终端）
python -m reactive_diffusion_policy.real_world.robot.ur5e_server \
    --robot_ip 192.168.1.103 \
    --host_ip 0.0.0.0 \
    --port 8093

## 2. 启动相机节点（Device Mapping Server 会在其中自动内嵌启动）
##    Device Mapping Server 是仓库内置的轻量 HTTP 服务（默认端口 8062），
##    负责动态探测当前接入的 RealSense / USB 触觉相机设备，
##    并维护"设备 → ROS2 话题名"映射表，供 RealEnv 订阅正确的话题。
##    代码位于：reactive_diffusion_policy/real_world/device_mapping/device_mapping_server.py
sudo setcap cap_sys_nice+eip $(which python)
source /opt/ros/humble/setup.bash
python camera_node_launcher.py task=ur5e_rdp_image_tactile_emb_ldp_24fps

## 3. 运行本推理脚本（LDP 部署，加载 LDP checkpoint，内含 AT 子模块）
python eval_real_robot_ur5e.py \
    --config-name train_latent_diffusion_unet_real_image_workspace \
    task=ur5e_rdp_image_tactile_emb_ldp_24fps \
    # ↑ 与 camera_node_launcher.py 使用相同的 task，确保 device_mapping_server 端口匹配
    ckpt_path="data/outputs/2026.05.14/15.09.20_train_latent_diffusion_unet_image_umi_rdp_image_tactile_emb_ldp_24fps_0514150913/checkpoints/latest.ckpt" \
    at_load_dir="data/outputs/2026.05.12/18.08.58_train_vae_umi_rdp_image_tactile_emb_at_24fps_0512180857/checkpoints" \
    hydra.run.dir="data/outputs/ur5e_eval"

参数说明：
----------
- ckpt_path      : 训练好的 LDP checkpoint 路径（.ckpt 文件）
- at_load_dir    : 训练好的 AT checkpoint 目录路径
- task           : 使用 ur5e_rdp_image_tactile_emb_ldp_24fps 任务配置
- hydra.run.dir  : 推理输出目录（日志、录像等）

快慢系统架构说明：
------------------
                  ┌──────────────────────────────────────────────┐
                  │          eval_real_robot_ur5e.py              │
                  │  加载 LDP checkpoint（包含 AT 子模块）          │
                  └─────────────────┬────────────────────────────┘
                                    │ policy.predict_action()
                  ┌─────────────────▼────────────────────────────┐
                  │        慢系统（推理线程，6 Hz）                 │
                  │  ResNet 编码 obs → LDP 生成 latent action 序列 │
                  │  → 写入 EnsembleBuffer                        │
                  └─────────────────┬────────────────────────────┘
                                    │ EnsembleBuffer.get_action()
                  ┌─────────────────▼────────────────────────────┐
                  │        快系统（控制线程，24 Hz）                │
                  │  取 latent action → AT RNN Decoder            │
                  │  + 最新触觉 extended_obs → 真实 TCP 动作       │
                  │  → HTTP → UR5eServer → servoL → UR5e          │
                  └──────────────────────────────────────────────┘
"""

# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy

import os
import psutil

# 限制 numpy / OpenCV 线程数，避免与机器人控制线程竞争 CPU
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
cv2.setNumThreads(12)

# 绑定前 10 个 CPU 核，降低调度抖动
total_cores = psutil.cpu_count()
num_cores_to_bind = 10
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------------------
# 主函数：通过 Hydra 配置加载模型并运行真机推理
# ---------------------------------------------------------------------------
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'reactive_diffusion_policy', 'config')),
    # 默认使用 LDP workspace 配置；部署纯 AT 时使用 train_at_workspace
    config_name="train_latent_diffusion_unet_real_image_workspace"
)
def main(cfg):
    # ------------------------------------------------------------------
    # 1. 加载 checkpoint
    # ------------------------------------------------------------------
    ckpt_path = cfg.ckpt_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ------------------------------------------------------------------
    # 2. 根据模型类型做初始化
    #    目前支持：
    #      - LDP (latent diffusion)：cfg.name 包含 'diffusion'
    #        - 标准 LDP：cfg.name 不含 'latent'
    #        - LDP + AT RNN decoder：cfg.name 含 'latent'（当前主要路径）
    # ------------------------------------------------------------------
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        # LDP 需要将 normalizer 传给内置的 AT 模块
        if 'latent' in cfg.name:
            policy.at.set_normalizer(policy.normalizer)

        device = torch.device('cuda')
        policy.eval().to(device)

        # 设置 DDIM 推理步数（减少步数可提升推理速度，默认 8 步）
        policy.num_inference_steps = 8  # DDIM inference iterations
    else:
        raise NotImplementedError(
            f"不支持的模型类型：cfg.name={cfg.name}。"
            "请使用 LDP checkpoint（train_latent_diffusion_unet_real_image_workspace）。"
        )

    # ------------------------------------------------------------------
    # 3. 实例化 EnvRunner 并运行评估
    #    env_runner 由 Hydra 根据 task 配置自动实例化 RealRunner，
    #    其中包含快慢系统逻辑，无需在本脚本中手动实现。
    # ------------------------------------------------------------------
    env_runner = hydra.utils.instantiate(cfg.task.env_runner)
    env_runner.run(policy)


# %%
if __name__ == '__main__':
    main()
