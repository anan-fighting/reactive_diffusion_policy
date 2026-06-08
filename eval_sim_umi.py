"""
eval_sim.py
===========
离线推理验证脚本：用 zarr 数据集中的真实观测作为模型输入，
模拟慢-快双系统的完整推理链路，将模型输出的 action 与数据集中的 GT action 进行对比。

目的：在没有真机和相机的情况下，验证模型训练/推理链路是否正确。

使用方法：
----------
# 基于UMI数据测试
CUDA_VISIBLE_DEVICES=0 python eval_sim_umi.py \
    --config-name train_latent_diffusion_unet_real_image_workspace \
    task=umi_rdp_image_tactile_emb_ldp_24fps \
    task.dataset_path=data/hf_dataset/dataset_mini/umi_rdp_zarr \
    +ckpt_path="data/outputs/2026.05.14_umi_ldp/15.09.20_train_latent_diffusion_unet_image_umi_rdp_image_tactile_emb_ldp_24fps_0514150913/checkpoints/epoch-0110-train_loss-0.121.ckpt" \
    +at_load_dir="data/outputs/2026.05.12_umi_at/18.08.58_train_vae_umi_rdp_image_tactile_emb_at_24fps_0512180857/checkpoints/epoch-0600-train_loss-0.001908.ckpt" \
    hydra.run.dir="data/outputs/eval_sim" \
    +episode_idx=0 \
    +output_fig="eval_sim_result.png"

可选参数（通过 Hydra 覆盖）：
    +episode_idx=0          选择第几个 episode（默认 0）
    +n_inference_steps=30   推理前多少步（默认测试整个 episode）
    +output_fig=eval_sim_result.png  输出图片路径

慢-快系统模拟说明：
------------------
- 慢系统（6Hz）：每 steps_per_inference 步调用一次 LDP，生成 latent action 序列，写入 EnsembleBuffer
- 快系统（24Hz）：每步从 EnsembleBuffer 取 latent action，用 AT RNN decoder + 当前触觉 extended_obs 解码出真实 action
- 本脚本用 zarr 中的观测替代真实传感器，完全模拟 real_runner.py 的推理流程
"""

# %%
import pathlib
import os
import sys
import argparse

import torch
import dill
import hydra
import numpy as np
import zarr
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from typing import Dict

from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)
from reactive_diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from reactive_diffusion_policy.common.pytorch_util import dict_apply

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ──────────────────────────────────────────────
# 从 zarr 读取单个 episode 的所有观测
# ──────────────────────────────────────────────

def load_episode_from_zarr(zarr_path: str, episode_idx: int):
    """
    从 replay_buffer.zarr 中读取指定 episode 的全部数据。
    返回 dict，key 与 zarr data/ 下字段名一致。
    """
    z = zarr.open(os.path.join(zarr_path, 'replay_buffer.zarr'), 'r')
    ends = z['meta/episode_ends'][:] # 读取数据集里记录的 每个 episode 结束帧的索引
    starts = np.concatenate([[0], ends[:-1]]) # 计算 每个 episode 开始的索引

    assert episode_idx < len(ends), \
        f"episode_idx={episode_idx} 超出范围（共 {len(ends)} 个 episode）"

    s, e = int(starts[episode_idx]), int(ends[episode_idx])
    print(f"[zarr] Episode {episode_idx}: frames [{s}, {e})，共 {e - s} 帧")

    # 遍历数据集里所有传感器字段：image, wrist_image, vio_pose, action, gripper, wrench...
    # 每个字段都 只截取 s ~ e 帧，全部放进 episode 字典
    episode = {}
    for key in z['data'].keys():
        episode[key] = z[f'data/{key}'][s:e]

    return episode, s, e


# ──────────────────────────────────────────────
# 构建单步的 obs_dict（模拟 real_env.get_obs）
# ──────────────────────────────────────────────

def build_obs_dict(episode: dict,
                   t: int,
                   n_obs_steps: int,
                   obs_temporal_downsample_ratio: int,
                   shape_meta: dict,
                   is_extended_obs: bool = False) -> Dict[str, np.ndarray]:
    """
    函数作用：
        模拟真机推理时 real_env.get_obs() 的行为。
        给定当前时刻 t，从离线 episode 中取出"观测窗口"，
        即最近 n_obs_steps 帧的历史数据，作为模型的输入。

    参数说明：
        episode: 整个 episode 的所有数据，key 为字段名（如 'action'、'left_wrist_img'），value 为 (T, ...) 的 numpy 数组
        t: 当前控制步索引（第几帧）
        n_obs_steps: 模型输入需要多少帧历史观测（配置中为 2）
        obs_temporal_downsample_ratio: 时间降采样比例，例如为 2 时每隔一帧取一帧；为 1 时不降采样
        shape_meta: 配置中的 shape_meta，描述每个字段的类型和维度（本函数预留接口，暂未使用）
        is_extended_obs: 是否构建快系统的 extended_obs（预留接口，暂未使用）

    返回：
        obs_raw: dict，每个 key 对应形状为 (n_obs_steps, ...) 的 numpy 数组
    """
    # 计算原始需要截取的帧数窗口大小。
    # 例如 n_obs_steps=2，ratio=1 → window=2；ratio=4 → window=8（从 8 帧里隔 4 帧取 2 帧）
    window = n_obs_steps * obs_temporal_downsample_ratio

    # 截取窗口的右边界（不含），即"取到第 t 帧为止"，+1 是因为 Python 切片不含右端点。
    # min() 防止 t 超过 episode 总长度
    t_end = min(t + 1, len(episode['action']))

    # 截取窗口的左边界，向前取 window 帧。
    # max(0, ...) 防止在 episode 开头越界
    t_start = max(0, t_end - window)

    obs_raw = {}
    for key in episode:
        # 取出 [t_start, t_end) 的原始片段，最多 window 帧，形状 (<=window, ...)
        seg = episode[key][t_start:t_end]

        # 时间降采样（核心逻辑，分两步）：
        #   步骤1: seg[::-ratio]  从最后一帧开始，每隔 ratio 帧取一帧（倒序）
        #   步骤2: [::-1]         再翻转回正序（时间从旧到新）
        # 举例：seg=[帧0,帧1,帧2,帧3]，ratio=2
        #   → 倒序隔帧 → [帧3, 帧1] → 翻转 → [帧1, 帧3]（取最近的2帧，保证时间正序）
        sampled = seg[::-obs_temporal_downsample_ratio][::-1]  # (<=n_obs_steps, ...)

        # 前端补齐：当 episode 开头历史帧不足 n_obs_steps 时（如第 0 帧只有 1 帧历史），
        # 用最早的那帧重复填充到最前面，保证输出始终是 (n_obs_steps, ...) 的固定形状
        if sampled.shape[0] < n_obs_steps:
            pad = np.repeat(sampled[:1], n_obs_steps - sampled.shape[0], axis=0)
            sampled = np.concatenate([pad, sampled], axis=0)

        obs_raw[key] = sampled  # (n_obs_steps, ...)

    return obs_raw


def preprocess_obs_dict(obs_raw: dict,
                        shape_meta: dict,
                        is_extended_obs: bool = False) -> dict:
    """
    对观测做维度对齐和图像归一化，模拟 get_real_obs_dict() 的处理。
    """
    if is_extended_obs:
        obs_shape_meta = shape_meta['extended_obs']
    else:
        obs_shape_meta = shape_meta['obs']

    obs_dict = {}
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')

        if key not in obs_raw:
            continue

        if obs_type == 'rgb':
            imgs = obs_raw[key]  # (T, H, W, C) uint8
            T, H, W, C = imgs.shape
            co, ho, wo = shape
            if H != ho or W != wo:
                resized = np.stack([
                    cv2.resize(imgs[i], (wo, ho)) for i in range(T)
                ])
            else:
                resized = imgs
            # uint8 → float32 [0,1]，THWC → TCHW
            obs_dict[key] = np.moveaxis(resized.astype(np.float32) / 255.0, -1, 1)
        elif obs_type == 'low_dim':
            data = obs_raw[key]  # (T, D)
            obs_dict[key] = data[..., :shape[0]].astype(np.float32)

    return obs_dict


# ──────────────────────────────────────────────
# 相对动作预处理（对齐 real_runner.py 的 pre_process_obs）
# ──────────────────────────────────────────────

def apply_relative_tcp_obs(obs_dict: dict, shape_meta: dict):
    """
    若使用相对动作，将 tcp_pose obs 转换为相对于当前帧的增量（与训练数据预处理一致）。
    同时返回 base_absolute（最后一帧绝对坐标），用于后续动作恢复。
    """
    base_absolute = None
    tcp_keys = [k for k in obs_dict if 'robot_tcp_pose' in k and 'wrt' not in k]
    if tcp_keys:
        # 单臂只有 left_robot_tcp_pose
        base_parts = [obs_dict[k][-1] for k in sorted(tcp_keys)]
        base_absolute = np.concatenate(base_parts, axis=-1)
        for key in tcp_keys:
            obs_dict[key] = absolute_actions_to_relative_actions(
                obs_dict[key], base_absolute_action=obs_dict[key][-1])
    return obs_dict, base_absolute


# ──────────────────────────────────────────────
# 主推理函数
# ──────────────────────────────────────────────

def run_sim_inference(policy,
                      episode: dict,
                      shape_meta: dict,
                      cfg,
                      device: torch.device,
                      episode_len: int = None,
                      ) -> dict:
    """
    模拟 real_runner.py 的慢-快双系统推理流程。

    返回：
        pred_actions:  List[np.ndarray]  每个控制步的模型输出 action（4D）
        gt_actions:    List[np.ndarray]  每个控制步对应的 GT action
        step_times:    List[int]         对应的帧索引
    """
    # ── 从 cfg 读取参数 ────────────────────────────────
    env_runner_cfg = cfg.task.env_runner
    control_fps     = int(env_runner_cfg.control_fps)         # 24
    inference_fps   = int(env_runner_cfg.inference_fps)       # 6
    steps_per_inference = control_fps // inference_fps        # 4
    n_obs_steps          = int(env_runner_cfg.n_obs_steps)    # 2
    obs_ds_ratio         = int(env_runner_cfg.obs_temporal_downsample_ratio)   # 1
    dataset_ds_ratio     = int(env_runner_cfg.dataset_obs_temporal_downsample_ratio)  # 1
    latency_step         = int(env_runner_cfg.latency_step)   # 4
    gripper_latency_step = int(env_runner_cfg.get('gripper_latency_step', latency_step))
    tcp_update_interval  = int(env_runner_cfg.tcp_action_update_interval)  # 16
    gripper_update_interval = int(env_runner_cfg.gripper_action_update_interval)  # 16
    use_relative_action  = bool(env_runner_cfg.use_relative_action)  # True
    use_latent_rnn       = bool(env_runner_cfg.use_latent_action_with_rnn_decoder)  # True
    downsample_ext_obs   = bool(env_runner_cfg.get('downsample_extended_obs', False))

    T_total = episode_len if episode_len is not None else len(episode['action'])
    print(f"[sim] 共 {T_total} 帧，control_fps={control_fps}, inference_fps={inference_fps}")
    print(f"use_relative_action={use_relative_action}, use_latent_rnn={use_latent_rnn}")

    # ── 初始化 EnsembleBuffer ──────────────────────────
    tcp_buf     = EnsembleBuffer(ensemble_mode='new')
    gripper_buf = EnsembleBuffer(ensemble_mode='new')

    pred_actions = []
    gt_actions   = []
    step_times   = []

    # ── 每个控制步 ─────────────────────────────────────
    for ctrl_step in range(T_total):
        # ─ 慢系统：每 steps_per_inference 步做一次 LDP 推理（24/6）
        is_inference_step = (ctrl_step % steps_per_inference == 0)
        if is_inference_step:
            # 1. 构建观测
            obs_raw = build_obs_dict(episode, ctrl_step, n_obs_steps, obs_ds_ratio, shape_meta)
            obs_dict = preprocess_obs_dict(obs_raw, shape_meta, is_extended_obs=False)

            # 2. 相对 TCP obs
            if use_relative_action:
                obs_dict, base_absolute = apply_relative_tcp_obs(obs_dict, shape_meta)
            else:
                base_absolute = None

            # 3. 转 tensor → device
            obs_tensor = dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

            # 4. LDP 推理（慢系统）
            with torch.no_grad():
                action_dict = policy.predict_action(
                    obs_tensor,
                    dataset_obs_temporal_downsample_ratio=dataset_ds_ratio,
                    return_latent_action=use_latent_rnn
                )

            np_action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
            action_all = np_action['action'].squeeze(0)  # (horizon, latent_dim) or (horizon, 4)

            # 5. 附加辅助信息（对齐 real_runner.py）
            if use_latent_rnn:
                if use_relative_action and base_absolute is not None:
                    action_all = np.concatenate([
                        action_all,
                        base_absolute[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                    ], axis=-1)
                # 附加时间步索引
                action_all = np.concatenate([
                    action_all,
                    np.arange(
                        n_obs_steps * dataset_ds_ratio,
                        action_all.shape[0] + n_obs_steps * dataset_ds_ratio
                    )[:, np.newaxis]
                ], axis=-1)
            else:
                if use_relative_action and base_absolute is not None:
                    action_all = relative_actions_to_absolute_actions(action_all, base_absolute)

            # 6. 写入 EnsembleBuffer
            if ctrl_step % tcp_update_interval == 0:
                tcp_action = action_all[latency_step:, ...]
                tcp_buf.add_action(tcp_action, ctrl_step)

            if ctrl_step % gripper_update_interval == 0:
                gripper_action = action_all[gripper_latency_step:, ...]
                gripper_buf.add_action(gripper_action, ctrl_step)

        # ─ 快系统：每控制步从 EnsembleBuffer 取动作 ─
        tcp_step_action     = tcp_buf.get_action()
        gripper_step_action = gripper_buf.get_action()

        if tcp_step_action is None or gripper_step_action is None:
            # EnsembleBuffer 还没数据，跳过
            continue

        if use_latent_rnn:
            # 提取附加的时间步索引
            tcp_obs_step     = int(tcp_step_action[-1])
            gripper_obs_step = int(gripper_step_action[-1])
            tcp_step_action     = tcp_step_action[:-1]
            gripper_step_action = gripper_step_action[:-1]

            # 提取 base_absolute（若有）
            if use_relative_action and base_absolute is not None:
                action_dim = shape_meta['obs'].get('left_robot_tcp_pose', {}).get('shape', [3])[0]
                tcp_base     = tcp_step_action[-action_dim:]
                gripper_base = gripper_step_action[-action_dim:]
                tcp_step_action     = tcp_step_action[:-action_dim]
                gripper_step_action = gripper_step_action[:-action_dim]
            else:
                tcp_base = gripper_base = None

            # 取 extended_obs（触觉 embedding）
            longer_step = max(tcp_obs_step, gripper_obs_step)
            ext_ds_ratio = obs_ds_ratio if downsample_ext_obs else 1
            obs_raw_ext  = build_obs_dict(episode, ctrl_step, longer_step, ext_ds_ratio, shape_meta)
            ext_obs_dict = preprocess_obs_dict(obs_raw_ext, shape_meta, is_extended_obs=True)

            # 相对 TCP obs（extended_obs 也要做，如有 tcp_pose）
            if use_relative_action:
                ext_obs_dict, _ = apply_relative_tcp_obs(ext_obs_dict, shape_meta)

            ext_obs_tensor = dict_apply(ext_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            tcp_latent     = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
            gripper_latent = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

            # AT RNN decoder 解码（快系统核心）
            with torch.no_grad():
                tcp_decoded = policy.predict_from_latent_action(
                    tcp_latent, ext_obs_tensor, tcp_obs_step, dataset_ds_ratio
                )['action'][0].detach().cpu().numpy()
                gripper_decoded = policy.predict_from_latent_action(
                    gripper_latent, ext_obs_tensor, gripper_obs_step, dataset_ds_ratio
                )['action'][0].detach().cpu().numpy()

            # 相对→绝对
            if use_relative_action and tcp_base is not None:
                tcp_decoded     = relative_actions_to_absolute_actions(tcp_decoded,     tcp_base)
                gripper_decoded = relative_actions_to_absolute_actions(gripper_decoded, gripper_base)

            # 取当前步最后一帧（和 real_runner 一致）
            final_tcp     = tcp_decoded[-1]
            final_gripper = gripper_decoded[-1]

            # action_dim 决定 tcp 部分和 gripper 部分的分界
            # action shape = (4,): (x, y, z, gripper_width)
            action_total_dim = shape_meta['action']['shape'][0]  # 4
            tcp_dim = action_total_dim - 1   # 3  (x,y,z)
            predicted_action = np.concatenate([final_tcp[:tcp_dim], final_gripper[tcp_dim:]])

        else:
            # 普通 DP 模式
            predicted_action = np.concatenate([tcp_step_action, gripper_step_action])

        # GT action
        gt_action = episode['action'][ctrl_step]

        pred_actions.append(predicted_action)
        gt_actions.append(gt_action)
        step_times.append(ctrl_step)

    print(f"[sim] 推理完成，共输出 {len(pred_actions)} 步 action")
    return {
        'pred': np.array(pred_actions),   # (N, 4)
        'gt':   np.array(gt_actions),     # (N, 4)
        'steps': np.array(step_times),
    }


# ──────────────────────────────────────────────
# 绘图对比
# ──────────────────────────────────────────────

def plot_comparison(result: dict, output_path: str, episode_idx: int):
    pred  = result['pred']   # (N, 4)
    gt    = result['gt']     # (N, 4)
    steps = result['steps']

    action_names = ['Δx (m)', 'Δy (m)', 'Δz (m)', 'gripper_width']
    n_dims = pred.shape[-1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    fig.suptitle(f'Model Action vs GT Action  (Episode {episode_idx})', fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes, action_names[:n_dims])):
        ax.plot(steps, gt[:, i],   label='GT action',        color='steelblue',   linewidth=1.2, alpha=0.85)
        ax.plot(steps, pred[:, i], label='Predicted action',  color='darkorange',  linewidth=1.0, linestyle='--', alpha=0.9)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 计算 MAE
        mae = np.mean(np.abs(pred[:, i] - gt[:, i]))
        ax.set_title(f'{name}  —  MAE = {mae:.5f}', fontsize=9)

    axes[-1].set_xlabel('Frame index', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[plot] 对比图已保存到：{output_path}")
    plt.close(fig)

    # 打印统计信息
    print("\n[stats] 各维度 MAE：")
    for i, name in enumerate(action_names[:n_dims]):
        mae = np.mean(np.abs(pred[:, i] - gt[:, i]))
        print(f"  {name:<20s}: MAE = {mae:.6f}")


# ──────────────────────────────────────────────
# 主入口（Hydra 配置加载）
# ──────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'reactive_diffusion_policy', 'config')),
    config_name="train_latent_diffusion_unet_real_image_workspace"
)
def main(cfg: DictConfig):
    # ── 读取自定义参数 ─────────────────────────────────
    episode_idx  = int(cfg.get('episode_idx',  0))
    n_inf_steps  = cfg.get('n_inference_steps', None)  # None = 整个 episode
    output_fig   = str(cfg.get('output_fig', 'eval_sim_result.png'))

    zarr_path    = str(cfg.task.dataset_path)
    shape_meta   = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)

    print(f"[cfg] zarr_path    = {zarr_path}")
    print(f"[cfg] episode_idx  = {episode_idx}")
    print(f"[cfg] output_fig   = {output_fig}")
    print(f"[cfg] shape_meta obs keys: {list(shape_meta['obs'].keys())}")

    # ── 加载 checkpoint ────────────────────────────────
    ckpt_path = cfg.ckpt_path
    print(f"[ckpt] 加载 checkpoint：{ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ── 初始化 policy ──────────────────────────────────
    assert 'diffusion' in cfg.name and 'latent' in cfg.name, \
        "eval_sim.py 目前仅支持 LDP（latent diffusion）模式"

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.at.set_normalizer(policy.normalizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    policy.num_inference_steps = 8  # DDIM steps

    print(f"[policy] device = {device}")
    print(f"[policy] DDIM steps = {policy.num_inference_steps}")

    # ── 读取 episode ───────────────────────────────────
    episode, ep_start, ep_end = load_episode_from_zarr(zarr_path, episode_idx)

    # 若只测试前 N 步
    if n_inf_steps is not None:
        n_inf_steps = int(n_inf_steps)
        for key in episode:
            episode[key] = episode[key][:n_inf_steps]
        ep_length = n_inf_steps
        print(f"[sim] 仅测试前 {n_inf_steps} 帧")
    else:
        ep_length = ep_end - ep_start

    # ── 运行推理 ───────────────────────────────────────
    result = run_sim_inference(
        policy=policy,
        episode=episode,
        shape_meta=shape_meta,
        cfg=cfg,
        device=device,
        episode_len=ep_length,
    )

    # ── 绘图 ───────────────────────────────────────────
    plot_comparison(result, output_fig, episode_idx)

    # ── 保存数值结果 ───────────────────────────────────
    npy_path = output_fig.replace('.png', '_data.npz')
    np.savez(npy_path,
             pred=result['pred'],
             gt=result['gt'],
             steps=result['steps'])
    print(f"[save] 数值结果已保存到：{npy_path}")


if __name__ == '__main__':
    main()
