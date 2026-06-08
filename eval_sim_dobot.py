"""
eval_sim_dobot.py
=================
离线推理验证脚本（Dobot 数据集版本）：用 zarr 数据集中的真实观测作为模型输入，
模拟慢-快双系统的完整推理链路，将模型输出的 action 与 GT action 进行对比。

与 eval_sim_umi.py 的差异：
    - 使用 task=dobot_rdp_image_tactile_emb_ldp_24fps 配置
    - Dobot zarr 数据集含有 external_img 字段（俯视相机），
      shape_meta['obs'] 中已配置该字段为 rgb 类型，会被 LDP 的 MultiImageObsEncoder 编码使用。
    - left_robot_tcp_pose 在 zarr 中为 9 维（6D 旋转），shape_meta 中配置为 3 维（仅 xyz），
      preprocess_obs_dict 会自动截取前 3 维，与训练一致。
    - at_load_dir 接受 .ckpt 文件路径（直接指向具体 checkpoint 文件）。

使用方法：
----------
CUDA_VISIBLE_DEVICES=0 python eval_sim_dobot.py \
    --config-name train_latent_diffusion_unet_real_image_workspace \
    task=dobot_rdp_image_tactile_emb_ldp_24fps \
    task.dataset_path=/home/zzw/project/reactive_diffusion_policy/data/hf_dataset/dataset_mini/dobot_peg_in_hole_zarr_test \
    +ckpt_path="data/outputs/2026.05.28_dobot_ldp/11.47.29_train_latent_diffusion_unet_image_dobot_rdp_image_tactile_emb_ldp_24fps_0528114721/checkpoints/epoch-0400-train_loss-0.002.ckpt" \
    +at_load_dir="data/outputs/2026.05.27_dobot_at/18.46.26_train_vae_dobot_rdp_image_tactile_emb_at_24fps_0527184624/checkpoints/epoch-0580-train_loss-0.010161.ckpt" \
    hydra.run.dir="data/outputs/eval_sim" \
    +episode_idx=0 \
    +output_fig="eval_sim_dobot_0.png"

可选参数（通过 Hydra 覆盖）：
    +episode_idx=0          选择第几个 episode（默认 0，共 50 个）
    +n_inference_steps=30   只测试前 N 步（默认 None = 整个 episode）
    +output_fig=eval_sim_dobot_0.png  输出图片路径
"""

# %%
import pathlib
import os

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
    ends = z['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])

    assert episode_idx < len(ends), \
        f"episode_idx={episode_idx} 超出范围（共 {len(ends)} 个 episode）"

    s, e = int(starts[episode_idx]), int(ends[episode_idx])
    print(f"[zarr] Episode {episode_idx}: frames [{s}, {e})，共 {e - s} 帧")
    print(f"[zarr] 数据集字段: {sorted(z['data'].keys())}")

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
    给定当前时刻 t，从离线 episode 中取出最近 n_obs_steps 帧（带时间降采样），
    模拟 real_env.get_obs() 的行为。
    """
    window = n_obs_steps * obs_temporal_downsample_ratio
    t_end   = min(t + 1, len(episode['action']))
    t_start = max(0, t_end - window)

    obs_raw = {}
    for key in episode:
        seg = episode[key][t_start:t_end]
        # 倒序隔帧取样，再翻转回正序
        sampled = seg[::-obs_temporal_downsample_ratio][::-1]  # (<=n_obs_steps, ...)
        # 开头帧不足时用最早帧重复填充
        if sampled.shape[0] < n_obs_steps:
            pad = np.repeat(sampled[:1], n_obs_steps - sampled.shape[0], axis=0)
            sampled = np.concatenate([pad, sampled], axis=0)
        obs_raw[key] = sampled  # (n_obs_steps, ...)

    return obs_raw


def preprocess_obs_dict(obs_raw: dict,
                        shape_meta: dict,
                        is_extended_obs: bool = False) -> dict:
    """
    按照 shape_meta 对观测做维度对齐和归一化。
    只处理 shape_meta 中声明的字段，其余字段（如 external_img）自动忽略。

    注意（Dobot 特有）：
        - external_img（俯视相机）在 shape_meta['obs'] 中声明为 rgb 类型，会被 LDP 的 MultiImageObsEncoder 编码使用。
        - left_robot_tcp_pose 在 zarr 中为 9 维，shape_meta 配置为 3 维，
          此处截取前 3 维，与训练一致。
    """
    obs_shape_meta = shape_meta['extended_obs'] if is_extended_obs else shape_meta['obs']

    obs_dict = {}
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape    = attr.get('shape')

        if key not in obs_raw:
            print(f"[warn] shape_meta 中声明的字段 '{key}' 在 zarr 中不存在，跳过。")
            continue

        if obs_type == 'rgb':
            imgs = obs_raw[key]  # (T, H, W, C) uint8
            T, H, W, C = imgs.shape
            co, ho, wo = shape
            if H != ho or W != wo:
                resized = np.stack([cv2.resize(imgs[i], (wo, ho)) for i in range(T)])
            else:
                resized = imgs
            # uint8 → float32 [0,1]，THWC → TCHW
            obs_dict[key] = np.moveaxis(resized.astype(np.float32) / 255.0, -1, 1)

        elif obs_type == 'low_dim':
            data = obs_raw[key]  # (T, D_zarr)，D_zarr 可能 > shape[0]
            obs_dict[key] = data[..., :shape[0]].astype(np.float32)

    return obs_dict


# ──────────────────────────────────────────────
# 相对动作预处理
# ──────────────────────────────────────────────

def apply_relative_tcp_obs(obs_dict: dict, shape_meta: dict):
    """
    将 tcp_pose obs 转换为相对于当前帧的增量（与训练数据预处理一致）。
    返回修改后的 obs_dict 和最后一帧的绝对坐标（用于后续动作恢复）。
    """
    base_absolute = None
    tcp_keys = [k for k in obs_dict if 'robot_tcp_pose' in k and 'wrt' not in k]
    if tcp_keys:
        base_parts = [obs_dict[k][-1] for k in sorted(tcp_keys)]
        base_absolute = np.concatenate(base_parts, axis=-1)
        for key in tcp_keys:
            obs_dict[key] = absolute_actions_to_relative_actions(
                obs_dict[key], base_absolute_action=obs_dict[key][-1])
    return obs_dict, base_absolute


# ──────────────────────────────────────────────
# 主推理函数（模拟慢-快双系统）
# ──────────────────────────────────────────────

def run_sim_inference(policy,
                      episode: dict,
                      shape_meta: dict,
                      cfg,
                      device: torch.device,
                      episode_len: int = None) -> dict:
    """
    模拟 real_runner.py 的慢-快双系统推理流程。

    返回：
        pred:  np.ndarray (N, 4)  每个控制步的模型输出 action
        gt:    np.ndarray (N, 4)  GT action
        steps: np.ndarray (N,)    对应帧索引
    """
    env_runner_cfg = cfg.task.env_runner
    control_fps          = int(env_runner_cfg.control_fps)            # 24
    inference_fps        = int(env_runner_cfg.inference_fps)          # 6
    steps_per_inference  = control_fps // inference_fps               # 4
    n_obs_steps          = int(env_runner_cfg.n_obs_steps)            # 2
    obs_ds_ratio         = int(env_runner_cfg.obs_temporal_downsample_ratio)
    dataset_ds_ratio     = int(env_runner_cfg.dataset_obs_temporal_downsample_ratio)
    latency_step         = int(env_runner_cfg.latency_step)
    gripper_latency_step = int(env_runner_cfg.get('gripper_latency_step', latency_step))
    tcp_update_interval  = int(env_runner_cfg.tcp_action_update_interval)
    gripper_update_interval = int(env_runner_cfg.gripper_action_update_interval)
    use_relative_action  = bool(env_runner_cfg.use_relative_action)
    use_latent_rnn       = bool(env_runner_cfg.use_latent_action_with_rnn_decoder)
    downsample_ext_obs   = bool(env_runner_cfg.get('downsample_extended_obs', False))

    T_total = episode_len if episode_len is not None else len(episode['action'])
    print(f"\n[sim] 共 {T_total} 帧  control_fps={control_fps}  inference_fps={inference_fps}")
    print(f"[sim] use_relative_action={use_relative_action}  use_latent_rnn={use_latent_rnn}")
    print(f"[sim] obs_ds_ratio={obs_ds_ratio}  dataset_ds_ratio={dataset_ds_ratio}")
    print(f"[sim] shape_meta obs keys: {list(shape_meta['obs'].keys())}")

    tcp_buf     = EnsembleBuffer(ensemble_mode='new')
    gripper_buf = EnsembleBuffer(ensemble_mode='new')

    pred_actions = []
    gt_actions   = []
    step_times   = []
    base_absolute = None  # 最近一次慢系统的绝对坐标基准

    for ctrl_step in range(T_total):
        # ── 慢系统：每 steps_per_inference 步推理一次 LDP ──
        if ctrl_step % steps_per_inference == 0:
            # 1. 构建观测（只用 shape_meta['obs'] 声明的字段）
            obs_raw  = build_obs_dict(episode, ctrl_step, n_obs_steps, obs_ds_ratio, shape_meta)
            obs_dict = preprocess_obs_dict(obs_raw, shape_meta, is_extended_obs=False)

            # 2. 相对 TCP obs
            if use_relative_action:
                obs_dict, base_absolute = apply_relative_tcp_obs(obs_dict, shape_meta)

            # 3. 转 tensor → GPU
            obs_tensor = dict_apply(
                obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

            # 4. LDP 推理
            with torch.no_grad():
                action_dict = policy.predict_action(
                    obs_tensor,
                    dataset_obs_temporal_downsample_ratio=dataset_ds_ratio,
                    return_latent_action=use_latent_rnn
                )

            np_action  = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
            action_all = np_action['action'].squeeze(0)  # (horizon, latent_dim or 4)

            # 5. 附加辅助信息（对齐 real_runner.py）
            if use_latent_rnn:
                if use_relative_action and base_absolute is not None:
                    action_all = np.concatenate([
                        action_all,
                        np.tile(base_absolute[np.newaxis, :], (action_all.shape[0], 1))
                    ], axis=-1)
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
                tcp_buf.add_action(action_all[latency_step:, ...], ctrl_step)
            if ctrl_step % gripper_update_interval == 0:
                gripper_buf.add_action(action_all[gripper_latency_step:, ...], ctrl_step)

        # ── 快系统：每控制步解码一次 action ──
        tcp_step_action     = tcp_buf.get_action()
        gripper_step_action = gripper_buf.get_action()

        if tcp_step_action is None or gripper_step_action is None:
            continue  # EnsembleBuffer 还没数据

        if use_latent_rnn:
            # 提取时间步索引
            tcp_obs_step     = int(tcp_step_action[-1])
            gripper_obs_step = int(gripper_step_action[-1])
            tcp_step_action     = tcp_step_action[:-1]
            gripper_step_action = gripper_step_action[:-1]

            # 提取 base_absolute（若使用相对动作）
            if use_relative_action and base_absolute is not None:
                action_dim = shape_meta['obs']['left_robot_tcp_pose']['shape'][0]  # 3
                tcp_base     = tcp_step_action[-action_dim:]
                gripper_base = gripper_step_action[-action_dim:]
                tcp_step_action     = tcp_step_action[:-action_dim]
                gripper_step_action = gripper_step_action[:-action_dim]
            else:
                tcp_base = gripper_base = None

            # 取 extended_obs（触觉 embedding）
            longer_step  = max(tcp_obs_step, gripper_obs_step)
            ext_ds_ratio = obs_ds_ratio if downsample_ext_obs else 1
            obs_raw_ext  = build_obs_dict(episode, ctrl_step, longer_step, ext_ds_ratio, shape_meta)
            ext_obs_dict = preprocess_obs_dict(obs_raw_ext, shape_meta, is_extended_obs=True)

            if use_relative_action:
                ext_obs_dict, _ = apply_relative_tcp_obs(ext_obs_dict, shape_meta)

            ext_obs_tensor = dict_apply(
                ext_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            tcp_latent     = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
            gripper_latent = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

            # AT RNN decoder 解码
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

            # action shape=(4,): (x, y, z, gripper_width)，tcp 部分 = 前 3 维
            action_total_dim = shape_meta['action']['shape'][0]  # 4
            tcp_dim          = action_total_dim - 1              # 3
            predicted_action = np.concatenate([
                tcp_decoded[-1][:tcp_dim],
                gripper_decoded[-1][tcp_dim:]
            ])
        else:
            predicted_action = np.concatenate([tcp_step_action, gripper_step_action])

        gt_action = episode['action'][ctrl_step]
        pred_actions.append(predicted_action)
        gt_actions.append(gt_action)
        step_times.append(ctrl_step)

    print(f"[sim] 推理完成，共输出 {len(pred_actions)} 步 action")
    return {
        'pred':  np.array(pred_actions),
        'gt':    np.array(gt_actions),
        'steps': np.array(step_times),
    }


# ──────────────────────────────────────────────
# 绘图对比
# ──────────────────────────────────────────────

def plot_comparison(result: dict, output_path: str, episode_idx: int):
    pred  = result['pred']   # (N, 4)
    gt    = result['gt']     # (N, 4)
    steps = result['steps']

    action_names = ['x (m)', 'y (m)', 'z (m)', 'gripper_width']
    n_dims = pred.shape[-1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    fig.suptitle(f'Dobot — Model Action vs GT Action  (Episode {episode_idx})',
                 fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes, action_names[:n_dims])):
        ax.plot(steps, gt[:, i],   label='GT action',       color='steelblue',  linewidth=1.2, alpha=0.85)
        ax.plot(steps, pred[:, i], label='Predicted action', color='darkorange', linewidth=1.0,
                linestyle='--', alpha=0.9)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        mae = np.mean(np.abs(pred[:, i] - gt[:, i]))
        ax.set_title(f'{name}  —  MAE = {mae:.5f}', fontsize=9)

    axes[-1].set_xlabel('Frame index', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[plot] 对比图已保存：{output_path}")
    plt.close(fig)

    print("\n[stats] 各维度 MAE：")
    for i, name in enumerate(action_names[:n_dims]):
        mae = np.mean(np.abs(pred[:, i] - gt[:, i]))
        print(f"  {name:<20s}: MAE = {mae:.6f}")


# ──────────────────────────────────────────────
# 主入口（Hydra）
# ──────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'reactive_diffusion_policy', 'config')),
    config_name="train_latent_diffusion_unet_real_image_workspace"
)
def main(cfg: DictConfig):
    episode_idx = int(cfg.get('episode_idx', 0))
    n_inf_steps = cfg.get('n_inference_steps', None)
    output_fig  = str(cfg.get('output_fig', 'eval_sim_result.png'))
    zarr_path   = str(cfg.task.dataset_path)
    shape_meta  = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)

    print(f"[cfg] zarr_path   = {zarr_path}")
    print(f"[cfg] episode_idx = {episode_idx}")
    print(f"[cfg] output_fig  = {output_fig}")
    print(f"[cfg] shape_meta obs keys: {list(shape_meta['obs'].keys())}")

    # ── 加载 LDP checkpoint ────────────────────────────
    ckpt_path = cfg.ckpt_path
    print(f"\n[ckpt] 加载 LDP checkpoint：{ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    assert 'diffusion' in cfg.name and 'latent' in cfg.name, \
        "eval_sim_dobot.py 仅支持 LDP（latent diffusion）模式"

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # AT 子模块的 normalizer 初始化（LDP 内嵌 AT）
    policy.at.set_normalizer(policy.normalizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    policy.num_inference_steps = 8
    print(f"[policy] device={device}  DDIM steps={policy.num_inference_steps}")

    # ── 读取 episode ──────────────────────────────────
    episode, ep_start, ep_end = load_episode_from_zarr(zarr_path, episode_idx)

    if n_inf_steps is not None:
        n_inf_steps = int(n_inf_steps)
        for key in episode:
            episode[key] = episode[key][:n_inf_steps]
        ep_length = n_inf_steps
        print(f"[sim] 仅测试前 {n_inf_steps} 帧")
    else:
        ep_length = ep_end - ep_start

    # ── 运行推理 ──────────────────────────────────────
    result = run_sim_inference(
        policy=policy,
        episode=episode,
        shape_meta=shape_meta,
        cfg=cfg,
        device=device,
        episode_len=ep_length,
    )

    # ── 绘图 & 保存 ───────────────────────────────────
    plot_comparison(result, output_fig, episode_idx)

    npy_path = output_fig.replace('.png', '_data.npz')
    np.savez(npy_path, pred=result['pred'], gt=result['gt'], steps=result['steps'])
    print(f"[save] 数值结果已保存：{npy_path}")


if __name__ == '__main__':
    main()
