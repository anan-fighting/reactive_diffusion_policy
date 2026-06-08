#!/usr/bin/env python3
# coding=utf-8
"""
触觉标记点提取 + PCA 降维脚本（Dobot 版本）
==========================================
针对 Dobot 采集的数据集（如 /home/zzw/teleop_data/Dobot_peg_in_hole），
从 tactile_left.mp4 / tactile_right.mp4 中提取
2D 标记点坐标，并对位移场做 PCA 降维，生成 RDP 训练所需的三个数组：

    initial_marker     [T, N*M, 2]  第 0 帧标记点坐标（归一化到 [0,1]）
    marker_offset      [T, N*M, 2]  每帧相对第 0 帧的位移（归一化）
    marker_offset_emb  [T, 15]      PCA 降维后的触觉嵌入

与 extract_umi_tactile_markers_and_pca.py 的主要差异：
    1. 数据集无 hand_subdir 子目录，触觉视频直接位于 episode_X/ 下
    2. 视频文件名为 tactile_{side}.mp4
    3. 位姿文件名为 robot_tcp_pose.npy（原数据集为 vio_pose.npy）

工作流程：
    Step 1  遍历所有 episode，用 pyvitaisdk 提取 marker_origin / marker_offset，
            保存为 initial_marker_{side}.npy 和 marker_offset_{side}.npy（归一化）
    Step 2  汇总所有 episode 的 marker_offset，训练 PCA（n_components=15），
            保存变换矩阵 W 和均值 mean 到 data/PCA_Dobot/
    Step 3  对每个 episode 应用 PCA，保存 marker_offset_emb_{side}.npy

用法：
    # 完整流程（推荐首次运行）
    python extract_dobot_tactile_markers_and_pca.py

    # 只做 Step1（提取标记点，跳过 PCA）
    python extract_dobot_tactile_markers_and_pca.py --step 1

    # 只做 Step2+3（已有 .npy，重新训练 PCA 并生成嵌入）
    python extract_dobot_tactile_markers_and_pca.py --step 23

    # 只处理特定 side
    python extract_dobot_tactile_markers_and_pca.py --sides left

    # 单 episode 快速调试
    python extract_dobot_tactile_markers_and_pca.py --episode episode_0 --step 1

配置：
    修改脚本顶部 CONFIG 字典，无需命令行参数。
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────── CONFIG ──────────────────────────────────────
CONFIG = {
    # 数据集根目录（Dobot 采集）
    "episodes_root": "/home/zzw/teleop_data/Dobot_peg_in_hole_0604",
    # "episodes_root": "/home/zzw/teleop_data/test",

    # episode 目录前缀
    "episode_prefix": "episode_",

    # 处理哪些 side
    # 视频文件名规则：tactile_{side}.mp4
    "sides": ["left", "right"],

    # ViTai 传感器型号
    "sensor_type": "GF225",

    # 使用视频第几帧做传感器校准（0 = 第一帧）
    "bg_frame_index": 0,

    # PCA 维数
    "n_pca_components": 15,

    # PCA 矩阵保存目录（相对于 reactive_diffusion_policy 项目根）
    "pca_save_dir": "data/PCA_Dobot_peg_in_hole_0604",
    # "pca_save_dir": "data/000test",

    # 每处理多少帧打印一次进度
    "log_interval": 100,
}
# ─────────────────────────────────────────────────────────────────────────────


def import_sdk():
    try:
        from pyvitaisdk import VTSensor, VTSDataType, VTSError, VTSensorType
        return VTSensor, VTSDataType, VTSError, VTSensorType
    except ImportError:
        print("[ERROR] pyvitaisdk 未安装，请先安装 ViTai SDK：")
        print("  pip install pyvitaisdk*.whl")
        sys.exit(1)


def get_video_path(episode_dir: Path, side: str) -> Path:
    """
    返回 Dobot 数据集中触觉视频的路径。
    文件名格式：tactile_{side}.mp4
    """
    return episode_dir / f"tactile_{side}.mp4"


# ─────────────────────────── Step 1：提取标记点 ───────────────────────────────

def extract_markers_for_video(video_path: str, bg_frame_index: int = 0,
                               sensor_type_str: str = "GF225",
                               log_interval: int = 100):
    """
    从单个触觉视频提取每帧的 marker_origin 和 marker_offset。

    Returns
    -------
    initial_marker  : np.ndarray  shape (T, N*M, 2)  归一化坐标 [0,1]
    marker_offset   : np.ndarray  shape (T, N*M, 2)  归一化位移 [0,1]
    """
    VTSensor, VTSDataType, VTSError, VTSensorType = import_sdk()

    sensor_type_map = {"GF225": VTSensorType.GF225}
    if sensor_type_str not in sensor_type_map:
        raise ValueError(f"不支持的传感器型号: {sensor_type_str}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  视频: {Path(video_path).name}  分辨率: {W}x{H}  帧数: {total_frames}")

    # 读取所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"视频无帧: {video_path}")

    # 背景校准帧
    bg_idx = min(bg_frame_index, len(frames) - 1)
    bg_image = frames[bg_idx]

    # 初始化传感器
    vtsensor = VTSensor(config=None, sensor_type=sensor_type_map[sensor_type_str])
    vtsensor.calibrate(calib_image=bg_image)

    # 确认标记点网格形状（从第一帧推断）
    first_data = vtsensor.collect_sensor_data(
        VTSDataType.MARKER_ORIGIN_VECTOR,
        VTSDataType.MARKER_OFFSET_VECTOR,
        frame=frames[0]
    )
    grid_shape = first_data[VTSDataType.MARKER_ORIGIN_VECTOR].shape[:2]  # (N, M)
    n_markers = grid_shape[0] * grid_shape[1]
    print(f"  标记点网格: {grid_shape[0]}×{grid_shape[1]} = {n_markers} 个标记点")

    norm = np.array([W, H], dtype=np.float32)

    initial_marker_list = []
    marker_offset_list = []
    prev_offset = np.zeros((n_markers, 2), dtype=np.float32)  # 容错用

    for i, frame in enumerate(frames):
        try:
            data = vtsensor.collect_sensor_data(
                VTSDataType.MARKER_ORIGIN_VECTOR,
                VTSDataType.MARKER_OFFSET_VECTOR,
                frame=frame
            )
            origin = data[VTSDataType.MARKER_ORIGIN_VECTOR]   # (N, M, 2) 像素坐标
            offset = data[VTSDataType.MARKER_OFFSET_VECTOR]   # (N, M, 2) 像素位移

            # 展平为 (n_markers, 2) 并归一化
            origin_flat = origin.reshape(n_markers, 2).astype(np.float32) / norm
            offset_flat = offset.reshape(n_markers, 2).astype(np.float32) / norm
            prev_offset = offset_flat

        except Exception as e:
            print(f"  [WARN] 第 {i} 帧处理失败: {e}，使用上一帧填充")
            # origin 用第一帧
            origin_flat = initial_marker_list[0] if initial_marker_list else np.zeros((n_markers, 2), dtype=np.float32)
            offset_flat = prev_offset

        initial_marker_list.append(origin_flat)
        marker_offset_list.append(offset_flat)

        if (i + 1) % log_interval == 0 or (i + 1) == len(frames):
            print(f"  进度: {i + 1}/{len(frames)}")

    vtsensor.release()

    initial_marker = np.stack(initial_marker_list, axis=0)  # (T, n_markers, 2)
    marker_offset = np.stack(marker_offset_list, axis=0)    # (T, n_markers, 2)
    return initial_marker, marker_offset


def step1_extract_all(episodes_root: str, episodes: list, sides: list,
                       bg_frame_index: int, sensor_type_str: str,
                       log_interval: int):
    """
    遍历所有 episode，提取并保存 initial_marker_{side}.npy 和 marker_offset_{side}.npy。

    Dobot 数据集文件直接位于 episode_X/ 目录下，无 hand_subdir 子目录。
    视频文件名：tactile_{side}.mp4
    输出文件名：initial_marker_{side}.npy / marker_offset_{side}.npy
    """
    print("\n" + "="*60)
    print("Step 1：提取标记点坐标")
    print("="*60)

    for ep_name in episodes:
        ep_dir = Path(episodes_root) / ep_name
        for side in sides:
            video_path = get_video_path(ep_dir, side)
            if not video_path.exists():
                print(f"[SKIP] {video_path} 不存在")
                continue

            out_initial = ep_dir / f"initial_marker_{side}.npy"
            out_offset = ep_dir / f"marker_offset_{side}.npy"

            if out_initial.exists() and out_offset.exists():
                print(f"[SKIP] {ep_name}/{side} 已存在，跳过（删除文件可重新生成）")
                continue

            print(f"\n[处理] {ep_name} / {side}")
            try:
                initial_marker, marker_offset = extract_markers_for_video(
                    str(video_path),
                    bg_frame_index=bg_frame_index,
                    sensor_type_str=sensor_type_str,
                    log_interval=log_interval,
                )
                np.save(str(out_initial), initial_marker)
                np.save(str(out_offset), marker_offset)
                print(f"  已保存: {out_initial.name}  shape={initial_marker.shape}")
                print(f"  已保存: {out_offset.name}   shape={marker_offset.shape}")
            except Exception as e:
                print(f"  [ERROR] {ep_name}/{side}: {e}")


# ─────────────────────────── Step 2：训练 PCA ─────────────────────────────────

def step2_train_pca(episodes_root: str, episodes: list, sides: list,
                     n_components: int, pca_save_dir: str):
    """
    汇总所有 episode 的 marker_offset，训练 PCA，保存变换矩阵。

    保存文件：
        {pca_save_dir}/pca_transform_matrix_{side}.npy  shape (D, n_components)
        {pca_save_dir}/pca_mean_matrix_{side}.npy       shape (D,)
    """
    print("\n" + "="*60)
    print("Step 2：训练 PCA")
    print("="*60)

    from sklearn.decomposition import PCA

    os.makedirs(pca_save_dir, exist_ok=True)

    for side in sides:
        print(f"\n[Side: {side}]")
        all_offsets = []

        for ep_name in episodes:
            offset_path = Path(episodes_root) / ep_name / f"marker_offset_{side}.npy"
            if not offset_path.exists():
                continue
            offset = np.load(str(offset_path))  # (T, n_markers, 2)
            T, n_markers, _ = offset.shape
            all_offsets.append(offset.reshape(T, n_markers * 2))  # (T, D)

        if not all_offsets:
            print(f"  [ERROR] 未找到任何 marker_offset_{side}.npy，请先运行 Step 1")
            continue

        X = np.concatenate(all_offsets, axis=0)  # (N_total, D)
        D = X.shape[1]
        print(f"  汇总数据: {X.shape}  ({len(all_offsets)} 个 episode，D={D})")

        # 训练 PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)

        explained = pca.explained_variance_ratio_.cumsum()[-1]
        print(f"  前 {n_components} 个主成分累计方差解释率: {explained*100:.2f}%")
        print(f"  各主成分方差贡献: {pca.explained_variance_ratio_[:5].round(4)} ...")

        W = pca.components_.T    # (D, n_components)
        mean = pca.mean_         # (D,)

        w_path = os.path.join(pca_save_dir, f"pca_transform_matrix_{side}.npy")
        m_path = os.path.join(pca_save_dir, f"pca_mean_matrix_{side}.npy")
        np.save(w_path, W)
        np.save(m_path, mean)
        print(f"  已保存: {w_path}  shape={W.shape}")
        print(f"  已保存: {m_path}  shape={mean.shape}")


# ─────────────────────────── Step 3：应用 PCA 生成嵌入 ───────────────────────

def step3_apply_pca(episodes_root: str, episodes: list, sides: list,
                     n_components: int, pca_save_dir: str):
    """
    对每个 episode 的 marker_offset 应用已训练的 PCA，保存 marker_offset_emb_{side}.npy。
    输出文件直接保存在 episode_X/ 目录下。
    """
    print("\n" + "="*60)
    print("Step 3：应用 PCA，生成嵌入")
    print("="*60)

    for side in sides:
        w_path = os.path.join(pca_save_dir, f"pca_transform_matrix_{side}.npy")
        m_path = os.path.join(pca_save_dir, f"pca_mean_matrix_{side}.npy")

        if not os.path.exists(w_path) or not os.path.exists(m_path):
            print(f"[ERROR] 找不到 PCA 矩阵: {w_path}，请先运行 Step 2")
            continue

        W = np.load(w_path)     # (D, n_components)
        mean = np.load(m_path)  # (D,)
        print(f"\n[Side: {side}]  W={W.shape}  mean={mean.shape}")

        for ep_name in episodes:
            ep_dir = Path(episodes_root) / ep_name
            offset_path = ep_dir / f"marker_offset_{side}.npy"
            out_path = ep_dir / f"marker_offset_emb_{side}.npy"

            if not offset_path.exists():
                continue

            offset = np.load(str(offset_path))   # (T, n_markers, 2)
            T, n_markers, _ = offset.shape
            X = offset.reshape(T, n_markers * 2).astype(np.float32)  # (T, D)

            # PCA 投影: (T, D) -> (T, n_components)
            X_centered = X - mean.astype(np.float32)
            emb = X_centered @ W.astype(np.float32)  # (T, n_components)

            np.save(str(out_path), emb)

        # 汇报一次
        sample_ep = next((e for e in episodes
                          if (Path(episodes_root)/e/f"marker_offset_emb_{side}.npy").exists()), None)
        if sample_ep:
            sample = np.load(str(Path(episodes_root)/sample_ep/f"marker_offset_emb_{side}.npy"))
            print(f"  示例 ({sample_ep}): marker_offset_emb_{side}.npy  shape={sample.shape}  dtype={sample.dtype}")

        total = sum(1 for e in episodes
                    if (Path(episodes_root)/e/f"marker_offset_emb_{side}.npy").exists())
        print(f"  共生成 {total} 个 episode 的 marker_offset_emb_{side}.npy")


# ─────────────────────────── 验证工具 ─────────────────────────────────────────

def verify(episodes_root: str, episodes: list, sides: list):
    """快速验证输出文件是否完整。"""
    print("\n" + "="*60)
    print("验证输出")
    print("="*60)
    for side in sides:
        for fname in [f"initial_marker_{side}.npy",
                      f"marker_offset_{side}.npy",
                      f"marker_offset_emb_{side}.npy"]:
            found = sum(
                1 for e in episodes
                if (Path(episodes_root)/e/fname).exists()
            )
            print(f"  {fname:45s}  {found}/{len(episodes)} episodes")


# ─────────────────────────── 入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dobot 数据集触觉标记点提取 + PCA 降维",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python extract_dobot_tactile_markers_and_pca.py              # 完整流程
  python extract_dobot_tactile_markers_and_pca.py --step 1     # 只提取标记点
  python extract_dobot_tactile_markers_and_pca.py --step 23    # 只训练+应用 PCA
  python extract_dobot_tactile_markers_and_pca.py --sides left # 只处理左侧
  python extract_dobot_tactile_markers_and_pca.py --episode episode_0 --step 1
        """
    )
    parser.add_argument("--step", type=str, default="123",
                        help="执行步骤：'1'=提取, '2'=训练PCA, '3'=应用PCA, '123'=全部（默认）")
    parser.add_argument("--sides", type=str, default=None,
                        help="处理哪些 side，逗号分隔，例如 'left' 或 'left,right'")
    parser.add_argument("--episode", type=str, default=None,
                        help="只处理单个 episode，例如 'episode_0'（调试用）")
    args = parser.parse_args()

    cfg = CONFIG.copy()

    # 命令行覆盖 sides
    if args.sides:
        cfg["sides"] = [s.strip() for s in args.sides.split(",")]

    # 收集 episode 列表
    root = Path(cfg["episodes_root"])
    if not root.exists():
        print(f"[ERROR] episodes_root 不存在: {root}")
        sys.exit(1)

    if args.episode:
        episodes = [args.episode]
    else:
        episodes = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name.startswith(cfg["episode_prefix"])
        )

    print(f"[INFO] episodes_root : {root}")
    print(f"[INFO] 共 {len(episodes)} 个 episode，side: {cfg['sides']}")
    print(f"[INFO] 执行步骤: {args.step}")
    print(f"[INFO] 视频文件名格式: tactile_{{side}}.mp4")

    pca_save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        cfg["pca_save_dir"]
    )

    steps = set(args.step)

    if "1" in steps:
        step1_extract_all(
            episodes_root=str(root),
            episodes=episodes,
            sides=cfg["sides"],
            bg_frame_index=cfg["bg_frame_index"],
            sensor_type_str=cfg["sensor_type"],
            log_interval=cfg["log_interval"],
        )

    if "2" in steps:
        step2_train_pca(
            episodes_root=str(root),
            episodes=episodes,
            sides=cfg["sides"],
            n_components=cfg["n_pca_components"],
            pca_save_dir=pca_save_dir,
        )

    if "3" in steps:
        step3_apply_pca(
            episodes_root=str(root),
            episodes=episodes,
            sides=cfg["sides"],
            n_components=cfg["n_pca_components"],
            pca_save_dir=pca_save_dir,
        )

    verify(
        episodes_root=str(root),
        episodes=episodes,
        sides=cfg["sides"],
    )

    print("\n[完成]")


if __name__ == "__main__":
    main()
