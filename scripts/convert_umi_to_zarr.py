"""
将 pick_up_and_place_blocks_sync_0416 格式的 UMI 数据集

使用前请先运行数据转换脚本,将触觉的tactile_left.mp4 / tactile_right.mp4 中提取 2D 标记点坐标，并对位移场做 PCA 降维，生成 RDP 训练所需的三个数组
    python extract_umi_tactile_markers_and_pca.py 

转换为 RDP (Reactive Diffusion Policy) 所需的 zarr 格式。

数据字段对应关系：
  left_hand/vio_pose.npy                -> left_robot_tcp_pose  (xyz + 6D rotation，由四元数[qx,qy,qz,qw]转换)
  left_hand/gripper.npy                 -> left_robot_gripper_width (原始读数，不做单位换算)
  left_hand/marker_offset_emb_left.npy  -> left_gripper1_marker_offset_emb
  left_hand/marker_offset_emb_right.npy -> left_gripper2_marker_offset_emb
  left_hand/marker_offset_left.npy      -> left_gripper1_marker_offset
  left_hand/marker_offset_right.npy     -> left_gripper2_marker_offset
  left_hand/initial_marker_left.npy     -> left_gripper1_initial_marker
  left_hand/initial_marker_right.npy    -> left_gripper2_initial_marker
  left_hand/rgb.mp4                     -> left_wrist_img  (resize 到 240×320，与 shape_meta image_shape 一致)
  left_hand/tactile_left.mp4            -> left_gripper1_img  (保持原始尺寸 240×240；触觉图不作为模型输入)
  left_hand/tactile_right.mp4           -> left_gripper2_img  (保持原始尺寸 240×240；触觉图不作为模型输入)
  timestamps.npy                        -> timestamp
  derived from above                    -> action (next frame xyz + gripper)
  derived from above                    -> target (current frame xyz + gripper)

图像 resize 说明：
  - rgb.mp4 (640×480)：固定 resize 到 240×320，与 shape_meta image_shape 保持一致（硬编码，无需参数）。
  - tactile_left/right.mp4 (240×240)：当前训练配置（AT/LDP）中触觉图不作为模型输入，
    只用 marker_offset_emb（低维向量），故保持原始 240×240 存储，不做 resize。

用法：
  python scripts/convert_umi_to_zarr.py \
      --src /home/zzw/project/umi-act/ur5e_datasets/pick_up_and_place_blocks_sync_0416 \
      --dst data/hf_dataset/dataset_mini/umi_rdp_zarr
"""

import argparse
import os
import sys
import numpy as np
import zarr
import cv2
from scipy.spatial.transform import Rotation
import tqdm

# 将项目根路径加入 sys.path，以便导入 reactive_diffusion_policy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    将四元数 [qx, qy, qz, qw] 批量转换为 6D 旋转表示。
    6D = 旋转矩阵前两列展开：[R[:,0], R[:,1]] = R[:,:2].T.flatten()
    输入: (N, 4)
    输出: (N, 6)
    """
    rots = Rotation.from_quat(quat_xyzw)          # scipy: [qx,qy,qz,qw]
    mats = rots.as_matrix()                        # (N, 3, 3)
    # 取前两列，转置后展平: (N, 2, 3) -> (N, 6)
    rot_6d = mats[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return rot_6d


def read_video_frames(video_path: str, target_h: int = None, target_w: int = None) -> np.ndarray:
    """
    读取 mp4 视频，返回 (N, H, W, 3) uint8（RGB）。
    若 target_h / target_w 非 None，则 resize 到指定尺寸；否则保持原始分辨率。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 读取为 BGR，转 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_h is not None and target_w is not None:
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(frame, (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"视频无帧: {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def process_episode(ep_dir: str):
    """
    处理单个 episode，返回 dict of np.ndarray（各字段，第 0 轴均为时间步）。

    夹爪：保持原始读数，不做任何单位换算（训练和部署保持一致即可）。
    位姿：vio_pose = [x, y, z, qx, qy, qz, qw]，四元数由 scipy 转为 6D 旋转。
    图像：
      - rgb.mp4 (640×480) 固定 resize 到 240×320，与 shape_meta image_shape 一致。
      - tactile_left/right.mp4 (240×240) 保持原始尺寸，当前训练不直接用触觉图。
    """
    lh = os.path.join(ep_dir, 'left_hand')
    ts_path = os.path.join(ep_dir, 'timestamps.npy')

    # ── 基础数据 ──────────────────────────────────────────────────────────────
    vio_pose  = np.load(os.path.join(lh, 'vio_pose.npy')).astype(np.float32)   # (N,7)
    gripper   = np.load(os.path.join(lh, 'gripper.npy')).astype(np.float32)    # (N,1)
    timestamp = np.load(ts_path).astype(np.float32)                             # (N,)

    N = len(vio_pose)
    assert len(gripper) == N and len(timestamp) == N, \
        f"时间步不一致: vio_pose={N}, gripper={len(gripper)}, timestamp={len(timestamp)}"

    # ── left_robot_tcp_pose: [x,y,z] + 6D rotation ──────────────────────────
    # vio_pose[:, 3:] = [qx, qy, qz, qw]，scipy.Rotation.from_quat 接受此格式 ✓
    xyz   = vio_pose[:, :3]                             # (N,3)
    quat  = vio_pose[:, 3:]                             # (N,4) [qx,qy,qz,qw]
    rot6d = quat_to_6d(quat)                            # (N,6)
    tcp_pose = np.concatenate([xyz, rot6d], axis=1).astype(np.float32)  # (N,9)

    # ── left_robot_gripper_width: 保持原始读数，不做换算 ──────────────────────
    # 范围约 0~45（原始传感器读数），训练和部署时保持一致
    gripper_raw = gripper.astype(np.float32)            # (N,1)

    # ── target: [x,y,z, gripper] at current frame ───────────────────────────
    target = np.concatenate([xyz, gripper_raw], axis=1).astype(np.float32)  # (N,4)

    # ── action: [x,y,z, gripper] at NEXT frame (shift +1, last frame repeats)
    action = np.concatenate(
        [np.concatenate([xyz[1:],         xyz[-1:]],         axis=0), # xyz 往前移1帧，末尾补最后帧
         np.concatenate([gripper_raw[1:], gripper_raw[-1:]], axis=0)], # gripper 同理
        axis=1).astype(np.float32)                          # (N,4)

    # ── 触觉 marker 数据 ──────────────────────────────────────────────────────
    marker_emb_l  = np.load(os.path.join(lh, 'marker_offset_emb_left.npy')).astype(np.float32)
    marker_emb_r  = np.load(os.path.join(lh, 'marker_offset_emb_right.npy')).astype(np.float32)
    marker_off_l  = np.load(os.path.join(lh, 'marker_offset_left.npy')).astype(np.float32)
    marker_off_r  = np.load(os.path.join(lh, 'marker_offset_right.npy')).astype(np.float32)
    init_marker_l = np.load(os.path.join(lh, 'initial_marker_left.npy')).astype(np.float32)
    init_marker_r = np.load(os.path.join(lh, 'initial_marker_right.npy')).astype(np.float32)

    # ── 图像 ──────────────────────────────────────────────────────────────────
    # rgb.mp4 (640×480) → 固定 resize 到 240×320（与 shape_meta image_shape 一致）
    wrist_img = read_video_frames(os.path.join(lh, 'rgb.mp4'), target_h=240, target_w=320)
    # tactile (240×240) → 保持原始尺寸，target_h/w=None 表示不 resize
    tactile_l = read_video_frames(os.path.join(lh, 'tactile_left.mp4'))
    tactile_r = read_video_frames(os.path.join(lh, 'tactile_right.mp4'))

    # 对齐帧数（视频帧可能与 npy 帧数略有偏差）
    for name, arr in [('left_wrist_img', wrist_img),
                       ('left_gripper1_img', tactile_l),
                       ('left_gripper2_img', tactile_r)]:
        if len(arr) != N:
            print(f"  [警告] {name} 帧数 {len(arr)} 与 npy 帧数 {N} 不一致，将截断/填充")

    def align(arr, n):
        if len(arr) >= n:
            return arr[:n]
        # 用最后一帧填充
        pad = np.repeat(arr[-1:], n - len(arr), axis=0)
        return np.concatenate([arr, pad], axis=0)

    wrist_img = align(wrist_img, N)
    tactile_l = align(tactile_l, N)
    tactile_r = align(tactile_r, N)

    return {
        'timestamp':                      timestamp,
        'left_robot_tcp_pose':            tcp_pose,
        'left_robot_gripper_width':       gripper_raw,
        'action':                         action,
        'target':                         target,
        'left_gripper1_marker_offset_emb':  marker_emb_l,
        'left_gripper2_marker_offset_emb':  marker_emb_r,
        'left_gripper1_marker_offset':      marker_off_l,
        'left_gripper2_marker_offset':      marker_off_r,
        'left_gripper1_initial_marker':     init_marker_l,
        'left_gripper2_initial_marker':     init_marker_r,
        'left_wrist_img':                   wrist_img,
        'left_gripper1_img':                tactile_l,
        'left_gripper2_img':                tactile_r,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Convert UMI dataset to RDP zarr format')
    parser.add_argument('--src', type=str,
                        default='/home/zzw/project/umi-act/ur5e_datasets/pick_up_and_place_blocks_sync_0416',
                        help='源数据集根目录（包含 episode_* 子文件夹）')
    parser.add_argument('--dst', type=str,
                        default='data/hf_dataset/dataset_mini/umi_rdp_zarr',
                        help='输出目录（相对于项目根目录或绝对路径）')
    parser.add_argument('--gripper_scale', type=float, default=1.0,
                        help='夹爪缩放因子（默认 1.0：不做换算，保持原始读数）')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='最多处理的 episode 数量（调试用）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有的 zarr 中断点继续转换（跳过已写入的 episode）')
    args = parser.parse_args()

    # 解析路径
    src_dir = os.path.expanduser(args.src)
    dst_dir = args.dst if os.path.isabs(args.dst) else os.path.join(ROOT, args.dst)
    zarr_path = os.path.join(dst_dir, 'replay_buffer.zarr')

    # 找出所有 episode 目录
    all_eps = sorted(
        [d for d in os.listdir(src_dir) if d.startswith('episode_')],
        key=lambda x: int(x.split('_')[1])
    )
    if args.max_episodes is not None:
        all_eps = all_eps[:args.max_episodes]

    # 创建或打开 zarr ReplayBuffer
    os.makedirs(dst_dir, exist_ok=True)
    start_ep_idx = 0

    if os.path.exists(zarr_path):
        if args.resume:
            # 打开已有的 zarr，统计已写入的 episode 数
            store = zarr.DirectoryStore(zarr_path)
            root = zarr.open_group(store=store, mode='a')
            replay_buffer = ReplayBuffer.create_from_group(root)
            start_ep_idx = replay_buffer.n_episodes
            print(f"[续传] 检测到已有 zarr，已写入 {start_ep_idx} 个 episode，从第 {start_ep_idx} 个继续...")
        else:
            print(f"[警告] 目标 zarr 已存在: {zarr_path}")
            ans = input("是否覆盖？(y/n，输入 r 续传): ").strip().lower()
            if ans == 'r':
                store = zarr.DirectoryStore(zarr_path)
                root = zarr.open_group(store=store, mode='a')
                replay_buffer = ReplayBuffer.create_from_group(root)
                start_ep_idx = replay_buffer.n_episodes
                print(f"[续传] 已写入 {start_ep_idx} 个 episode，从第 {start_ep_idx} 个继续...")
            elif ans == 'y':
                store = zarr.DirectoryStore(zarr_path)
                root = zarr.group(store=store, overwrite=True)
                replay_buffer = ReplayBuffer.create_from_group(root)
            else:
                print("已取消。")
                return
    else:
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)
        replay_buffer = ReplayBuffer.create_from_group(root)

    eps_to_process = all_eps[start_ep_idx:]
    print(f"共找到 {len(all_eps)} 个 episode，本次处理 {len(eps_to_process)} 个...")

    # 图像 chunk：腕部相机 240×320（固定），触觉相机 240×240（原始尺寸）
    chunks_cfg = {
        'left_wrist_img':    (100, 240, 320, 3),
        'left_gripper1_img': (100, 240, 240, 3),
        'left_gripper2_img': (100, 240, 240, 3),
    }

    failed = []
    for ep_name in tqdm.tqdm(eps_to_process, desc='Converting episodes'):
        ep_dir = os.path.join(src_dir, ep_name)
        try:
            ep_data = process_episode(ep_dir=ep_dir)
            replay_buffer.add_episode(data=ep_data, chunks=chunks_cfg)
        except Exception as e:
            print(f"\n[错误] {ep_name}: {e}")
            failed.append(ep_name)
            continue

    print(f"\n转换完成！")
    print(f"  成功 episode: {len(all_eps) - len(failed)}/{len(all_eps)}")
    if failed:
        print(f"  失败 episode: {failed}")
    print(f"  总时间步数: {replay_buffer.n_steps}")
    print(f"  总 episode 数: {replay_buffer.n_episodes}")
    print(f"  输出路径: {zarr_path}")

    # 打印数据结构摘要
    print("\n数据结构摘要：")
    for k in sorted(replay_buffer.keys()):
        arr = replay_buffer[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == '__main__':
    main()
