"""
将 Dobot_peg_in_hole 格式的数据集

使用前请先运行触觉数据转换脚本，从 tactile_left_warped_image.mp4 /
tactile_right_warped_image.mp4 中提取 2D 标记点坐标，并对位移场做 PCA 降维：
    python extract_dobot_tactile_markers_and_pca.py

转换为 RDP (Reactive Diffusion Policy) 所需的 zarr 格式。

数据字段对应关系：
  robot_tcp_pose.npy                -> left_robot_tcp_pose  (xyz + 6D rotation，由四元数[qx,qy,qz,qw]转换)
  gripper.npy                       -> left_robot_gripper_width (原始读数，不做单位换算)
  marker_offset_emb_left.npy        -> left_gripper1_marker_offset_emb
  marker_offset_emb_right.npy       -> left_gripper2_marker_offset_emb
  marker_offset_left.npy            -> left_gripper1_marker_offset
  marker_offset_right.npy           -> left_gripper2_marker_offset
  initial_marker_left.npy           -> left_gripper1_initial_marker
  initial_marker_right.npy          -> left_gripper2_initial_marker
  realsense_wrist_rgb.mp4           -> left_wrist_img   (resize 到 240×320)
  realsense_top_rgb.mp4             -> external_img     (resize 到 240×320)
  tactile_left_warped_image.mp4     -> left_gripper1_img (保持原始尺寸 240×240)
  tactile_right_warped_image.mp4    -> left_gripper2_img (保持原始尺寸 240×240)
  timestamps.npy                    -> timestamp
  derived from above                -> action (next frame xyz + gripper)
  derived from above                -> target (current frame xyz + gripper)

与 convert_umi_to_zarr.py 的主要差异：
  1. 数据文件直接位于 episode_X/ 下，无 left_hand/ 子目录
  2. 视频名改为 realsense_wrist_rgb.mp4 / realsense_top_rgb.mp4
     以及 tactile_{side}_warped_image.mp4
  3. 新增 external_img 字段（俯视 RGB 相机）
  4. timestamps.npy 比 npy 数据多一帧（N+1），转换时截断为 N

位姿转换说明：
  robot_tcp_pose = [x, y, z, qx, qy, qz, qw]（7 维），与 UMI 数据集格式相同，
  四元数由 scipy 转为 6D 旋转表示，最终 left_robot_tcp_pose 为 (N, 9)。

用法：
  python scripts/convert_dobot_to_zarr.py \\
      --src /home/zzw/teleop_data/Dobot_peg_in_hole \\
      --dst data/hf_dataset/dataset_mini/dobot_peg_in_hole_zarr
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
    6D = 旋转矩阵前两列展开：[R[:,0], R[:,1]] -> R[:,:2].T.flatten()
    输入: (N, 4)
    输出: (N, 6)
    """
    rots = Rotation.from_quat(quat_xyzw)           # scipy: [qx,qy,qz,qw]
    mats = rots.as_matrix()                         # (N, 3, 3)
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

    位姿：robot_tcp_pose = [x, y, z, qx, qy, qz, qw]，四元数由 scipy 转为 6D 旋转。
    夹爪：保持原始读数，不做任何单位换算（训练和部署保持一致即可）。
    图像：
      - realsense_wrist_rgb.mp4 (640×480) → resize 到 240×320
      - realsense_top_rgb.mp4   (640×480) → resize 到 240×320
      - tactile_*_warped_image.mp4 (240×240) → 保持原始尺寸，当前训练不直接用触觉图
    timestamps：各 npy 之间帧数可能有 ±1 的偏差，以 tcp_pose / gripper / timestamps 三者
              的最小帧数为基准 N，所有数据统一截断至 N，图像同理只截断不填充。
    """
    # ── 基础数据 ──────────────────────────────────────────────────────────────
    # ── 读取所有数据 ──────────────────────────────────────────────────────────
    tcp_pose_raw  = np.load(os.path.join(ep_dir, 'robot_tcp_pose.npy')).astype(np.float32)  # (?, 7)
    gripper_raw   = np.load(os.path.join(ep_dir, 'gripper.npy')).astype(np.float32)         # (?,) 或 (?,1)
    timestamp_raw = np.load(os.path.join(ep_dir, 'timestamps.npy')).astype(np.float32)

    # gripper 统一为 (?, 1)
    if gripper_raw.ndim == 1:
        gripper_raw = gripper_raw[:, None]

    marker_emb_l  = np.load(os.path.join(ep_dir, 'marker_offset_emb_left.npy')).astype(np.float32)
    marker_emb_r  = np.load(os.path.join(ep_dir, 'marker_offset_emb_right.npy')).astype(np.float32)
    marker_off_l  = np.load(os.path.join(ep_dir, 'marker_offset_left.npy')).astype(np.float32)
    marker_off_r  = np.load(os.path.join(ep_dir, 'marker_offset_right.npy')).astype(np.float32)
    init_marker_l = np.load(os.path.join(ep_dir, 'initial_marker_left.npy')).astype(np.float32)
    init_marker_r = np.load(os.path.join(ep_dir, 'initial_marker_right.npy')).astype(np.float32)

    # realsense_wrist_rgb.mp4  (640×480) → resize 到 240×320
    wrist_img    = read_video_frames(os.path.join(ep_dir, 'realsense_wrist_rgb.mp4'),
                                     target_h=240, target_w=320)
    # realsense_top_rgb.mp4    (640×480) → resize 到 240×320
    external_img = read_video_frames(os.path.join(ep_dir, 'realsense_top_rgb.mp4'),
                                     target_h=240, target_w=320)
    # tactile (240×240) → 保持原始尺寸，target_h/w=None 表示不 resize
    tactile_l    = read_video_frames(os.path.join(ep_dir, 'tactile_left_warped_image.mp4'))
    tactile_r    = read_video_frames(os.path.join(ep_dir, 'tactile_right_warped_image.mp4'))

    # ── 以所有数据中最短的帧数为基准 N，统一截断 ──────────────────────────────
    all_lens = {
        'robot_tcp_pose':             len(tcp_pose_raw),
        'gripper':                    len(gripper_raw),
        'timestamps':                 len(timestamp_raw),
        'marker_offset_emb_left':     len(marker_emb_l),
        'marker_offset_emb_right':    len(marker_emb_r),
        'marker_offset_left':         len(marker_off_l),
        'marker_offset_right':        len(marker_off_r),
        'initial_marker_left':        len(init_marker_l),
        'initial_marker_right':       len(init_marker_r),
        'realsense_wrist_rgb':        len(wrist_img),
        'realsense_top_rgb':          len(external_img),
        'tactile_left_warped_image':  len(tactile_l),
        'tactile_right_warped_image': len(tactile_r),
    }
    N = min(all_lens.values())
    if len(set(all_lens.values())) > 1:
        print(f"  [警告] 各数据帧数不一致，统一截断至最小帧数 N={N}:")
        for k, v in all_lens.items():
            if v != N:
                print(f"    {k}: {v} -> {N}")

    tcp_pose_raw  = tcp_pose_raw[:N]
    gripper_raw   = gripper_raw[:N]
    timestamp     = timestamp_raw[:N]
    marker_emb_l  = marker_emb_l[:N]
    marker_emb_r  = marker_emb_r[:N]
    marker_off_l  = marker_off_l[:N]
    marker_off_r  = marker_off_r[:N]
    init_marker_l = init_marker_l[:N]
    init_marker_r = init_marker_r[:N]
    wrist_img     = wrist_img[:N]
    external_img  = external_img[:N]
    tactile_l     = tactile_l[:N]
    tactile_r     = tactile_r[:N]

    # ── left_robot_tcp_pose: [x,y,z] + 6D rotation ──────────────────────────
    # robot_tcp_pose[:, :3] = [x, y, z]
    # robot_tcp_pose[:, 3:] = [qx, qy, qz, qw]，scipy.Rotation.from_quat 接受此格式 ✓
    xyz   = tcp_pose_raw[:, :3]           # (N, 3)
    quat  = tcp_pose_raw[:, 3:]           # (N, 4) [qx, qy, qz, qw]
    rot6d = quat_to_6d(quat)             # (N, 6)
    tcp_pose = np.concatenate([xyz, rot6d], axis=1).astype(np.float32)  # (N, 9)

    # ── target: [x,y,z, gripper] at current frame ───────────────────────────
    target = np.concatenate([xyz, gripper_raw], axis=1).astype(np.float32)  # (N, 4)

    # ── action: [x,y,z, gripper] at NEXT frame (shift +1, last frame repeats)
    action = np.concatenate(
        [np.concatenate([xyz[1:],          xyz[-1:]],          axis=0),
         np.concatenate([gripper_raw[1:],  gripper_raw[-1:]],  axis=0)],
        axis=1).astype(np.float32)                                           # (N, 4)

    return {
        'timestamp':                          timestamp,
        'left_robot_tcp_pose':                tcp_pose,
        'left_robot_gripper_width':           gripper_raw,
        'action':                             action,
        'target':                             target,
        'left_gripper1_marker_offset_emb':    marker_emb_l,
        'left_gripper2_marker_offset_emb':    marker_emb_r,
        'left_gripper1_marker_offset':        marker_off_l,
        'left_gripper2_marker_offset':        marker_off_r,
        'left_gripper1_initial_marker':       init_marker_l,
        'left_gripper2_initial_marker':       init_marker_r,
        'left_wrist_img':                     wrist_img,
        'external_img':                       external_img,
        'left_gripper1_img':                  tactile_l,
        'left_gripper2_img':                  tactile_r,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert Dobot dataset to RDP zarr format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/convert_dobot_to_zarr.py
  python scripts/convert_dobot_to_zarr.py --src /home/zzw/teleop_data/Dobot_peg_in_hole --dst data/hf_dataset/dobot_zarr
  python scripts/convert_dobot_to_zarr.py --max_episodes 5   # 调试：只处理前5个
  python scripts/convert_dobot_to_zarr.py --resume           # 断点续传
        """
    )
    parser.add_argument('--src', type=str,
                        default='/home/zzw/teleop_data/Dobot_peg_in_hole',
                        help='源数据集根目录（包含 episode_* 子文件夹）')
    parser.add_argument('--dst', type=str,
                        default='data/hf_dataset/dobot_peg_in_hole_zarr',
                        help='输出目录（相对于项目根目录或绝对路径）')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='最多处理的 episode 数量（调试用）')
    parser.add_argument('--resume', action='store_true',
                        help='从已有的 zarr 中断点继续转换（跳过已写入的 episode）')
    args = parser.parse_args()

    # 解析路径
    src_dir = os.path.expanduser(args.src)
    dst_dir = args.dst if os.path.isabs(args.dst) else os.path.join(ROOT, args.dst)
    zarr_path = os.path.join(dst_dir, 'replay_buffer.zarr')

    if not os.path.isdir(src_dir):
        print(f"[错误] 源数据集目录不存在: {src_dir}")
        sys.exit(1)

    # 找出所有 episode 目录，按数字排序
    all_eps = sorted(
        [d for d in os.listdir(src_dir) if d.startswith('episode_')],
        key=lambda x: int(x.split('_')[1])
    )
    if args.max_episodes is not None:
        all_eps = all_eps[:args.max_episodes]

    print(f"[INFO] 源数据集: {src_dir}")
    print(f"[INFO] 输出路径: {zarr_path}")
    print(f"[INFO] 共找到 {len(all_eps)} 个 episode")

    # 创建或打开 zarr ReplayBuffer
    os.makedirs(dst_dir, exist_ok=True)
    start_ep_idx = 0

    if os.path.exists(zarr_path):
        if args.resume:
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
    print(f"[INFO] 本次处理 {len(eps_to_process)} 个 episode...")

    # 图像 chunk 配置
    # 腕部 / 俯视相机 resize 到 240×320；触觉相机保持 240×240
    chunks_cfg = {
        'left_wrist_img':    (100, 240, 320, 3),
        'external_img':      (100, 240, 320, 3),
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
            import traceback
            print(f"\n[错误] {ep_name}: {e}")
            traceback.print_exc()
            failed.append(ep_name)
            continue

    print(f"\n转换完成！")
    print(f"  成功 episode: {len(all_eps) - len(failed)}/{len(all_eps)}")
    if failed:
        print(f"  失败 episode: {failed}")
    print(f"  总时间步数:  {replay_buffer.n_steps}")
    print(f"  总 episode 数: {replay_buffer.n_episodes}")
    print(f"  输出路径: {zarr_path}")

    # 打印数据结构摘要
    print("\n数据结构摘要：")
    for k in sorted(replay_buffer.keys()):
        arr = replay_buffer[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == '__main__':
    main()
