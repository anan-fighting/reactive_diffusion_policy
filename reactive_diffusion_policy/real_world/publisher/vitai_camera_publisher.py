"""
vitai_camera_publisher.py
=========================
ViTai GF225 触觉传感器 ROS2 Publisher。

与 GelsightCameraPublisher / MCTacCameraPublisher 结构对齐，
使用 pyvitaisdk 做实时标记点检测，发布：
  - /{camera_name}/color/image_raw       : 传感器 RGB 图像
  - /{camera_name}/marker_offset/information : 标记点位置 + 位移 (PointCloud2)

使用方法（单独测试）：
    python -m reactive_diffusion_policy.real_world.publisher.vitai_camera_publisher

依赖：
    pip install pyvitaisdk*.whl  (ViTai 官方 SDK)
"""

import os
import copy
import math
import socket
import struct
import time
import uuid

import bson
import cv2
import numpy as np
import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField

from reactive_diffusion_policy.common.data_models import TactileSensorMessage, Arrow
from reactive_diffusion_policy.common.tactile_marker_utils import marker_normalization


def _import_sdk():
    """延迟导入 pyvitaisdk，给出友好错误提示。"""
    try:
        from pyvitaisdk import VTSensor, VTSDataType, VTSError, VTSensorType
        return VTSensor, VTSDataType, VTSError, VTSensorType
    except ImportError:
        logger.error(
            "[VitaiCameraPublisher] pyvitaisdk 未安装！\n"
            "  请先安装 ViTai SDK：pip install pyvitaisdk*.whl"
        )
        raise


class VitaiCameraPublisher(Node):
    """
    ViTai GF225 触觉传感器 ROS2 Publisher。

    内部使用 pyvitaisdk 对每帧图像做实时标记点检测，
    输出的 ROS2 话题格式与 GelsightCameraPublisher 完全一致，
    可被 device_mapping_server 和 RealEnv 直接消费。
    """

    def __init__(self,
                 camera_index: int = 0,
                 camera_type: str = 'vitai',
                 fps: int = 24,
                 camera_name: str = 'left_gripper_camera_1',
                 vr_server_ip: str = '127.0.0.1',
                 vr_server_port: int = 10002,
                 dimension: int = 2,
                 marker_vis_rotation_angle: float = 0.,  # 度
                 sensor_type_str: str = 'GF225',
                 # 用第几帧做背景校准（0 = 启动时第一帧）
                 calib_frame_count: int = 30,
                 debug: bool = False,
                 ):
        node_name = f'{camera_name}_vitai_publisher_{camera_index}'
        super().__init__(node_name)

        self.camera_index = camera_index
        self.camera_name = camera_name
        self.camera_type = camera_type
        self.fps = fps
        self.dimension = dimension
        self.marker_vis_rotation_angle = np.deg2rad(marker_vis_rotation_angle)
        self.sensor_type_str = sensor_type_str
        self.calib_frame_count = calib_frame_count
        self.debug = debug

        self.vr_server_ip = vr_server_ip
        self.vr_server_port = vr_server_port

        # ROS2 Publishers
        self.color_publisher_ = self.create_publisher(
            Image, f'/{camera_name}/color/image_raw', 10)
        self.marker_publisher = self.create_publisher(
            PointCloud2, f'/{camera_name}/marker_offset/information', 10)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

        # UDP socket（向 VR 服务器发送触觉可视化消息）
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 相机
        self.cap = None
        # ViTai GF225 原生分辨率：尝试设置 240×240，实际可能返回 320×240，需中心裁剪
        self.width = 240
        self.height = 240
        # 相机实际输出分辨率（_start 后确定）
        self._cap_width = 240
        self._cap_height = 240

        # ViTai SDK 相关
        self.vtsensor = None
        self._VTSensor = None
        self._VTSDataType = None
        self._VTSError = None
        self._VTSensorType = None
        self._calib_frames_collected = 0
        self._calibrated = False
        self._calib_buffer = []  # 存储校准帧

        # 标记点网格（首次检测后确定）
        self.n_markers = None

        # 帧率统计
        self.prev_time = time.time()
        self.frame_count = 0
        self.last_print_time = time.time()
        self.last_frame_time = None

        self._start()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _start(self):
        """打开 USB 相机并初始化 ViTai SDK。"""
        VTSensor, VTSDataType, VTSError, VTSensorType = _import_sdk()
        self._VTSensor = VTSensor
        self._VTSDataType = VTSDataType
        self._VTSError = VTSError
        self._VTSensorType = VTSensorType

        sensor_type_map = {"GF225": VTSensorType.GF225}
        if self.sensor_type_str not in sensor_type_map:
            raise ValueError(f"不支持的 ViTai 传感器型号: {self.sensor_type_str}，目前支持: {list(sensor_type_map.keys())}")
        self._sensor_type = sensor_type_map[self.sensor_type_str]

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机 index={self.camera_index}")

        # 尝试设置 240×240（pyvitaisdk 期望值）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 读取相机实际输出分辨率
        self._cap_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._cap_width != self.width or self._cap_height != self.height:
            logger.warning(f"[{self.camera_name}] 相机实际分辨率 {self._cap_width}×{self._cap_height} "
                           f"与 pyvitaisdk 期望 {self.width}×{self.height} 不符，"
                           f"将对每帧做中心裁剪至 {self.width}×{self.height}。")

        logger.info(f"[{self.camera_name}] 相机 index={self.camera_index} 已打开，"
                    f"采集分辨率: {self._cap_width}×{self._cap_height}，"
                    f"送入 SDK 分辨率: {self.width}×{self.height}，"
                    f"将采集 {self.calib_frame_count} 帧后完成背景校准。")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.vtsensor is not None:
            try:
                self.vtsensor.release()
            except Exception:
                pass
            self.vtsensor = None
        logger.info(f"[{self.camera_name}] 已停止")

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        """将帧中心裁剪至 self.width × self.height（pyvitaisdk 期望分辨率）。"""
        h, w = frame.shape[:2]
        th, tw = self.height, self.width
        if h == th and w == tw:
            return frame
        y0 = (h - th) // 2
        x0 = (w - tw) // 2
        return frame[y0:y0 + th, x0:x0 + tw]

    # ------------------------------------------------------------------
    # 背景校准（运行时自动完成）
    # ------------------------------------------------------------------

    def _try_calibrate(self, frame: np.ndarray) -> bool:
        """
        积累足够帧后取均值做背景校准。
        返回 True 表示本次调用完成了校准。
        """
        self._calib_buffer.append(frame.copy())
        self._calib_frames_collected += 1

        if self._calib_frames_collected < self.calib_frame_count:
            if self._calib_frames_collected % 10 == 0:
                logger.info(f"[{self.camera_name}] 背景校准进度: "
                            f"{self._calib_frames_collected}/{self.calib_frame_count}")
            return False

        # 取所有校准帧的均值作为背景
        calib_image = np.mean(np.stack(self._calib_buffer, axis=0), axis=0).astype(np.uint8)
        self._calib_buffer.clear()

        self.vtsensor = self._VTSensor(config=None, sensor_type=self._sensor_type)
        self.vtsensor.calibrate(calib_image=calib_image)

        # 从当前帧推断标记点网格尺寸
        data = self.vtsensor.collect_sensor_data(
            self._VTSDataType.MARKER_ORIGIN_VECTOR,
            self._VTSDataType.MARKER_OFFSET_VECTOR,
            frame=frame
        )
        grid_shape = data[self._VTSDataType.MARKER_ORIGIN_VECTOR].shape[:2]
        self.n_markers = grid_shape[0] * grid_shape[1]
        self._calibrated = True

        logger.info(f"[{self.camera_name}] 背景校准完成！"
                    f"标记点网格: {grid_shape[0]}×{grid_shape[1]} = {self.n_markers} 个标记点")
        return True

    # ------------------------------------------------------------------
    # 标记点检测
    # ------------------------------------------------------------------

    def _detect_markers(self, frame: np.ndarray):
        """
        用 pyvitaisdk 检测当前帧的标记点位置和位移。

        Returns
        -------
        initial_markers : np.ndarray  shape (n_markers, 3)  第三维固定为 0（2D 模式）
        marker_offset   : np.ndarray  shape (n_markers, 2)
        """
        data = self.vtsensor.collect_sensor_data(
            self._VTSDataType.MARKER_ORIGIN_VECTOR,
            self._VTSDataType.MARKER_OFFSET_VECTOR,
            frame=frame
        )
        # 像素坐标
        origin = data[self._VTSDataType.MARKER_ORIGIN_VECTOR]  # (N, M, 2)
        # 像素位移
        offset = data[self._VTSDataType.MARKER_OFFSET_VECTOR]  # (N, M, 2)

        n = self.n_markers
        # 归一化到 [0, 1]，与 extract_dobot_tactile_markers_and_pca.py 训练时一致：
        #   norm = np.array([W, H])  → 此处 W=H=self.width=240
        norm = np.array([self.width, self.height], dtype=np.float32)
        origin_flat = origin.reshape(n, 2).astype(np.float32) / norm  # (n_markers, 2) 归一化坐标 [0,1]
        offset_flat = offset.reshape(n, 2).astype(np.float32) / norm  # (n_markers, 2) 归一化位移 [0,1]

        # 拼成 (n_markers, 3)，第三维=0（2D 模式与 gelsight 格式一致）
        initial_markers = np.hstack([origin_flat, np.zeros((n, 1), dtype=np.float32)])
        return initial_markers, offset_flat

    # ------------------------------------------------------------------
    # 发布 ROS2 消息（与 GelsightCameraPublisher 接口一致）
    # ------------------------------------------------------------------

    def publish_marker_offset(self, marker_loc: np.ndarray, marker_offset: np.ndarray,
                              camera_timestamp: Time):
        """发布 PointCloud2 格式的标记点位置 + 位移。"""
        cur_marker = marker_loc[:, :2].copy()
        marker_information = np.hstack((cur_marker, marker_offset)).astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'
        msg.is_bigendian = False
        msg.point_step = 16  # 4 * float32
        msg.is_dense = True
        msg.fields = [
            PointField(name='marker_location_x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_location_y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_x',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_y',   offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = b''.join(
            struct.pack('ffff', row[0], row[1], row[2], row[3])
            for row in marker_information
        )
        self.marker_publisher.publish(msg)

    def publish_color_image(self, color_image: np.ndarray, camera_timestamp: Time):
        """发布 BGR8 格式的 RGB 图像（原始像素，rviz2 可直接渲染）。"""
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_color_frame_{self.camera_index}'
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = color_image.tobytes()
        self.color_publisher_.publish(msg)

    def send_tactile_sensor_msg(self, initial_markers: np.ndarray, marker_offsets: np.ndarray):
        """向 VR 服务器发送 BSON 格式的触觉可视化消息。"""
        initial_markers = initial_markers.copy()
        marker_offsets = marker_offsets.copy()

        # 坐标系变换（与 GelsightCameraPublisher 一致）
        initial_markers[:, :2] -= 0.5
        initial_markers *= 0.25
        initial_markers[:, 2] *= 0.1
        marker_offsets *= 2.0
        z_offset = 0.1

        if self.dimension == 2:
            marker_offsets = np.concatenate(
                [marker_offsets, np.zeros((marker_offsets.shape[0], 1))], axis=1)

        rotation_matrix = np.array([
            [np.cos(self.marker_vis_rotation_angle), -np.sin(self.marker_vis_rotation_angle)],
            [np.sin(self.marker_vis_rotation_angle),  np.cos(self.marker_vis_rotation_angle)]
        ])
        initial_markers[:, :2] = initial_markers[:, :2] @ rotation_matrix.T
        marker_offsets[:, :2]  = marker_offsets[:, :2]  @ rotation_matrix.T

        arrow = []
        for init_m, offset in zip(initial_markers, marker_offsets):
            start = [init_m[0], init_m[1], z_offset + init_m[2]]
            end   = [init_m[0] + offset[0], init_m[1] + offset[1], z_offset + init_m[2]]
            arrow.append(Arrow(start=start, end=end))

        tactile_msg = TactileSensorMessage(device_id=self.camera_name, arrows=arrow).model_dump()
        self.socket.sendto(bson.dumps(tactile_msg), (self.vr_server_ip, self.vr_server_port))

    # ------------------------------------------------------------------
    # ROS2 Timer 回调（主循环）
    # ------------------------------------------------------------------

    def timer_callback(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"[{self.camera_name}] 读取帧失败，跳过")
            return

        # 中心裁剪至 pyvitaisdk 期望的分辨率（240×240）
        frame = self._center_crop(frame)
        camera_timestamp = self.get_clock().now()

        # ── 背景校准阶段：只发布图像，不发布标记点 ──
        if not self._calibrated:
            done = self._try_calibrate(frame)
            self.publish_color_image(frame, camera_timestamp)
            if not done:
                return
            # 校准完成，继续本帧处理

        # ── 标记点检测 ──
        try:
            initial_markers, marker_offset = self._detect_markers(frame)
        except Exception as e:
            logger.warning(f"[{self.camera_name}] 标记点检测失败: {e}")
            self.publish_color_image(frame, camera_timestamp)
            return

        # ── 归一化（像素坐标 → [0,1]）──
        initial_markers_norm, marker_offset_norm = marker_normalization(
            copy.deepcopy(initial_markers),
            copy.deepcopy(marker_offset),
            self.dimension,
            width=self.width,
            height=self.height
        )

        # ── 发布 ROS2 消息 ──
        self.publish_marker_offset(initial_markers_norm, marker_offset_norm, camera_timestamp)
        self.publish_color_image(frame, camera_timestamp)

        # ── 向 VR 服务器发送触觉可视化 ──
        self.send_tactile_sensor_msg(
            copy.deepcopy(initial_markers_norm),
            copy.deepcopy(marker_offset_norm)
        )

        # ── 帧率统计 ──
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed >= 1.0:
            logger.debug(f"[{self.camera_name}] FPS: {self.frame_count / elapsed:.1f}")
            self.prev_time = current_time
            self.frame_count = 0

        if current_time - self.last_print_time >= 5.0:
            logger.info(f"[{self.camera_name}] 发布中，时间戳(s): "
                        f"{camera_timestamp.nanoseconds / 1e9:.3f}")
            self.last_print_time = current_time


# ---------------------------------------------------------------------------
# 单独测试入口
# ---------------------------------------------------------------------------

def main(args=None):
    import psutil
    cpu_core_id = {11, 12, 13}
    os.sched_setaffinity(0, cpu_core_id)

    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    cv2.setNumThreads(4)

    rclpy.init(args=args)
    node = VitaiCameraPublisher(
        camera_index=0,
        camera_name='left_gripper_camera_1',
        fps=24,
        dimension=2,
        sensor_type_str='GF225',
        calib_frame_count=30,
        debug=True,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
