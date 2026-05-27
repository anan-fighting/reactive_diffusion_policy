"""
UR5e Robot HTTP Server
======================
提供与 nova5_server.py / bimanual_flexiv_server.py 完全相同的 REST API 接口，
底层通过 ur_rtde 库驱动 Universal Robots UR5e 机械臂。

依赖安装：
    pip install ur-rtde

通信说明（ur_rtde）：
    RTDEControlInterface  → 发送运动指令（servoL / moveJ / stopL 等）
    RTDEReceiveInterface  → 实时读取机器人状态（~500Hz）

坐标系约定：
    UR 内部：(x_m, y_m, z_m, rx, ry, rz)  ← 旋转向量（轴角，单位 m / rad）
    本接口对外：(x_m, y_m, z_m, qw, qx, qy, qz) ← 四元数，单位 m
    → 本服务器在两者之间自动转换（使用 scipy.spatial.transform）

高频 servoL 控制参数（24Hz 推荐值）：
    velocity      = 0.5   (m/s)  — 最大笛卡尔速度上限
    acceleration  = 0.5   (m/s²) — 最大笛卡尔加速度上限
    dt            = 1/24  (s)    — 与 RDP 控制周期一致
    lookahead_time= 0.1   (s)    — 前瞻窗口（类 PID D 项），推荐 0.03~0.2
    gain          = 300          — 位置增益（类 PID P 项），推荐 100~2000

夹爪控制（Robotiq Hand-E）：
    通过本仓库内置夹爪库
    reactive_diffusion_policy.real_world.robot.gripper.robotiq 驱动，
    底层使用 Modbus RTU 串口（默认 /dev/ttyUSB0）。
    ServoHandler → HandEForRtu → ModbusRTU
    夹爪位置单位：mm（0 = 全开，50mm = 全闭）
    对外 API 统一使用 width_m（米），内部自动完成换算。
"""

import math
import time
import threading
import argparse
from typing import List, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from scipy.spatial.transform import Rotation

# ur_rtde 库（pip install ur-rtde）
try:
    import rtde_control
    import rtde_receive
    import rtde_io
except ImportError:
    raise ImportError(
        "缺少 ur_rtde 库，请执行：pip install ur-rtde\n"
        "或参考 https://sdurobotics.gitlab.io/ur_rtde/installation/"
    )

# 本仓库内置的 Robotiq Hand-E 夹爪库（Modbus RTU 串口驱动）
from reactive_diffusion_policy.real_world.robot.gripper.robotiq.robotiq_gripper import (
    ServoHandler,
    AdaptiveGripper,
)

from reactive_diffusion_policy.common.data_models import (
    BimanualRobotStates,
    MoveGripperRequest,
    TargetTCPRequest,
)

# ──────────────────────────────────────────────
# 坐标转换工具函数
# ──────────────────────────────────────────────

def rotvec_to_quat(rotvec: List[float]):
    """
    UR 旋转向量 [rx, ry, rz]（轴角，rad）→ 四元数 [qw, qx, qy, qz]
    """
    r = Rotation.from_rotvec(rotvec)
    qx, qy, qz, qw = r.as_quat()   # scipy 返回 (qx, qy, qz, qw)
    return float(qw), float(qx), float(qy), float(qz)


def quat_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> List[float]:
    """
    四元数 [qw, qx, qy, qz] → UR 旋转向量 [rx, ry, rz]（轴角，rad）
    """
    r = Rotation.from_quat([qx, qy, qz, qw])   # scipy 接受 (qx, qy, qz, qw)
    return r.as_rotvec().tolist()


# ──────────────────────────────────────────────
# Robotiq Hand-E 夹爪控制器（本仓库内置库，Modbus RTU 串口）
# ──────────────────────────────────────────────

class RobotiqGripperController:
    """
    封装 Robotiq Hand-E 夹爪控制。
    底层使用本仓库内置库（ServoHandler + AdaptiveGripper），
    通过 Modbus RTU 串口（/dev/ttyUSB0）与夹爪通信。

    HandE 行程参数：
        maxPos = 0   mm → 完全张开（夹爪内部"最大位置"寄存器值对应开口最大）
        minPos = 50  mm → 完全闭合（50mm 对应 Full Stroke）
        → FULL_POS = 50，POS_CONVERSION_RATIO = 0.1953125（mm/count）

    对外 API 使用 width_m（m），内部换算：
        pos_mm = (1 - width_m / MAX_WIDTH_M) * STROKE_MM
        STROKE_MM = minPos - maxPos = 50 mm
        MAX_WIDTH_M = 0.05 m（50 mm 全开口）
    """

    MAX_WIDTH_M: float = 0.050   # Hand-E 全开口 50 mm = 0.05 m
    STROKE_MM:   float = 50.0    # minPos=50, maxPos=0

    def __init__(self,
                 port: str = "/dev/ttyUSB0",
                 min_pos_mm: float = 50.0,
                 max_pos_mm: float = 0.0):
        """
        port        : Modbus RTU 串口，默认 /dev/ttyUSB0
        min_pos_mm  : 夹爪关闭时的位置寄存器值（mm），默认 50
        max_pos_mm  : 夹爪打开时的位置寄存器值（mm），默认 0
        """
        self._min_pos_mm = min_pos_mm
        self._max_pos_mm = max_pos_mm
        self._stroke_mm  = abs(min_pos_mm - max_pos_mm)
        self._width_m:   float = self.MAX_WIDTH_M
        self._lock = threading.Lock()

        logger.info(f"初始化 Robotiq Hand-E 夹爪（串口={port}）…")
        handler = ServoHandler.__new__(ServoHandler)  # 先建对象，再自定义 port 初始化
        # ServoHandler.init() 硬编码 '/dev/ttyUSB0'，需绕过以支持自定义端口
        from reactive_diffusion_policy.real_world.robot.gripper.robotiq.HandE import HandEForRtu
        handler.hand = HandEForRtu(port)

        self._gripper = AdaptiveGripper(
            gripper=handler,
            minPos=min_pos_mm,    # 关闭位置（mm）
            maxPos=max_pos_mm,    # 打开位置（mm）
            maxDiff=25,
            maxDiffNum=100,
            speed=0,
        )
        # 初始打开夹爪
        self._gripper.open(block=True)
        logger.info("Robotiq Hand-E 夹爪初始化完成（已打开）")

    def _width_to_pos_mm(self, width_m: float) -> float:
        """将开口宽度（m）转换为夹爪位置（mm）"""
        ratio = 1.0 - min(max(width_m / self.MAX_WIDTH_M, 0.0), 1.0)
        return self._max_pos_mm + ratio * self._stroke_mm

    def move(self, width_m: float, speed: int = 150, force: int = 50):
        """移动夹爪到目标开口宽度（m）"""
        pos_mm = self._width_to_pos_mm(width_m)
        with self._lock:
            self._gripper.move(pos_mm, speed, force, block=False)
            self._width_m = width_m
            logger.debug(f"[HandE] 移动 → width={width_m:.4f}m  pos={pos_mm:.1f}mm")

    def grasp(self, force: int = 100):
        """关闭夹爪（force 控）"""
        with self._lock:
            self._gripper.close(speed=150, block=False)
            self._width_m = 0.0
            logger.debug("[HandE] 夹爪关闭（grasp）")

    def stop(self):
        """停止夹爪运动"""
        with self._lock:
            try:
                self._gripper._gripper.stop()
            except Exception as e:
                logger.warning(f"夹爪 stop 时出错（可忽略）：{e}")

    def get_state(self) -> List[float]:
        """
        返回夹爪当前状态 [width_m, force_N]。
        force_N 当前未从硬件读取，固定返回 0。
        """
        with self._lock:
            try:
                pos_mm = float(self._gripper.position())
                if self._stroke_mm > 0:
                    ratio = 1.0 - (pos_mm - self._max_pos_mm) / self._stroke_mm
                    self._width_m = max(0.0, min(ratio, 1.0)) * self.MAX_WIDTH_M
            except Exception as e:
                logger.warning(f"读取夹爪位置失败（使用缓存值）：{e}")
            return [self._width_m, 0.0]


# ──────────────────────────────────────────────
# UR5e 控制器
# ──────────────────────────────────────────────

# servoL 高频控制参数（24Hz 推荐值）
SERVO_VELOCITY     = 0.5    # m/s  — 关节速度上限
SERVO_ACCELERATION = 0.5    # m/s² — 关节加速度上限
SERVO_DT           = 1.0 / 24.0   # s — 控制周期（与 RDP 24Hz 一致）
SERVO_LOOKAHEAD    = 0.1    # s — 前瞻时间（类 D 项），推荐 0.03~0.2
SERVO_GAIN         = 300    # 位置增益（类 P 项），推荐 100~2000

GRIPPER_MAX_WIDTH_M    = 0.050   # Robotiq Hand-E 最大开口 50mm = 0.05m
GRIPPER_WIDTH_THRESHOLD = 0.025  # 宽度低于此值视为"需要关闭"
GRIPPER_PORT           = "/dev/ttyUSB0"  # Hand-E 串口（Modbus RTU）


class UR5eController:
    """
    封装 UR5e 的 RTDE 通信，
    提供与 Nova5Controller / FrankaServer 相同的控制接口。
    """

    def __init__(self,
                 robot_ip: str = "192.168.1.100",
                 servo_velocity: float = SERVO_VELOCITY,
                 servo_acceleration: float = SERVO_ACCELERATION,
                 servo_dt: float = SERVO_DT,
                 servo_lookahead: float = SERVO_LOOKAHEAD,
                 servo_gain: float = SERVO_GAIN,
                 gripper_port: str = GRIPPER_PORT,
                 gripper_min_pos_mm: float = 50.0,
                 gripper_max_pos_mm: float = 0.0):
        self.robot_ip = robot_ip
        self.servo_velocity     = servo_velocity
        self.servo_acceleration = servo_acceleration
        self.servo_dt           = servo_dt
        self.servo_lookahead    = servo_lookahead
        self.servo_gain         = servo_gain

        self._lock = threading.Lock()

        # ── 连接 RTDE 接口 ──────────────────────────────────────────
        logger.info(f"连接 UR5e RTDEControlInterface ({robot_ip})…")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)

        logger.info(f"连接 UR5e RTDEReceiveInterface ({robot_ip})…")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        if not self.rtde_c.isConnected():
            raise RuntimeError(f"UR5e RTDEControlInterface 连接失败 ({robot_ip})")
        if not self.rtde_r.isConnected():
            raise RuntimeError(f"UR5e RTDEReceiveInterface 连接失败 ({robot_ip})")
        logger.info("UR5e RTDE 连接成功")

        # ── 连接夹爪（Modbus RTU 串口）──────────────────────────────
        self.gripper = RobotiqGripperController(
            port=gripper_port,
            min_pos_mm=gripper_min_pos_mm,
            max_pos_mm=gripper_max_pos_mm,
        )
        time.sleep(0.5)

    # ── TCP 位姿获取 ──────────────────────────
    def get_current_tcp(self) -> List[float]:
        """
        获取当前 TCP 位姿。
        返回格式：[x_m, y_m, z_m, qw, qx, qy, qz]
        """
        pose = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        qw, qx, qy, qz = rotvec_to_quat(pose[3:6])
        return [pose[0], pose[1], pose[2], qw, qx, qy, qz]

    def get_tcp_vel(self) -> List[float]:
        """获取 TCP 速度 [vx, vy, vz, wx, wy, wz]（m/s, rad/s）"""
        vel = self.rtde_r.getActualTCPSpeed()  # [vx, vy, vz, vrx, vry, vrz]
        return list(vel)

    def get_tcp_wrench(self) -> List[float]:
        """获取 TCP 处外力 [fx, fy, fz, mx, my, mz]（N, Nm）"""
        wrench = self.rtde_r.getActualTCPForce()  # [fx, fy, fz, mx, my, mz]
        return list(wrench)

    def is_protective_stop(self) -> bool:
        """检查机器人是否处于保护停止状态"""
        safety_status = self.rtde_r.getSafetyStatusBits()
        # bit 3: protective stop
        return bool(safety_status & (1 << 3))

    def is_fault(self) -> bool:
        """检查是否存在故障（保护停止 / 紧急停止）"""
        safety_status = self.rtde_r.getSafetyStatusBits()
        # bit 3: protective stop, bit 5: robot emergency stop
        return bool(safety_status & ((1 << 3) | (1 << 5)))

    def clear_fault(self):
        """解除保护停止"""
        if self.rtde_c.isConnected():
            self.rtde_c.unlockProtectiveStop()
            logger.info("UR5e 保护停止已解除")

    # ── TCP 运动控制 ──────────────────────────
    def tcp_move(self, target_tcp_7d: List[float]):
        """
        使用 servoL 高频伺服控制 TCP 运动。
        target_tcp_7d: [x_m, y_m, z_m, qw, qx, qy, qz]

        servoL 适合 24Hz 高频调用，控制周期由 dt 指定。
        注意：UR servoL 使用旋转向量（轴角 rad），位置单位 m。
        """
        x, y, z, qw, qx, qy, qz = target_tcp_7d
        rotvec = quat_to_rotvec(qw, qx, qy, qz)
        pose_ur = [x, y, z, rotvec[0], rotvec[1], rotvec[2]]

        with self._lock:
            self.rtde_c.servoL(
                pose_ur,
                self.servo_velocity,
                self.servo_acceleration,
                self.servo_dt,
                self.servo_lookahead,
                self.servo_gain,
            )

    # ── 夹爪控制 ─────────────────────────────
    def gripper_move(self, width_m: float, speed: int = 150, force: int = 50):
        """移动夹爪到目标宽度（m），speed/force 均为 0~255"""
        self.gripper.move(width_m, speed=speed, force=force)

    def gripper_grasp(self, force: int = 100):
        """关闭夹爪（力控）"""
        self.gripper.grasp(force=force)

    def gripper_stop(self):
        """停止夹爪运动"""
        self.gripper.stop()

    def get_gripper_state(self) -> List[float]:
        return self.gripper.get_state()

    def go_home(self, home_joint_rad: List[float] = None):
        """
        关节空间回零（moveJ），使机器人回到安全初始姿态。
        home_joint_rad: 6 个关节角度（弧度），默认为竖直向上姿态。
        请根据实际工作空间修改默认值！

        UR5e 常用安全初始姿态（仅供参考）：
            [0, -π/2, 0, -π/2, 0, 0]  — 标准 "ready" 姿态
        """
        if home_joint_rad is None:
            home_joint_rad = [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]
        logger.info(f"UR5e 回零：关节角(rad) = {[f'{v:.3f}' for v in home_joint_rad]}")
        # moveJ(q, speed, acceleration, async)
        self.rtde_c.moveJ(home_joint_rad, 1.05, 1.4, False)

    def shutdown(self):
        """安全关闭 RTDE 连接"""
        try:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            logger.info("UR5e RTDE 连接已关闭")
        except Exception as e:
            logger.warning(f"关闭 RTDE 连接时出错（可忽略）：{e}")


# ──────────────────────────────────────────────
# FastAPI HTTP 服务器（与 Nova5Server / FlexivServer 接口完全一致）
# ──────────────────────────────────────────────

class UR5eServer:
    """
    UR5e HTTP Server

    提供与 Nova5Server / BimanualFlexivServer 完全相同的 REST API，
    供 RealRobotEnvironment 通过 HTTP 调用控制 UR5e 单臂机器人。
    """

    def __init__(self,
                 host_ip: str = "0.0.0.0",
                 port: int = 8093,
                 robot_ip: str = "192.168.1.100",
                 servo_velocity: float = SERVO_VELOCITY,
                 servo_acceleration: float = SERVO_ACCELERATION,
                 servo_dt: float = SERVO_DT,
                 servo_lookahead: float = SERVO_LOOKAHEAD,
                 servo_gain: float = SERVO_GAIN,
                 gripper_port: str = GRIPPER_PORT,
                 gripper_min_pos_mm: float = 50.0,
                 gripper_max_pos_mm: float = 0.0,
                 **kwargs):
        self.host_ip = host_ip
        self.port = port

        logger.info(f"正在连接 UR5e（{robot_ip}）…")
        self.robot = UR5eController(
            robot_ip=robot_ip,
            servo_velocity=servo_velocity,
            servo_acceleration=servo_acceleration,
            servo_dt=servo_dt,
            servo_lookahead=servo_lookahead,
            servo_gain=servo_gain,
            gripper_port=gripper_port,
            gripper_min_pos_mm=gripper_min_pos_mm,
            gripper_max_pos_mm=gripper_max_pos_mm,
        )

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/clear_fault")
        async def clear_fault() -> List[str]:
            if self.robot.is_fault():
                logger.warning("UR5e 处于保护/急停状态，正在解除…")
                self.robot.clear_fault()
                return ["UR5e fault cleared"]
            return ["No fault detected"]

        @self.app.get("/get_current_robot_states")
        async def get_current_robot_states() -> BimanualRobotStates:
            tcp_7d       = self.robot.get_current_tcp()
            tcp_vel      = self.robot.get_tcp_vel()
            tcp_wrench   = self.robot.get_tcp_wrench()
            gripper_state = self.robot.get_gripper_state()

            return BimanualRobotStates(
                leftRobotTCP=tcp_7d,
                rightRobotTCP=[0.0] * 7,        # 单臂：右臂填零
                leftRobotTCPVel=tcp_vel,
                rightRobotTCPVel=[0.0] * 6,
                leftRobotTCPWrench=tcp_wrench,
                rightRobotTCPWrench=[0.0] * 6,
                leftGripperState=gripper_state,
                rightGripperState=[0.0] * 2,
            )

        @self.app.get("/get_current_tcp/{robot_side}")
        async def get_current_tcp(robot_side: str) -> List[float]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return [0.0] * 7
            return self.robot.get_current_tcp()

        @self.app.post("/move_tcp/{robot_side}")
        async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right arm not available (UR5e single-arm), ignored."}

            self.robot.tcp_move(request.target_tcp)
            return {"message": f"UR5e servoL → {request.target_tcp}"}

        @self.app.post("/move_gripper/{robot_side}")
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available (single-arm), ignored."}

            # MoveGripperRequest.velocity / force_limit 转换为 Hand-E 整数参数（0~255）
            speed = min(int(getattr(request, "velocity", 10.0) * 15), 255)
            force = min(int(getattr(request, "force_limit", 5.0) * 20), 255)
            self.robot.gripper_move(
                width_m=request.width,
                speed=max(speed, 50),
                force=max(force, 20),
            )
            return {"message": f"Gripper → width={request.width:.4f}m"}

        @self.app.post("/move_gripper_force/{robot_side}")
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available, ignored."}

            force = min(int(getattr(request, "force_limit", 5.0) * 20), 255)
            self.robot.gripper_grasp(force=max(force, 20))
            return {"message": f"Gripper grasp (force={request.force_limit})"}

        @self.app.post("/stop_gripper/{robot_side}")
        async def stop_gripper(robot_side: str) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available, ignored."}

            self.robot.gripper_stop()
            return {"message": "UR5e gripper stopped"}

        @self.app.post("/go_home")
        async def go_home() -> Dict[str, str]:
            """让机器人回到预设安全初始关节角（moveJ）。
            推理前调用，确保机器人处于已知起始位姿。
            如需修改默认回零姿态，请在 UR5eController.go_home() 中修改 home_joint_rad。
            """
            self.robot.go_home()
            return {"message": "UR5e go_home done"}

    def run(self):
        logger.info(f"UR5e HTTP Server 启动：http://{self.host_ip}:{self.port}")
        try:
            uvicorn.run(self.app, host=self.host_ip, port=self.port, log_level="warning")
        finally:
            self.robot.shutdown()


def main():
    """
    启动 UR5e Server 的入口。

    示例：
        python -m reactive_diffusion_policy.real_world.robot.ur5e_server \\
            --robot_ip 192.168.1.100 --port 8093
    """
    parser = argparse.ArgumentParser(description="UR5e Robot HTTP Server")
    parser.add_argument("--host_ip",    type=str,   default="0.0.0.0")
    parser.add_argument("--port",       type=int,   default=8093)
    parser.add_argument("--robot_ip",   type=str,   default="192.168.1.100",
                        help="UR5e 机器人 IP 地址")
    parser.add_argument("--servo_velocity",    type=float, default=SERVO_VELOCITY,
                        help=f"servoL 速度上限 (m/s)，默认 {SERVO_VELOCITY}")
    parser.add_argument("--servo_acceleration",type=float, default=SERVO_ACCELERATION,
                        help=f"servoL 加速度上限 (m/s²)，默认 {SERVO_ACCELERATION}")
    parser.add_argument("--servo_dt",          type=float, default=SERVO_DT,
                        help=f"控制周期 (s)，默认 {SERVO_DT:.4f} (=1/24Hz)")
    parser.add_argument("--servo_lookahead",   type=float, default=SERVO_LOOKAHEAD,
                        help=f"前瞻时间 (s)，默认 {SERVO_LOOKAHEAD}")
    parser.add_argument("--servo_gain",        type=float, default=SERVO_GAIN,
                        help=f"位置增益，默认 {SERVO_GAIN}")
    parser.add_argument("--gripper_port",      type=str,   default=GRIPPER_PORT,
                        help=f"夹爪 Modbus RTU 串口，默认 {GRIPPER_PORT}")
    parser.add_argument("--gripper_min_pos_mm",type=float, default=50.0,
                        help="夹爪关闭位置（mm），默认 50（Hand-E 全闭）")
    parser.add_argument("--gripper_max_pos_mm",type=float, default=0.0,
                        help="夹爪打开位置（mm），默认 0（Hand-E 全开）")
    args = parser.parse_args()

    server = UR5eServer(
        host_ip=args.host_ip,
        port=args.port,
        robot_ip=args.robot_ip,
        servo_velocity=args.servo_velocity,
        servo_acceleration=args.servo_acceleration,
        servo_dt=args.servo_dt,
        servo_lookahead=args.servo_lookahead,
        servo_gain=args.servo_gain,
        gripper_port=args.gripper_port,
        gripper_min_pos_mm=args.gripper_min_pos_mm,
        gripper_max_pos_mm=args.gripper_max_pos_mm,
    )
    server.run()


if __name__ == "__main__":
    main()
