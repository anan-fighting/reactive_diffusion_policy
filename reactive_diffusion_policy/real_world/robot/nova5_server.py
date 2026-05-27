"""
DOBOT Nova 5 Robot HTTP Server
==============================
提供与 bimanual_flexiv_server.py 完全相同的 REST API 接口，
底层通过官方 TCP-IP-Python-V4 SDK 驱动 DOBOT Nova 5 机械臂。

SDK 位置：
    /home/zzw/project/reactive_diffusion_policy/robot/TCP-IP-Python-V4-main/

通信端口说明（DOBOT V4 协议）：
    29999 → Dashboard 控制指令（EnableRobot / ClearError / GetPose / MovJ / ServoP 等）
    30004 → 实时状态反馈（ToolVectorActual / RobotMode / 关节角等，~200Hz）

坐标系约定：
    DOBOT 内部：(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)  ← 欧拉角，单位 mm/度
    本接口对外：(x_m,  y_m,  z_m,  qw, qx, qy, qz)          ← 四元数，单位 m
    → 本服务器在两者之间自动转换

夹爪控制：
    使用 DOBOT 数字输出 DO(index=1) 控制气动/电动夹爪
    DO=1(ON)  → 夹爪关闭（抓取）
    DO=0(OFF) → 夹爪打开
    如使用 Robotiq 等串行夹爪，可替换 _gripper_move() 实现
"""

import sys
import os
import re
import time
import math
import threading
from typing import List, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

# 将 DOBOT SDK 目录加入 Python 路径
_SDK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    'robot', 'TCP-IP-Python-V4-main'
)
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from dobot_api import DobotApiDashboard, DobotApiFeedBack  # noqa: E402

from reactive_diffusion_policy.common.data_models import (
    BimanualRobotStates,
    MoveGripperRequest,
    TargetTCPRequest,
)

# ──────────────────────────────────────────────
# 坐标转换工具函数
# ──────────────────────────────────────────────

def euler_deg_to_quat(rx_deg: float, ry_deg: float, rz_deg: float):
    """
    ZYX 欧拉角（度）→ 四元数 (qw, qx, qy, qz)
    DOBOT 使用 Rz-Ry-Rx (ZYX) 外旋欧拉角
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return qw, qx, qy, qz


def quat_to_euler_deg(qw: float, qx: float, qy: float, qz: float):
    """
    四元数 (qw, qx, qy, qz) → ZYX 欧拉角（度）
    返回 (rx_deg, ry_deg, rz_deg)
    """
    # 归一化
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    # ZYX: Rz→Ry→Rx
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    ry = math.asin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    rz = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def parse_pose_from_response(resp: str) -> Optional[List[float]]:
    """
    解析 GetPose() 返回字符串，提取 [x, y, z, rx, ry, rz]（mm, 度）
    典型返回格式："{0,{146.38,-283.43,332.40,177.79,-1.85,147.58},GetPose()}"
    """
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', resp)
    if len(nums) >= 7:
        # 第一个数通常是错误码(0)，后6个是位姿
        floats = [float(x) for x in nums]
        for i in range(len(floats) - 5):
            # 找到6个连续的合理数值（位置在-2000~2000mm，旋转在-360~360度）
            candidate = floats[i:i+6]
            if all(-2000 <= v <= 2000 for v in candidate):
                return candidate
    return None


def parse_result_id(resp: str) -> List[int]:
    """解析命令返回值中的数字，第一个为错误码（0=成功）"""
    if "Not Tcp" in resp:
        logger.error("Control Mode Is Not TCP!")
        return [1]
    nums = re.findall(r'-?\d+', resp)
    return [int(n) for n in nums] if nums else [2]


# ──────────────────────────────────────────────
# DOBOT Nova 5 控制器
# ──────────────────────────────────────────────

# Nova 5 ServoP 参数（高频控制 24Hz 推荐值）
SERVO_T = 1.0 / 24.0        # 控制周期 ~0.0417s
SERVO_AHEADTIME = 50.0      # 前瞻时间（类 PID D项），推荐 20~100
SERVO_GAIN = 500.0          # 位置增益（类 PID P项），推荐 200~1000

# 夹爪 DO 端口索引（按实际接线修改）
GRIPPER_DO_INDEX = 1        # DO1 控制夹爪
GRIPPER_CLOSE_STATUS = 1    # DO=1 → 关闭（抓取）
GRIPPER_OPEN_STATUS = 0     # DO=0 → 打开
GRIPPER_MAX_WIDTH_M = 0.085  # 夹爪最大开口（m），用于归一化
GRIPPER_WIDTH_THRESHOLD = 0.04  # 宽度低于此值视为"需要关闭"


class Nova5Controller:
    """
    封装 DOBOT Nova 5 的 Dashboard + Feedback 通信，
    提供与 UR5e / Flexiv 相同的控制接口。
    """

    DASHBOARD_PORT = 29999
    FEEDBACK_PORT = 30004

    def __init__(self,
                 robot_ip: str = "192.168.5.1",
                 servo_t: float = SERVO_T,
                 servo_aheadtime: float = SERVO_AHEADTIME,
                 servo_gain: float = SERVO_GAIN):
        self.robot_ip = robot_ip
        self.servo_t = servo_t
        self.servo_aheadtime = servo_aheadtime
        self.servo_gain = servo_gain

        self._lock = threading.Lock()
        self._feedback_lock = threading.Lock()

        # 连接 Dashboard 控制端口
        logger.info(f"连接 DOBOT Nova 5 Dashboard ({robot_ip}:{self.DASHBOARD_PORT})…")
        self.dashboard = DobotApiDashboard(robot_ip, self.DASHBOARD_PORT)

        # 连接反馈端口
        logger.info(f"连接 DOBOT Nova 5 Feedback ({robot_ip}:{self.FEEDBACK_PORT})…")
        self.feedback = DobotApiFeedBack(robot_ip, self.FEEDBACK_PORT)

        # 缓存最新反馈数据（后台线程持续更新）
        self._latest_feedback = None
        self._gripper_width_m: float = GRIPPER_MAX_WIDTH_M

        # 启动反馈数据后台线程
        self._feedback_thread = threading.Thread(
            target=self._feedback_loop, daemon=True)
        self._feedback_thread.start()

        # 使能机器人
        # 先尝试清除报警（上次异常退出可能留有报警）
        self.dashboard.ClearError()
        time.sleep(0.5)

        logger.info("使能 DOBOT Nova 5…")
        resp = self.dashboard.EnableRobot()
        code = parse_result_id(resp)
        if code[0] != 0:
            # 机器人可能已经处于使能状态，尝试查询当前模式
            mode_resp = self.dashboard.RobotMode()
            mode_nums = parse_result_id(mode_resp)
            # RobotMode: 5=空闲(可操作), 7=运动中, 4=禁用, 9=报警
            robot_mode = mode_nums[1] if len(mode_nums) > 1 else -1
            if robot_mode not in (5, 7):
                raise RuntimeError(
                    f"DOBOT EnableRobot 失败（返回：{resp}），"
                    f"当前 RobotMode={robot_mode}，请检查机器人状态。"
                )
            logger.info(f"DOBOT Nova 5 已处于可操作状态（RobotMode={robot_mode}），跳过使能。")
        else:
            logger.info("DOBOT Nova 5 使能成功")

        # 打开夹爪
        self._gripper_set(GRIPPER_OPEN_STATUS)
        time.sleep(0.5)

    # ── 反馈线程 ──────────────────────────────
    def _feedback_loop(self):
        """后台线程：持续读取 DOBOT 30004 端口反馈数据"""
        while True:
            try:
                data = self.feedback.feedBackData()
                if data is not None:
                    with self._feedback_lock:
                        self._latest_feedback = data
            except Exception as e:
                logger.warning(f"反馈数据读取异常（可忽略）：{e}")
                time.sleep(0.01)

    def _get_feedback(self):
        with self._feedback_lock:
            return self._latest_feedback

    # ── TCP 位姿获取 ──────────────────────────
    def get_current_tcp(self) -> List[float]:
        """
        获取当前 TCP 位姿。
        返回格式：[x_m, y_m, z_m, qw, qx, qy, qz]
        """
        # 优先从实时反馈数据获取（延迟更低）
        fb = self._get_feedback()
        if fb is not None:
            try:
                tv = fb['ToolVectorActual'][0]  # [x, y, z, rx, ry, rz] (mm, 度)
                x_m = tv[0] / 1000.0
                y_m = tv[1] / 1000.0
                z_m = tv[2] / 1000.0
                qw, qx, qy, qz = euler_deg_to_quat(tv[3], tv[4], tv[5])
                return [x_m, y_m, z_m, qw, qx, qy, qz]
            except Exception:
                pass

        # 备用：通过 Dashboard GetPose 获取
        resp = self.dashboard.GetPose()
        pose = parse_pose_from_response(resp)
        if pose is None:
            logger.warning(f"GetPose 解析失败，返回零位姿。原始响应：{resp}")
            return [0.0] * 7
        x_m, y_m, z_m = pose[0]/1000.0, pose[1]/1000.0, pose[2]/1000.0
        qw, qx, qy, qz = euler_deg_to_quat(pose[3], pose[4], pose[5])
        return [x_m, y_m, z_m, qw, qx, qy, qz]

    def get_tcp_vel(self) -> List[float]:
        """获取 TCP 速度 [vx, vy, vz, wx, wy, wz]（m/s, rad/s）"""
        fb = self._get_feedback()
        if fb is not None:
            try:
                vel = fb['TCPSpeedActual'][0]  # [vx, vy, vz, wx, wy, wz] mm/s, deg/s
                return [vel[0]/1000.0, vel[1]/1000.0, vel[2]/1000.0,
                        math.radians(vel[3]), math.radians(vel[4]), math.radians(vel[5])]
            except Exception:
                pass
        return [0.0] * 6

    def get_tcp_wrench(self) -> List[float]:
        """获取 TCP 处外力 [fx, fy, fz, mx, my, mz]（N, Nm）"""
        fb = self._get_feedback()
        if fb is not None:
            try:
                force = fb['ActualTCPForce'][0]  # [fx, fy, fz, mx, my, mz]
                return list(force)
            except Exception:
                pass
        return [0.0] * 6

    def is_fault(self) -> bool:
        """检查机器人是否处于报警/故障状态（RobotMode=9 表示报警）"""
        fb = self._get_feedback()
        if fb is not None:
            try:
                mode = int(fb['RobotMode'][0])
                # RobotMode: 1=初始化, 4=禁用, 5=空闲, 7=运动, 9=报警, 11=暂停
                return mode == 9
            except Exception:
                pass
        return False

    def clear_fault(self):
        """清除报警"""
        resp = self.dashboard.ClearError()
        code = parse_result_id(resp)
        if code[0] == 0:
            logger.info("DOBOT 报警已清除")
        else:
            logger.error(f"清除报警失败：{resp}")

    # ── TCP 运动控制 ──────────────────────────
    def tcp_move(self, target_tcp_7d: List[float]):
        """
        使用 ServoP 高频伺服控制 TCP 运动。
        target_tcp_7d: [x_m, y_m, z_m, qw, qx, qy, qz]

        ServoP 适合 24Hz 高频调用，参数 t 对应控制周期。
        注意：DOBOT ServoP 使用 ZYX 欧拉角（度），位置单位 mm。
        """
        x_m, y_m, z_m, qw, qx, qy, qz = target_tcp_7d

        # m → mm
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        z_mm = z_m * 1000.0

        # 四元数 → ZYX 欧拉角（度）
        rx_deg, ry_deg, rz_deg = quat_to_euler_deg(qw, qx, qy, qz)

        with self._lock:
            self.dashboard.ServoP(
                x_mm, y_mm, z_mm,
                rx_deg, ry_deg, rz_deg,
                t=self.servo_t,
                aheadtime=self.servo_aheadtime,
                gain=self.servo_gain,
            )

    # ── 夹爪控制 ─────────────────────────────
    def _gripper_set(self, status: int):
        """底层 DO 控制夹爪（status=1关/status=0开）"""
        resp = self.dashboard.DO(GRIPPER_DO_INDEX, status)
        code = parse_result_id(resp)
        if code[0] != 0:
            logger.warning(f"夹爪 DO 指令失败：{resp}")

    def gripper_move(self, width_m: float, velocity: float = 10.0, force_limit: float = 5.0):
        """
        控制夹爪到目标宽度（米）。
        宽度 < GRIPPER_WIDTH_THRESHOLD → 关闭（抓取）
        宽度 >= GRIPPER_WIDTH_THRESHOLD → 打开
        """
        if width_m < GRIPPER_WIDTH_THRESHOLD:
            self._gripper_set(GRIPPER_CLOSE_STATUS)
            self._gripper_width_m = 0.0
            logger.debug(f"夹爪关闭（目标宽度 {width_m:.4f}m < 阈值 {GRIPPER_WIDTH_THRESHOLD}m）")
        else:
            self._gripper_set(GRIPPER_OPEN_STATUS)
            self._gripper_width_m = GRIPPER_MAX_WIDTH_M
            logger.debug(f"夹爪打开（目标宽度 {width_m:.4f}m）")

    def gripper_grasp(self, force_limit: float = 5.0):
        """强制关闭夹爪（力控模式，DOBOT 使用 DO 实现）"""
        self._gripper_set(GRIPPER_CLOSE_STATUS)
        self._gripper_width_m = 0.0

    def gripper_stop(self):
        """停止夹爪（保持当前状态）"""
        pass  # DOBOT DO 控制无速度概念，无需实现

    def get_gripper_state(self) -> List[float]:
        """返回夹爪状态 [width_m, force_N]
        注意：DOBOT 通过 DO 数字输出控制夹爪（无位置传感器），
        width_m 为软件估值：关闭→0.0，打开→GRIPPER_MAX_WIDTH_M。
        """
        return [self._gripper_width_m, 0.0]

    def go_home(self, home_joint_deg: List[float] = None):
        """
        关节空间回零（MovJ），使机器人回到安全初始姿态。
        home_joint_deg: 6个关节角度（度），默认 [0, 0, 90, 0, 90, 0]（竖直向上姿态）。
        请根据实际工作空间修改默认值！
        """
        if home_joint_deg is None:
            home_joint_deg = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        j1, j2, j3, j4, j5, j6 = home_joint_deg
        logger.info(f"Nova 5 回零：关节角 = {home_joint_deg}")
        resp = self.dashboard.MovJ(j1, j2, j3, j4, j5, j6, coordinateMode=0)
        code = parse_result_id(resp)
        if code[0] != 0:
            logger.warning(f"MovJ 回零指令返回异常：{resp}")
        return resp

    def shutdown(self):
        """安全关闭连接"""
        try:
            self.dashboard.close()
            self.feedback.close()
            logger.info("DOBOT Nova 5 连接已关闭")
        except Exception as e:
            logger.warning(f"关闭连接时出错（可忽略）：{e}")


# ──────────────────────────────────────────────
# FastAPI HTTP 服务器（与 Flexiv/UR5e 接口完全一致）
# ──────────────────────────────────────────────

class Nova5Server:
    """
    DOBOT Nova 5 HTTP Server

    提供与 BimanualFlexivServer / UR5eServer 完全相同的 REST API，
    供 RealRobotEnvironment 通过 HTTP 调用控制 DOBOT Nova 5 单臂机器人。
    """

    def __init__(self,
                 host_ip: str = "0.0.0.0",
                 port: int = 8092,
                 robot_ip: str = "192.168.5.1",
                 servo_t: float = SERVO_T,
                 servo_aheadtime: float = SERVO_AHEADTIME,
                 servo_gain: float = SERVO_GAIN,
                 **kwargs):
        self.host_ip = host_ip
        self.port = port

        logger.info(f"正在连接 DOBOT Nova 5（{robot_ip}）…")
        self.robot = Nova5Controller(
            robot_ip=robot_ip,
            servo_t=servo_t,
            servo_aheadtime=servo_aheadtime,
            servo_gain=servo_gain,
        )

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/clear_fault")
        async def clear_fault() -> List[str]:
            if self.robot.is_fault():
                logger.warning("DOBOT Nova 5 处于报警状态，正在清除…")
                self.robot.clear_fault()
                return ["DOBOT Nova 5 fault cleared"]
            return ["No fault detected"]

        @self.app.get("/get_current_robot_states")
        async def get_current_robot_states() -> BimanualRobotStates:
            tcp_7d = self.robot.get_current_tcp()
            tcp_vel = self.robot.get_tcp_vel()
            tcp_wrench = self.robot.get_tcp_wrench()
            gripper_state = self.robot.get_gripper_state()

            return BimanualRobotStates(
                leftRobotTCP=tcp_7d,
                rightRobotTCP=[0.0] * 7,       # 单臂：右臂填零
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
                return {"message": "Right arm not available (DOBOT Nova 5 single-arm), ignored."}

            self.robot.tcp_move(request.target_tcp)
            return {"message": f"DOBOT Nova 5 ServoP → {request.target_tcp}"}

        @self.app.post("/move_gripper/{robot_side}")
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available (single-arm), ignored."}

            self.robot.gripper_move(
                width_m=request.width,
                velocity=request.velocity,
                force_limit=request.force_limit,
            )
            return {"message": f"Gripper → width={request.width:.4f}m"}

        @self.app.post("/move_gripper_force/{robot_side}")
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available, ignored."}

            self.robot.gripper_grasp(force_limit=request.force_limit)
            return {"message": f"Gripper grasp (force={request.force_limit})"}

        @self.app.post("/stop_gripper/{robot_side}")
        async def stop_gripper(robot_side: str) -> Dict[str, str]:
            if robot_side not in ("left", "right"):
                raise HTTPException(status_code=400, detail="robot_side 应为 'left' 或 'right'")
            if robot_side == "right":
                return {"message": "Right gripper not available, ignored."}

            self.robot.gripper_stop()
            return {"message": "DOBOT Nova 5 gripper stopped"}

        @self.app.post("/go_home")
        async def go_home() -> Dict[str, str]:
            """让机器人回到预设安全初始关节角（MovJ）。
            推理前调用，确保机器人处于已知起始位姿。
            如需修改默认回零姿态，请在 Nova5Controller.go_home() 中修改 home_joint_deg。
            """
            resp = self.robot.go_home()
            return {"message": f"Nova 5 go_home → {resp}"}

    def run(self):
        logger.info(f"DOBOT Nova 5 HTTP Server 启动：http://{self.host_ip}:{self.port}")
        try:
            uvicorn.run(self.app, host=self.host_ip, port=self.port, log_level="warning")
        finally:
            self.robot.shutdown()


def main():
    """
    启动 DOBOT Nova 5 Server 的入口。

    示例：
        python -m reactive_diffusion_policy.real_world.robot.nova5_server \
            --robot_ip 192.168.5.1 --port 8092
    """
    import argparse
    parser = argparse.ArgumentParser(description="DOBOT Nova 5 Robot HTTP Server")
    parser.add_argument("--host_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--robot_ip", type=str, default="192.168.5.1",
                        help="DOBOT Nova 5 机器人 IP 地址（默认 192.168.5.1）")
    parser.add_argument("--servo_t", type=float, default=SERVO_T,
                        help=f"ServoP 控制周期（秒），默认 {SERVO_T:.4f}（=1/24Hz）")
    parser.add_argument("--servo_aheadtime", type=float, default=SERVO_AHEADTIME,
                        help=f"ServoP 前瞻时间，默认 {SERVO_AHEADTIME}，范围 [20, 100]")
    parser.add_argument("--servo_gain", type=float, default=SERVO_GAIN,
                        help=f"ServoP 位置增益，默认 {SERVO_GAIN}，范围 [200, 1000]")
    args = parser.parse_args()

    server = Nova5Server(
        host_ip=args.host_ip,
        port=args.port,
        robot_ip=args.robot_ip,
        servo_t=args.servo_t,
        servo_aheadtime=args.servo_aheadtime,
        servo_gain=args.servo_gain,
    )
    server.run()


if __name__ == "__main__":
    main()
