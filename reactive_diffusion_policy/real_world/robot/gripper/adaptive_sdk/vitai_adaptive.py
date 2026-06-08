"""
ViTai Adaptive SDK

Hardware:
    Two tactile sensors and one gripper.

Dependencies:
    pip install minimalmodbus pyserial
    pip install pyvitaisdk==1.0.12
"""

from threading import Event, Thread, Lock
from typing import Callable
import cv2
import time
import os
import sys
import numpy as np
import configparser

# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
# from .actuator_sdk import ActuatorSDK
from .changingtek_p_rtu_Servo import MotorController

from pyvitaisdk import VTSensor, VTSDeviceFinder, VTSDataType, VTSensorType
try:
    from pyvitaisdk import VTSlipState
except ImportError:
    # pyvitaisdk < 1.0.12 does not ship VTSlipState; define a stub so the
    # rest of the module (gripper open/close/move) still works.
    class VTSlipState:  # type: ignore
        INCIPIENT_SLIP = "INCIPIENT_SLIP"
        PARTIAL_SLIP   = "PARTIAL_SLIP"
        COMPLETE_SLIP  = "COMPLETE_SLIP"
        NO_OBJ         = "NO_OBJ"


# 夹爪传动比
gripper_ratio = 4.06
GRIPPER_CLOSE_POS = 9000

def pos_g(send_pos: float):
    "下发的pos * G2 = 夹爪实际获取的位置"
    return int(send_pos * gripper_ratio)

class ViTaiAdaptive:
    def __init__(self):
        # gripper parameters 
        self._port = 'COM6'
        self._slave_id = 1
        self._baudrate = 115200
        self._timeout = 0.5
        # 运动参数
        self._speed_pct = 100
        self._force_pct = 50
        self._accel = 2000
        self._decel = 2000
        # gripper位置参数
        self._open_pos = 0
        # 对应夹爪最大的开合读数2960/130mm = 23
        self._close_pos = pos_g(3000)
        self._gripper = None

        # tactile sensor parameters
        self._capGetOneFrameCallbacks = []
        self._minPos = GRIPPER_CLOSE_POS # 由夹爪最大闭合位置决定
        self._maxPos = 0 # 夹爪最大开合
        self._maxDiff = 15 # (单个MARK点)滤波阈值
        self._maxDiffNum = 50 # (所有MARK点之和)触发阈值
        self._openEvent, self._closeEvent = Event(), Event()
        self._openThread, self._closeThread = None, None
        self._showrecoredFrame1, self._showRecordFrame2 = None, None
        self._showFrame1, self._showFrame2 = None, None
        self._diffFrame1, self._diffFrame2 = None, None
        self._state = False
        self._adaptiveState = False
        self._cap1 = None # 触觉传感器
        self._cap2 = None

        self.init_gripper = None
        self.init_sensor = None

        # 30Hz检测线程相关变量
        self._tighten_check_thread = None  # 收紧检测线程
        self._tighten_check_stop = Event() # 停止检测事件
        self._tighten_check_running = False # 检测运行状态标记


    def read_config(self, config_file:str):
        # 初始化配置解析器
        config = configparser.ConfigParser()
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            print(f"Config file not found ：{config_file}")
            return -1
        
        config.read(config_file, encoding='utf-8')

        try:
            db_gripper_port = config.get("gripper", "port") 
            db_gripper_force = config.getint("gripper", "force")
            db_gripper_speed = config.getint("gripper", "speed")
            db_gripper_scale = config.getint("gripper", "scale")
            print(f"Read config file: Gripper port: {db_gripper_port}, speed: {db_gripper_speed}, force: {db_gripper_force}, scale: {db_gripper_scale}")
            db_gripper_speed = max(0, min(100, db_gripper_speed))
            if db_gripper_force < 0:
                db_gripper_force = 0
            if db_gripper_force > 100:
                db_gripper_force = 100
            self._port = db_gripper_port
            self._speed_pct = db_gripper_speed
            self._force_pct = db_gripper_force
            self._minPos = db_gripper_scale*100
        except Exception as e:
            print(f"Read config file error: {e}")
            return -2

        print(f"Actual config: Gripper port: {self._port}, force: {self._force_pct}")
        return 0    


    def initGripper(self):
        """
        init gripper

        Returns:
            int: 0-success, -1-fail
        """
        try:
            self._gripper = MotorController(self._port, self._slave_id, self._baudrate, self._timeout)
            # 设置运动参数
            self._gripper.set_target_speed(self._speed_pct)
            self._gripper.set_target_force(self._force_pct)
            self._gripper.set_target_acceleration(self._accel)
            self._gripper.set_target_deceleration(self._decel)
            self.init_gripper = True
            return 0
        except Exception as e:
            print(e)
            self.init_gripper = False
            return -1

    def initTactileSensors(self):
        """
        init tactile sensors

        Returns:
            int: 0-success, -1-sensor not found,  -2-fail
        """
        try:
            finder = VTSDeviceFinder()
            devices = finder.get_devices()
            if len(devices) != 2:
                print("Please connect two tactile sensors.")
                self.init_sensor = False
                return -1
            serial_numbers = finder.get_sns()
            print(f"Tactile Sensors: {serial_numbers}")

            self._cap1 = VTSensor(devices[0])
            self._cap1.calibrate(1)
            self._cap2 = VTSensor(devices[1])
            self._cap2.calibrate(1)

            for cap in [self._cap1, self._cap2]:
                if cap.sensor_type == VTSensorType.GF220:
                    self._maxDiff = 15
                    self._maxDiffNum = 50
                else:
                    self._maxDiff = 25
                    self._maxDiffNum = 100
            print(f'maxDiff:{self._maxDiff}, maxDiffNum:{self._maxDiffNum}')

            # 注册视触觉视频流
            self.registerCapGetFrame(self._cap1.get_warped_frame)
            self.registerCapGetFrame(self._cap2.get_warped_frame)

            self.init_sensor = True
            return 0
        except Exception as e:
            self.init_sensor = False
            return -2

    
    def registerCapGetFrame(self, callback:Callable):
        """
        register the callback functions for get one vaild frame from vision tactail camera.
        you can register multiple callback functions when there are multiple sensors

        Args:
            callback (Callable): _description_
        """
        self._capGetOneFrameCallbacks.append(callback)
        if len(self._capGetOneFrameCallbacks) > 2:
            print("You have registered more than two getFrame func for cap, please confirm whether the quantity is correct.")


    def isConnected(self):
        """
        check whether the gripper and tactile sensors are initialized successfully

        Returns:
            int: 0-success, -1-sensor init fail,  -2-gripper init fail
        """
        if self.init_gripper == True and self.init_sensor == True:
            return 0
        
        if self.init_sensor == False:
            return -1
        if self.init_gripper == False:
            return -2


    def adaptiveClose(self):
        """
        Adaptive closing

            the gripper will move in the direction of the finger closing, and auto stop after detecting the object.

        Args:
            block (bool, optional): When set to True, the function will return after the object is detected and the control claw stops. Otherwise, it will return immediately, and the detection will run in the background as a thread. Defaults to True.
            maxPos (float, optional): You can set the target open max pos, only this call takes effect. Defaults to None.
            maxDiff (int, optional): You can set the max diff value for images detected before and after, only this call takes effect. Defaults to None.
            maxDiffNum (int, optional): You can set the max diff num value for images detected before and after, only this call takes effect. Defaults to None.
        """
        if self.init_gripper != True or self.init_sensor != True:
            print("Gripper or tactile sensors not initialized successfully, please check the connection.")
            return -1

        # 停止已有30Hz检测线程
        self._stop_tighten_check()

        block = True
        minPos = None
        maxDiff = None
        maxDiffNum = None

        self._adaptiveState = True
        self.stop()
        pos = self._minPos if minPos is None else minPos
        diff = self._maxDiff if maxDiff is None else maxDiff
        diffNum = self._maxDiffNum if maxDiffNum is None else maxDiffNum

        # stop the open thread
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()
            self._openThread = None

        # 记录初始帧
        recordFrames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
        
        self.move_servo(pos)
        # print('--- ', pos)
        def adaptiveMove():
            while True:
                frames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
                absDiffs = list(map(cv2.absdiff, frames, recordFrames))
                absDiffNums = [np.sum(absDiff >= diff) for absDiff in absDiffs]
                # print(f"AdaptiveClose:{diff},{absDiffNums}")
                if all(absDiffNum >= diffNum for absDiffNum in absDiffNums):
                # if sum(absDiffNums) >= 100:
                    self.stop()
                    print(f"AdaptiveClose detected: {diff},{absDiffNums}")
                    self._state = True
                    break

                if abs(self.get_position() - pos) < 20:
                    time.sleep(0.2)
                    self.stop()
                    print("AdaptiveClose detected limit")
                    self._state = True
                    break

                if self._closeEvent.is_set():
                    print("AdaptiveClose _closeEvent")
                    break
                time.sleep(0.001)
            self._adaptiveState = False
            # 自适应闭合完成后，启动30Hz收紧检测
            self._start_tighten_check()
            print("=== 30Hz检测线程启动完成 ===")
                     
        if block:
            adaptiveMove()
            print(f"AdaptiveClose -- : {self.get_position()}")
            return 0
        else:
            # todo: the close thread will duplicate creation, please confirm whether only one close thread is allowed to run.
            self._closeThread = Thread(target=adaptiveMove, args=(),name="Adaptive Gripper Close", daemon=True)
            self._closeThread.start()
            # 非阻塞模式下直接返回，自适应完成后由子线程启动30Hz检测
            print(f"else AdaptiveClose -- : {self.get_position()}")

    def _position_reached(self, target_pos, wait_time=0.2):
        pos1 = self._gripper.read_real_position()
        time.sleep(wait_time)
        pos2 = self._gripper.read_real_position()
        if abs(pos2 - target_pos) < 3 or abs(pos2 - pos1) < 3:
            return True
        return False
    
    def move(self, pos:int):
        """
        control gripper move (阻塞式)

        Args:
            pos (int): [0, 2960]
        """
        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1
        
        # 停止30Hz检测线程
        self._stop_tighten_check()

        if pos < 0:
            return
        
        max_wait_time = 2.0 
        start_time = time.time() 
        
        self._gripper.set_target_position(pos)
        self._gripper.trigger_motion()
        
        while not self._position_reached(pos):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_wait_time:
                break 
            time.sleep(0.01)
        return 0
    
    def move_servo(self, pos:int):
        """
        control gripper move (非阻塞式)

        Args:
            pos (int): [0, 2960]
        """
        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1
        
        # 停止30Hz检测线程
        self._stop_tighten_check()

        if pos < 0:
            return
        self._gripper.set_target_position(pos)
        self._gripper.trigger_motion()

        return 0


    def open(self):
        """
        control gripper open to the max position (阻塞式)

        Returns:
            int: 0-success, -1-fail
        """
        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1
        
        # 停止30Hz检测线程
        self._stop_tighten_check()

        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1

        self.move(self._open_pos)
        return 0
    

    def close(self):
        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1
        
        # 停止30Hz检测线程
        self._stop_tighten_check()

        self.move(self._minPos)
        return 0


    def get_position(self):
        return self._gripper.read_real_position()
    

    def stop(self):
        if self.init_gripper != True:
            print("Gripper not initialized successfully, please check the connection.")
            return -1
        
        # 停止30Hz检测线程
        self._stop_tighten_check()

        position = self.get_position()
        self._gripper.set_target_position(position)
        self._gripper.trigger_motion()

        return 0


    def checkGripTighten(self, tighten_step: int = 1):
        """
        夹持检测：检测到滑动则自动收紧夹爪(配合adaptiveClose接口使用，该接口需要在自适应夹取后一直检测)

        Args:
            tighten_step (int): 每次收紧夹爪的位置步长（单位：mm）
        """
        if not isinstance(tighten_step, int) or tighten_step < 1:
            print(f"无效的收紧步长：{tighten_step}，要求为≥1的整数")
            return
    
        close_pos = 23 * tighten_step # 2960/130mm = 23

        # 定义滑动状态集合
        SLIP_STATES = {
            VTSlipState.INCIPIENT_SLIP,
            VTSlipState.PARTIAL_SLIP,
            VTSlipState.COMPLETE_SLIP
        }

        datatypes = [
            VTSDataType.WARPED_IMG,
            VTSDataType.DEPTH_MAP,
            VTSDataType.SLIP_STATE,
            VTSDataType.MARKER_OFFSET_VECTOR
        ]
        
        data1 = self._cap1.collect_sensor_data(*datatypes)
        data2 = self._cap2.collect_sensor_data(*datatypes)
        slip_state1 = data1[VTSDataType.SLIP_STATE]
        slip_state2 = data2[VTSDataType.SLIP_STATE]

        if slip_state1 in SLIP_STATES or slip_state2 in SLIP_STATES:
            cur_pos = self.get_position()
            target_pos = cur_pos + close_pos
            print("Slip detected! cur pose: ", cur_pos, " -> target Pos:", target_pos)
            if target_pos >= self._minPos+22:
                return
            self._moveImpl(target_pos)


    def isGrippered(self):
        """
        check whether the object is grippered.

        Returns:
            int: 0-object is grippered, 1-object is not grippered, -1-sensor or gripper init fail
        """
        if self.init_gripper != True or self.init_sensor != True:
            print("Gripper or tactile sensors not initialized successfully, please check the connection.")
            return -1
        
        datatypes = [
            VTSDataType.WARPED_IMG,
            VTSDataType.DEPTH_MAP,
            VTSDataType.SLIP_STATE,
            VTSDataType.MARKER_OFFSET_VECTOR
        ]
        
        data1 = self._cap1.collect_sensor_data(*datatypes)
        data2 = self._cap2.collect_sensor_data(*datatypes)
        slip_state1 = data1[VTSDataType.SLIP_STATE]
        slip_state2 = data2[VTSDataType.SLIP_STATE]

        if slip_state1 == VTSlipState.NO_OBJ and slip_state2 == VTSlipState.NO_OBJ:
            return 1
        else:
            return 0

    def release(self):
        # 停止30Hz检测线程
        self._stop_tighten_check()

        self._cap1.release()
        self._cap2.release()


    def status(self):
        """
        Get the status of the gripper and tactile sensors.

        Returns:
            int: 0-normal, -1-sensor fault,  -2-gripper fault
        """
        if self.init_gripper == True and self.init_sensor == True:
            return 0
        
        if self.init_sensor == False:
            return -1
        if self.init_gripper == False:
            return -2
        

    ################################################################
    def _tighten_check_loop(self):
        """30Hz循环（增加try-except，避免线程崩溃）"""
        interval = 1.0 / 30
        print("进入30Hz检测循环")  # 新增：确认线程启动
        while not self._tighten_check_stop.is_set():
            try:
                start_time = time.time()
                self.checkGripTighten()
                elapsed = time.time() - start_time
                time.sleep(max(0, interval - elapsed))
            except Exception as e:
                print(f"检测循环异常：{e}")
                time.sleep(0.001)  # 异常时短休眠，避免CPU占满
        print("30Hz检测循环退出")
        self._tighten_check_running = False


    def _start_tighten_check(self):
        """启动线程（无锁，简化逻辑）"""
        # 1. 先停止旧线程（简化版，无锁）
        self._stop_tighten_check()
        
        # 2. 重置停止事件
        self._tighten_check_stop.clear()
        
        # 3. 创建新线程（daemon=True，不阻塞主线程退出）
        self._tighten_check_thread = Thread(
            target=self._tighten_check_loop,
            name="TightenCheck",
            daemon=True
        )
        self._tighten_check_running = True
        self._tighten_check_thread.start()
        
        # 强制打印启动日志（不管后续是否异常）
        print("=== Tighten check thread started (30Hz) ===")

    def _stop_tighten_check(self):
        """停止线程（无锁，无阻塞）"""
        if self._tighten_check_running and self._tighten_check_thread is not None:
            self._tighten_check_stop.set()
            # 去掉join()，避免阻塞主线程！
            if self._tighten_check_thread and self._tighten_check_thread.is_alive():
                self._tighten_check_thread.join(timeout=1.0)
            self._tighten_check_thread = None
            self._tighten_check_running = False
            print("=== Tighten check thread stopped ===")


    def _moveImpl(self, pos:int):
        """
        control gripper move (阻塞式)

        Args:
            pos (int): [0, 2960]
        """
        max_wait_time = 2.0 
        start_time = time.time() 
        
        # self._gripper.set_target_force(self._force_pct)
        self._gripper.set_target_position(pos)
        self._gripper.trigger_motion()
        
        while not self._position_reached(pos):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_wait_time:
                break 
            time.sleep(0.01)
        return 0