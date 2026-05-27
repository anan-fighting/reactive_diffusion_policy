import logging
import warnings
from threading import Event, Thread
from typing import Callable
from .HandE import HandEForRtu

import cv2
from abc import ABC, abstractmethod
import numpy as np


ESC = 27


class BaseServoHandler(ABC):
    def __init__(self):
        self.init()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def mov(self, pos, block):
        pass

    @abstractmethod
    def move(self, pos, speed, force, block=True):
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    @abstractmethod
    def position(self):
        return

class ServoHandler(BaseServoHandler):
    def init(self):
        self.hand = HandEForRtu('/dev/ttyUSB0')

    def mov(self, pos, block):
        self.hand.move(pos=pos, speed=0, force=0, block=block)

    def move(self, pos, speed, force, block=True):
        self.hand.move(pos=pos, speed=speed, force=force, block=block)

    def stop(self):
        self.hand.stop()

    @property
    def position(self):
        return self.hand.position


class AdaptiveGripper:
    """
        How to Use:

        >>> hande = HandEForRtu(port="/dev/ttyUSB1")

        >>> hande.mov = lambda pos, block: hande.move(pos=pos, speed=0, force=0, block=block)

        >>> # the gripper obj must have the stop and mov and position func . and the stop func dont have the params, the mov func can pass in two parameters are pos and block。

        >>> adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=30, maxDiffNum=250)

        >>> def getOneFrame(cap):
        >>>     while 1:
        >>>         ret, frame = cap.read()
        >>>         if ret:
        >>>             warpedFrame = warp_perspective(frame, )
        >>>             return warpedFrame

        >>> adGripper.registerCapGetFrame(getOneFrame)  # the callback need to return the vaild realtime frame from the sensor

        >>> adgripper.open()

        >>> adgripper.close()

        >>> adgripper.adaptiveOpen()

        >>> adgripper.adaptiveClose()

    """
    SHOW_IMAGE = False

    def __init__(self, 
            gripper: ServoHandler, 
            minPos: float, 
            maxPos: float, 
            maxDiff: int, 
            maxDiffNum: int,
            speed: int = 0
            ) -> None:
        """
        _summary_

        Args:
            gripper (object): _description_
            minPos (float): _description_
            maxPos (float): _description_
            maxDiff (int): (单个MARK点)滤波阈值
            maxDiffNum (int): (所有MARK点之和)触发阈值
        """
        self._gripper = gripper

        self._capGetOneFrameCallbacks = []

        self._minPos, self._maxPos = minPos, maxPos
        self._maxDiff, self._maxDiffNum = maxDiff, maxDiffNum

        self._openEvent, self._closeEvent = Event(), Event()
        self._openThread, self._closeThread = None, None

        self._showrecoredFrame1, self._showRecordFrame2 = None, None
        self._showFrame1, self._showFrame2 = None, None
        self._diffFrame1, self._diffFrame2 = None, None

        self._state = False
        self._adaptiveState = False

        self._speed = speed

        self._capGetSlipStateCallbacks = []
        self._states = ['UNKNOWN', 'UNKNOWN']
        

    def open(self, block=True, maxPos: float=None):
        """
        open the gripper 

        Args:
            block (bool, optional): When set to True, the function will return after the gripper move done. Otherwise, it will return immediately, just call the move func. Defaults to True.
            minPos (float, optional): You can set the target open min pos, only this call takes effect. Defaults to None.
        """
        self._adaptiveState = False
        pos = self._maxPos if maxPos is None else maxPos
        if self._closeThread is not None and self._closeThread.is_alive():
            self._closeEvent.set()
            self._closeThread.join()
            self._closeEvent.clear()
            self._closeThread = None
        self._gripper.move(pos=pos, speed=255, force=0, block=block)
        self._state = False


    def close(self, speed=150, block=True, minPos: float=None):
        """
        close the gripper

        Args:
            block (bool, optional): When set to True, the function will return after the gripper move done. Otherwise, it will return immediately, just call the move func. Defaults to True.
            minPos (float, optional): You can set the target open min pos, only this call takes effect. Defaults to None.
        """
        self._adaptiveState = False
        pos = self._minPos if minPos is None else minPos
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()
            self._openThread = None
        # self._gripper.mov(pos=pos, block=block)
        self._gripper.move(pos, speed, 0, block=block)
        self._state = False

    def move(self, pos, speed, force, block=True):
        self._gripper.move(pos, speed, force, block=block)
    
    def position(self):
        return self._gripper.position
    

    def state(self):
        return self._state
    

    def adaptiveState(self):
        return self._adaptiveState
    

    def registerCapGetSlip(self, callback:Callable):
        """
        register the callback functions for get slip state from vision tactail camera.

            you can register multiple callback functions when there are multiple sensors

        Args:
            callback (Callable): _description_
        """
        self._capGetSlipStateCallbacks.append(callback)
        if len(self._capGetSlipStateCallbacks) > 2:
            warnings.warn("You have registered more than two getSlip func for cap, please confirm whether the quantity is correct.")


    def adaptiveFillWater(self):
        for gf225 in self._caps:
            if not gf225.is_inited_marker():
                gf225.calibrate(1)
    
        recordFrames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
        for gf225, frame in zip(self._caps, recordFrames):
            gf225.recon3d(frame)
        recordDepths:list[np.ndarray] = [gf225.get_depth_map() for gf225 in self._caps]

        thresholds = [20, 20]
        depthMeanDiffs = [0, 0]
        action = True
        while action:
            if round(self._gripper.position) >= 41:
                print("adaptiveFillWater detected limit")
                break

            frames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]

            absDiffs = list(map(cv2.absdiff, frames, recordFrames))
            absDiffNums = [np.sum(absDiff >= thresholds[i]) for i, absDiff in enumerate(absDiffs)]

            for gf225, frame in zip(self._caps, frames):
                gf225.recon3d(frame)
            depths:list[np.ndarray] = [gf225.get_depth_map() for gf225 in self._caps]
            depthDiffs = list(map(cv2.subtract, depths, recordDepths))
            for i, diff in enumerate(depthDiffs):
                alive = np.count_nonzero(diff) 
                diff_mean = 0 if alive == 0 else diff.sum()/alive
                if diff_mean > 0.19:
                    print(f"adaptiveFillWater depth limit, [{i}] {diff_mean}")
                    # action = False
                depthMeanDiffs[i] = diff_mean

        
            if not action or (all(depth > 0.14 for depth in depthMeanDiffs) and all(num < 500 for num in absDiffNums)):
                print(f"55")
                continue

            if absDiffNums[0] > 400:
                self._gripper.move(pos=self._gripper.position+1, speed=255, force=0, block=True)
                print(f"adaptiveFillWater detected, [0] {absDiffNums[0]}, {thresholds[0]}")
                thresholds[0] += 20 if thresholds[0] < 80 else 0
                recordFrames[0] = frames[0]
            elif absDiffNums[1] > 400:
                self._gripper.move(pos=self._gripper.position+1, speed=255, force=0, block=True)
                print(f"adaptiveFillWater detected, [1] {absDiffNums[1]}, {thresholds[1]}")
                thresholds[1] += 5 if thresholds[1] < 80 else 0
                recordFrames[1] = frames[1]
            
            # time.sleep(0.1)
        self._gripper.stop()


    def debounce(self, list1, list2):
        arr1 = np.array(list1)
        arr2 = np.array(list2)

        ox = arr1[:, 0]
        oy = arr1[:, 1]

        cx = arr2[:, 0]
        cy = arr2[:, 1]

        dx = cx - ox
        dy = cy - oy

        dist_squared = dx ** 2 + dy ** 2
        K = np.where(dist_squared < 3, 0, 1)

        return K * dx, K * dy


    def fillwater(self):
        for gf225 in self._caps:
            if not gf225.is_inited_marker():
                gf225.calibrate(1)
    
        recordFrames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
        for gf225, frame in zip(self._caps, recordFrames):
            gf225.recon3d(frame)
        recordDepths:list[np.ndarray] = [gf225.get_depth_map() for gf225 in self._caps]

        thresholds = [20]
        depthMeanDiffs = [0]
        action = True
        while action:
            if round(self._gripper.position) >= 41:
                print("adaptiveFillWater detected limit")
                break

            frames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]

            absDiffs = list(map(cv2.absdiff, frames, recordFrames))
            absDiffNums = [np.sum(absDiff >= thresholds[i]) for i, absDiff in enumerate(absDiffs)]

            for gf225, frame in zip(self._caps, frames):
                gf225.recon3d(frame)
            depths:list[np.ndarray] = [gf225.get_depth_map() for gf225 in self._caps]
            depthDiffs = list(map(cv2.subtract, depths, recordDepths))
            for i, diff in enumerate(depthDiffs):
                alive = np.count_nonzero(diff) 
                diff_mean = 0 if alive == 0 else diff.sum()/alive
                if diff_mean > 0.19:
                    print(f"adaptiveFillWater depth limit, [{i}] {diff_mean}")
                    # action = False
                depthMeanDiffs[i] = diff_mean

        
            if not action or (all(depth > 0.14 for depth in depthMeanDiffs) and all(num < 500 for num in absDiffNums)):
                print(f"55")
                continue

            if absDiffNums[0] > 400:
                self._gripper.move(pos=self._gripper.position+1, speed=255, force=0, block=True)
                print(f"adaptiveFillWater detected, [0] {absDiffNums[0]}, {thresholds[0]}")
                thresholds[0] += 5 if thresholds[0] < 80 else 0
                recordFrames[0] = frames[0]
            
            # time.sleep(0.1)
        self._gripper.stop()

