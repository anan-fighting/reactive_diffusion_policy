#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-05-16 17:19:52
LastEditTime : 2025-02-13 18:45:34
LastEditors  : error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Description  : 
'''
from enum import IntEnum
from .modbus import ModbusRTU
import time

DEFAULT_SYNC_TIME = 0.01

# --------------------------------------------------------
# Request 
class RACT(IntEnum):

    Deactivate = 0x0        # reset the gripper and clear any fault status
    Activate   = 0x1        # activate gripper, the first step befor all operation


class RGTO(IntEnum):
    Stop = 0x0              # 
    Go   = 0x1


class RATR(IntEnum):
    Normal       = 0x0
    EAutoRelease = 0x1


class RARD(IntEnum):
    CloseAutoRelease = 0x0
    OpenAutoRelease  = 0x1
# --------------------------------------------------------


# --------------------------------------------------------
# Response
class GACT(IntEnum): 
    GripperReset            = 0x00      # 
    GripperActivation       = 0x01


class GGTO(IntEnum): 
    Stopped                 = 0x00      
    Moving                  = 0x01


class GSTA(IntEnum): 
    InResetOrAutoRelease    = 0x00
    Activating              = 0x01
    NotUsed                 = 0x02
    ActivationCompleted     = 0x03


class GOBJ(IntEnum): 
    MovingAndNoObj          = 0x00
    ObjDetectedOpening      = 0x01
    ObjDetectedClosing      = 0x02
    MovDoneAndNoObj         = 0x03



# --------------------------------------------------------
class FaultStatus(IntEnum):
    NoFault = 0x00



def PosCheck(value):
    minValue = 0x00
    maxValue = 0xFF

    if value <= minValue: return minValue
    if value >= maxValue: return maxValue
    return value



SpeedCheck = PosCheck
ForceCheck = PosCheck




class HandERegister(IntEnum)   : 
    REQUEST_ACTION    = 0x03E8  # low byte, write
    REQUEST_POSITION  = 0x03E9  # high byte, write
    REQUEST_SPEED     = 0x03EA  # low byte, write
    REQUEST_FORCE     = 0x03EA  # high byte, write


    GRIPPER_STATUS    = 0x07D0  # low byte, read
    FAULT_STATUS      = 0x0701  # low byte, read
    ECHO_POSITION     = 0x0701  # high byte, read
    POSITION          = 0x07D2  # low byte,read
    CURRENT           = 0x07D2  # high byte,read


    @classmethod
    def getHighByteValue(cls, value):
        return (value >> 8) & 0xFF

    @classmethod
    def getLowByteValue(cls, value):
        return value & 0xFF


class HandEForRtu:
    DEFAULT_SLAVE_ID = 0x0009
    DEFAULT_BAUDRATE = 115200
    DEFAULT_DATA_BIT = 8
    DEFAULT_STOP_BIT = 1
    DEFAULT_PARITY   = "N"

    FULL_POS = 50
    POS_CONVERSION_RATIO = 0.1953125    # 50mm / 256


    def __init__(self, port, autoInit: bool=True) -> None:
        
        self.master = ModbusRTU(port=port,
                                baudrate=self.DEFAULT_BAUDRATE,
                                bytesize=self.DEFAULT_DATA_BIT,
                                parity=self.DEFAULT_PARITY,
                                stopbits=self.DEFAULT_STOP_BIT
                                )
        self.__rAct, self.__rGto, self.__rAtr, self.__rArd = None, None, None, None
        self.__gAct, self.__gGto, self.__gSta, self.__gObj = None, None, None, None
        self.__readGripperAction()
        self.readGripperStatus()

        self.__rAct = self.__gAct
        # reset and activate, must be execute befor first move
        if autoInit:
            self.initGripper()


    def initGripper(self):
        if self.__gAct != GACT.GripperActivation or self.__gSta != GSTA.ActivationCompleted:
            self.resetGripper()
            self.activateGripper()
            while True:
                self.readGripperStatus()
                if self.__gSta == GSTA.ActivationCompleted: 
                    break
                time.sleep(DEFAULT_SYNC_TIME)


    def activateGripper(self):
        self.__rAct = RACT.Activate
        self.master.write_single_register(HandERegister.REQUEST_ACTION, self.__requestAction(), slave=self.DEFAULT_SLAVE_ID)



    def move(self, pos, speed, force, block=True):

        self.__rGto   = RGTO.Go
        targetPos     = PosCheck(int(pos / self.POS_CONVERSION_RATIO))
        targetSpeed   = SpeedCheck(speed)
        targetForce   = ForceCheck(force)

        actionReg     = self.__requestAction()
        positionReg   = targetPos
        speedForceReg = targetSpeed << 8 | targetForce
        self.master.write_multi_registers(HandERegister.REQUEST_ACTION, [actionReg, positionReg, speedForceReg], slave=self.DEFAULT_SLAVE_ID)

        while block:
            self.readGripperStatus()
            if self.__gObj != GOBJ.MovingAndNoObj:
                break
            if self.__gGto == GGTO.Stopped:
                break
            time.sleep(DEFAULT_SYNC_TIME)


    def stop(self):
        self.__rGto = RGTO.Stop
        self.master.write_single_register(HandERegister.REQUEST_ACTION, self.__requestAction(), slave=self.DEFAULT_SLAVE_ID)


    def emergencyAutoRelease(self, releaseDirction:RARD=None):
        if releaseDirction is not None:
            self.__rArd = releaseDirction
        self.__rAtr = RATR.EAutoRelease
        self.master.write_single_register(HandERegister.REQUEST_ACTION, self.__requestAction(), slave=self.DEFAULT_SLAVE_ID)


    def resetGripper(self):
        self.__rAct = RACT.Deactivate
        self.master.write_single_register(HandERegister.REQUEST_ACTION, self.__requestAction(), slave=self.DEFAULT_SLAVE_ID)


    def __requestAction(self):
        value = 0x0000
        value = (self.__rAct << 0 + 8) | value
        value = (self.__rGto << 3 + 8) | value
        value = (self.__rAtr << 4 + 8) | value
        value = (self.__rArd << 5 + 8) | value
        return value


    def __readGripperAction(self):
        value = self.master.read_holding_registers(HandERegister.REQUEST_ACTION, slave=self.DEFAULT_SLAVE_ID)[0]
        action = HandERegister.getHighByteValue(value)
        self.__rAct = (action & 0b00000001)
        self.__rGto = (action & 0b00001000) >> 3
        self.__rAtr = (action & 0b00010000) >> 4
        self.__rArd = (action & 0b00100000) >> 5


    def readGripperStatus(self):
        value = self.master.read_holding_registers(HandERegister.GRIPPER_STATUS, slave=self.DEFAULT_SLAVE_ID)[0]
        state = HandERegister.getHighByteValue(value)
        self.__gAct = (state & 0b00000001)
        self.__gGto = (state & 0b00001000) >> 3
        self.__gSta = (state & 0b00110000) >> 4
        self.__gObj = (state & 0b11000000) >> 6
        return self.__gAct, self.__gGto, self.__gSta, self.__gObj


    @property
    def current(self):
        value = self.master.read_holding_registers(HandERegister.CURRENT, slave=self.DEFAULT_SLAVE_ID)[0]
        return HandERegister.getLowByteValue(value)
        

    @property
    def position(self):
        value = self.master.read_holding_registers(HandERegister.POSITION, slave=self.DEFAULT_SLAVE_ID)[0]
        return HandERegister.getHighByteValue(value) * self.POS_CONVERSION_RATIO


    @property
    def faultStatus(self):
        value = self.master.read_holding_registers(HandERegister.FAULT_STATUS, slave=self.DEFAULT_SLAVE_ID)[0]
        return HandERegister.getLowByteValue(value)


    @property
    def reqPosition(self):
        value = self.master.read_holding_registers(HandERegister.ECHO_POSITION, slave=self.DEFAULT_SLAVE_ID)[0]
        return HandERegister.getHighByteValue(value) * self.POS_CONVERSION_RATIO


