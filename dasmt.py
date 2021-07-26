import asyncio
import logging
import os
import socket
import time
from asyncio.streams import StreamReader, StreamWriter
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from queue import Empty, Queue
from typing import Tuple

import numpy as np

np.set_printoptions(threshold=np.inf)

# 多阈值识别法
###############
EVENT_MAX_TIME_RANGE = 20  # seconds
EVENT_MAX_PT_RANGE = 35  # pts
EVENT_GROW_TIME_GAP = 20  # seconds，事件增长时间阈值
EVENT_GROW_PT_GAP = 15  # pts,事件增长距离阈值
EVENT_EXPIRED_TIME = 50  # seconds,事件不增长时多久被遗弃
EVENT_VISIBLE_TIME = 20  # seconds,事件可见（上报）的时间阈值,即level=3
EVENT_SUSPECTED_TIME = 40  # seconds,事件变为疑似的时间阈值,即level=2
EVENT_DANGROUS_TIME = 80  # seconds,事件变为严重威胁的时间阈值,即level=1
EVENT_DIG_HEAT_INC = 15
EVENT_DIG_HEAT_DEC = 5
EVENT_EXCAVATOR_HEAT_INC = 8
EVENT_EXCAVATOR_HEAT_DEC = 2
KM_PER_PT = 0.02
###############
HEART_BEAT_DURATION = 5  # 5s
# serverAddr = ("127.0.0.1", 8848)
# deviceAddr = ('10.100.0.244', 2112)
# LOG_FILE = './dasml.log'

serverAddr = ("192.168.136.100", 9601)
deviceAddr = ('127.0.0.1', 2111)
LOG_FILE = '/home/jetsky/source/dasml.log'
###############
# 创建一个handler，用于写入日志文件
fh = RotatingFileHandler(LOG_FILE,
                         mode='a',
                         maxBytes=1 * 1024 * 1024,  # 1MB
                         backupCount=10)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# 定义handler的输出格式formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
# 创建一个logger
logger = logging.getLogger("dasmt")
logger.setLevel(level=logging.DEBUG)
# 给logger添加handler
# logger.addFilter(filter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info("test")

notifierQueue = Queue(maxsize=0)

EventIdPool: int = 0


def getNewEventId() -> int:
    global EventIdPool
    EventIdPool = (EventIdPool + 1) % 4294967296
    return EventIdPool


class Event(object):
    def __init__(self) -> None:
        self.id = getNewEventId()
        self.timeRange = (datetime.now() + timedelta(days=9999),
                          datetime.now() - timedelta(days=9999))
        self.ptRange = range(999999, -999999)
        self.level = 0
        self.type = -1
        self.heat = 0
        self.alarmTimes = 0
        self.updated = False

    def __init__(self, timeStamp, pt, type) -> None:
        self.id = getNewEventId()
        self.timeRange = (timeStamp, timeStamp)
        self.ptRange = (pt, pt)
        self.level = 0
        self.type = type
        self.heat = 0
        if type == 7:
            # dig
            self.heat += EVENT_DIG_HEAT_INC
        elif type == 8:
            # excavator
            self.heat += EVENT_EXCAVATOR_HEAT_INC
        elif type == 10:
            # fiberbreak
            self.heat += 10000
        self.alarmTimes = 1
        self.updated = True
        self.updateLevel()

    def getCenterKm(self) -> float:
        return (self.ptRange[0] + (self.ptRange[1] - self.ptRange[0]) / 2.0) * KM_PER_PT

    def getDuration(self) -> int:
        return (self.timeRange[1] - self.timeRange[0]).seconds

    def howLongSinceLastUpdate(self, timeStamp) -> int:
        return (timeStamp - self.timeRange[1]).seconds

    def isVisible(self) -> bool:
        if type == 10:
            return True
        return self.getDuration() >= EVENT_VISIBLE_TIME

    ###

    def isDistanceNearKm(self, km) -> bool:
        pt = km / KM_PER_PT
        return self.ptRange[0] <= pt <= self.ptRange[1]

    def isNearBy(self, timeStamp, pt, type) -> bool:
        if type != self.type:
            return False
        return ((self.timeRange[0] - timedelta(seconds=EVENT_MAX_TIME_RANGE)) <= timeStamp <= self.timeRange[1]) and (
                    self.ptRange[0] <= pt <= self.ptRange[1])

    def getType(self) -> int:
        return self.type

    def tryUpdate(self, timeStamp, pt, type) -> bool:
        if self.isNearBy(timeStamp, pt, type) or self.type == -1:
            self.timeRange[1] = timeStamp if timeStamp > self.timeRange[1] else self.timeRange[1]
            self.ptRange[1] = pt if pt > self.ptRange[1] else self.ptRange[1]
            if type == 7:
                # dig
                self.heat += EVENT_DIG_HEAT_INC
            elif type == 8:
                # excavator
                self.heat += EVENT_EXCAVATOR_HEAT_INC
            elif type == 10:
                # fiberbreak
                self.heat += 10000
            self.type = type
            self.updated = True
            self.updateLevel()
            return True
        return False

    def updateLevel(self) -> int:
        duration = self.getDuration()
        if EVENT_VISIBLE_TIME <= duration < EVENT_SUSPECTED_TIME:
            self.level = 1
        elif EVENT_SUSPECTED_TIME <= duration < EVENT_DANGROUS_TIME:
            self.level = 2
        elif duration >= EVENT_DANGROUS_TIME:
            self.level = 3
        return self.level

    def annealing(self):
        if not self.updated:
            if type == 7:
                # dig
                self.heat -= EVENT_DIG_HEAT_DEC
            elif type == 8:
                # excavator
                self.heat -= EVENT_EXCAVATOR_HEAT_DEC
            elif type == 10:
                # fiberbreak
                self.heat -= 0

    def genAlarmPackage(self, startOrStop) -> bytes:
        if startOrStop:
            self.alarmTimes += 1
        self.updated = False
        return f"DVS,{'S' if startOrStop else 'E'},{self.id},1,1,{self.getCenterKm():.3f},{self.level},{self.type},{self.alarmTimes},{self.heat}#".encode(
            encoding='UTF-8')


def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    x_min = np.min(x, axis=1, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    softmax = x_exp / x_sum
    return softmax


async def powerEngineProc(name, ipaddr):
    ip, port = ipaddr
    reader, writer = await asyncio.open_connection(ip, port)
    logger.info(f'[{name}] Connecting to server {ipaddr} succeed!')

    workDir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"[{name}] Trying to receive data from {ipaddr}")

    triNum = 24
    frmNum = 40

    try:
        while True:
            headerData = await reader.readexactly(4)
            if int.from_bytes(headerData, "little") == 0:
                pt_break = int.from_bytes(await reader.readexactly(4), "little")
                notifierQueue.put(pt_break)
                continue
            magic = str(headerData, encoding="utf-8")
            logger.debug(f"magic = {magic}")
            if magic == "MELD":
                headerData = await reader.readexactly(4)
                recordSize = int.from_bytes(headerData, "little")
                melData = await reader.readexactly(recordSize * triNum * frmNum * 4)
                logger.info(
                    f"[{name}] Recieving data from {ipaddr} ... magic = {magic}, recordSize = {recordSize}, melData.size = {len(melData)}")
                continue
            elif magic == "PWER":
                headerData = await reader.readexactly(4)
                recordSize = int.from_bytes(headerData, "big")
                powerData = np.frombuffer(await reader.readexactly(recordSize), dtype=np.uint8)
                scData = np.frombuffer(await reader.readexactly(recordSize * 4), dtype=np.float32)
                logger.info(
                    f"[{name}] Recieving data from {ipaddr} ... magic = {magic}, recordSize = {recordSize}, powerData.size = {recordSize}")

            # if 0 <= pt <= recordSize - 1:  # 在此划定点范围
            #     pass
            # else:
            #     pass


            outputs = np.array([[1, 0, 0, 0]] * recordSize)  # ['背景噪声', '重车通过', '人工挖掘', '机械挖掘']
            for pt in range(recordSize):  # 在此判断类型
                if 0 <= pt <= 25:
                    if powerData[pt] >= 0 and scData[pt] >= 78:
                        outputs[pt] = [0, 0, 1, 0]
                elif 26 <= pt <= recordSize - 1:
                    if powerData[pt] >= 0 and scData[pt] >= 78:
                        outputs[pt] = [0, 0, 1, 0]
                else:
                    if powerData[pt] >= 0 and scData[pt] >= 78:
                        outputs[pt] = [1, 0, 0, 0]
            # 调试用，存储特征量
            with open('powerData-1626421043-dg.txt', 'ab') as outputFile:
                np.save(outputFile, powerData)
            with open('scData.txt', 'ab') as outputFile:
                np.save(outputFile, scData)
            # input = np.frombuffer(melData, dtype='float32')
            # inputs = np.reshape(input, (-1, 1, frmNum, triNum))
            # logger.debug(inputs.shape)
            # logger.debug(input)
            # # inputs = np.divide(inputs, np.reshape(np.max(inputs, axis=-1),  (recordSize,1,frmNum,1)))
            # outputs = dasNet(torch.from_numpy(inputs).to(device))
            # outputs = outputs.cpu().detach().numpy()
            # outputs = softmax(outputs)

            notifierQueue.put(outputs)
    except:
        pass
    finally:
        writer.close()


async def notifierRecv(name, reader: StreamReader, writer: StreamWriter):
    try:
        buf: bytes = await asyncio.wait_for(reader.readuntil(b'#'), timeout=0.5)
    except asyncio.TimeoutError:
        return
    # pkg: bytes = await reader.readuntil(b'#')
    rxLen = len(buf)
    logger.debug(f"{name} rxLen = {rxLen}")
    logger.info(f"{name} Received command from: {str(buf)}")
    pkg = str(buf[3:-1], encoding="utf-8")
    logger.debug(f"{name} pkg.split: {pkg}")

    recv_sp = pkg.split(',')
    if recv_sp[1] == b"DATA":
        # 获取波形
        pass
    elif recv_sp[1] == "THR":
        # 设置阈值
        ch = recv_sp[2]
        thres = recv_sp[3].split('|')
        # TODO save thres and apply
        thresSuccess = True
        if thresSuccess:
            writer.write(b"DVS,THR,OK#")
        else:
            writer.write(b"DVS,THR,ERR#")
        await writer.drain()
    else:
        logger.error(f"{name} Unknow command: {recv_sp[1]}")


async def notifierProc(name, ipaddr):
    ip, port = ipaddr
    reader, writer = await asyncio.open_connection(ip, port)
    logger.info(f'[{name}] Connecting to server {ipaddr} succeed!')

    events = []
    tick = time.time()

    # buffer: bytearray = bytearray()
    outputs: list = None
    while True:
        # heartbeat 5s
        newTick = time.time()
        if newTick - tick >= 5:
            tick = newTick
            writer.write(b"DVS,H,1,State#")
            await writer.drain()
            logger.info(f"{name} sending heart beat!")

        # 应答通信
        await notifierRecv(name, reader, writer)

        try:
            outputs = notifierQueue.get_nowait()
        except Empty:
            await asyncio.sleep(1)
        if outputs is None:
            continue

        timeStamp = datetime.now()
        labelMap = ['背景噪声', '重车通过', '人工挖掘', '机械挖掘']
        result = []
        if isinstance(outputs, int):
            logger.debug("**************")
            result.append((outputs, 10, 1))  # 桂林周总要求 10:断纤
        else:
            for pt in outputs:
                type = pt.argmax()
                confidence = pt[type]  # /sum(pt) #sum(pt)==1
                # print(f"sum(pt) = {sum(pt)}")
                if type > 1 and pt[type] > 0.25:  # 桂林周总：只需要人工、机械报警
                    result.append((pt, type + 5, confidence))  # 桂林周总要求 7:人工，8：机械

        eventToSend = []
        i = 0
        while i < len(events):
            event: Event = events[i]
            if event.howLongSinceLastUpdate(timeStamp) > EVENT_EXPIRED_TIME:
                # expired, remove event
                pkg = event.genAlarmPackage(False)
                writer.write(pkg)  # send event-end
                await writer.drain()
                events.pop(i)
            else:
                i += 1
                j = 0
                while j < len(result):
                    pt, type, confidence = result[j]
                    if event.tryUpdate(timeStamp, pt, type):
                        # event有更新,只发送可见的事件（即level>=1的事件）
                        if event.isVisible():
                            eventToSend.append(event)
                        result.pop(j)
                    else:
                        # event无更新，不用发送
                        j += 1

        for pt, type, confidence in result:
            # 创建新event，level<1，不用发送
            event = Event(timeStamp, pt, type)
            events.append(event)
            if type == 10:
                logger.debug(f"firberBreak: {pt}")

        for event in events:
            # 退火
            event: Event = event
            event.annealing()

        for event in eventToSend:
            pkg = event.genAlarmPackage(True)
            writer.write(pkg)  # send event-start
            await writer.drain()


async def tcp_reconnect(name, ipaddr, proc):
    while True:
        logger.info(f'[{name}] Trying to (re)connect to server {ipaddr} ...')
        try:
            await proc(name, ipaddr)
        except ConnectionRefusedError:
            logger.info(f'[{name}] Connecting to server {ipaddr} failed!')
        except asyncio.TimeoutError:
            logger.info(f'[{name}] Connecting to server {ipaddr} timed out!')
        else:
            logger.info(f'[{name}] Connection to server {ipaddr} is closed.')
        await asyncio.sleep(2.0)


async def main():
    coros = [tcp_reconnect("PowerEngine", deviceAddr, powerEngineProc),
             tcp_reconnect("Notifier", serverAddr, notifierProc)]
    await asyncio.gather(*coros)


if __name__ == "__main__":
    logger.info("DAS Machine Learning Node is running")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
    logger.info("DAS Machine Learning Node is stopped")
