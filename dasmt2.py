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
import scipy.signal as signal
import torch
import nnAudio.Spectrogram
import sklearn.preprocessing
# import matplotlib.pyplot as plt
import traceback
import warnings

warnings.filterwarnings("ignore")
import numpy as np

# np.set_printoptions(threshold=np.inf)

RECORD_SIZE = 1152
# 多阈值识别法
###############
EVENT_MAX_TIME_RANGE = 45  # seconds
EVENT_MAX_PT_RANGE = 25  # pts
EVENT_GROW_TIME_GAP = 20  # seconds，事件增长时间阈值
EVENT_GROW_PT_GAP = 15  # pts,事件增长距离阈值
EVENT_EXPIRED_TIME = 60  # seconds,事件不增长时多久被遗弃
EVENT_VISIBLE_HEAT = 340  # seconds,事件可见（上报）的时间阈值,即level=3
EVENT_SUSPECTED_HEAT = 600  # seconds,事件变为疑似的时间阈值,即level=2
EVENT_DANGROUS_HEAT = 1300  # seconds,事件变为严重威胁的时间阈值,即level=1
EVENT_DIG_HEAT_INC = 15
EVENT_DIG_HEAT_DEC = 100
EVENT_EXCAVATOR_HEAT_INC = 8
EVENT_EXCAVATOR_HEAT_DEC = 200
KM_PER_PT = 0.002
###############
HEART_BEAT_DURATION = 60  # 5s
# serverAddr = ("10.100.0.145", 9601)
# deviceAddr = ('127.0.0.1', 9989)
# LOG_FILE = './dasml.log'

serverAddr = ("192.168.136.203", 9601)
deviceAddr = ('127.0.0.1', 9989)
LOG_FILE = '/home/jetsky/software/dasml.log'
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
        self.timeRange = [datetime.now() + timedelta(days=9999),
                          datetime.now() - timedelta(days=9999)]
        self.ptRange = [999999, -999999]
        self.level = 0
        self.type = -1
        self.heat = 0
        self.alarmTimes = 0
        self.visible = False
        self.published = False
        self.updated = False

    def __init__(self, timeStamp, pt, type) -> None:
        self.id = getNewEventId()
        self.timeRange = [timeStamp, timeStamp]
        self.ptRange = [pt, pt]
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
        self.visible = False
        self.published = False
        self.updated = False
        self._updateLevel()

    def getCenterKm(self) -> float:
        return (self.ptRange[0] + (self.ptRange[1] - self.ptRange[0]) / 2.0) * KM_PER_PT

    def getDuration(self) -> int:
        return (self.timeRange[1] - self.timeRange[0]).seconds

    def howLongSinceLastUpdate(self, timeStamp) -> int:
        return (timeStamp - self.timeRange[1]).seconds

    def isVisible(self) -> bool:
        newVisible = False
        if type == 10:
            newVisible = True
        newVisible = (self.heat >= EVENT_VISIBLE_HEAT)
        if self.visible != newVisible:
            self.visible = newVisible
        return newVisible

    ###

    def isDistanceNearKm(self, km) -> bool:
        pt = km / KM_PER_PT
        return self.ptRange[0] <= pt <= self.ptRange[1]

    def isNearBy(self, timeStamp, pt, type) -> bool:
        if type != self.type:
            return False
        return (self.timeRange[1] <= timeStamp <= (self.timeRange[1] + timedelta(seconds=EVENT_MAX_TIME_RANGE))) and (
                (self.ptRange[0] - EVENT_MAX_PT_RANGE) <= pt <= (self.ptRange[1] + EVENT_MAX_PT_RANGE))

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
            self._updateLevel()
            self.updated = True
            return True
        return False

    def _updateLevel(self) -> int:
        if EVENT_VISIBLE_HEAT <= self.heat < EVENT_SUSPECTED_HEAT:
            self.level = 1
        elif EVENT_SUSPECTED_HEAT <= self.heat < EVENT_DANGROUS_HEAT:
            self.level = 2
        elif self.heat >= EVENT_DANGROUS_HEAT:
            self.level = 3
        return self.level

    def annealing(self):
        if not self.updated:
            if self.type == 7:
                # dig
                self.heat -= EVENT_DIG_HEAT_DEC
            elif self.type == 8:
                # excavator
                self.heat -= EVENT_EXCAVATOR_HEAT_DEC
            elif self.type == 10:
                # fiberbreak
                self.heat -= 0
            if self.heat < 0:
                self.heat = 0

    def genAlarmPackage(self, startOrStop) -> bytes:
        if startOrStop:
            self.published = True
            self.alarmTimes += 1
        self.updated = False
        alarmStr = f"DVS,{'S' if startOrStop else 'E'},{self.id},1,1,{self.getCenterKm():.3f},{self.level},{self.type},{self.alarmTimes},{self.heat}#"
        return alarmStr.encode(encoding='UTF-8')

    def hasBeenPublished(self) -> bool:
        return self.published


async def mlEngineProc(name, ipaddr):
    ip, port = ipaddr
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fs = 500  # Hz
    NFFT = 512
    noverlap = 256

    # highpass filter
    cutoff = 50  # Hz
    order = 5
    # calculate the Nyquist frequency
    nyq = 0.5 * fs
    # design filter
    high = cutoff / nyq
    b, a = signal.butter(order, high, btype='highpass', analog=False)

    mfcc_layer = nnAudio.Spectrogram.MFCC(
        sr=fs, n_fft=NFFT, win_length=NFFT, hop_length=noverlap, window='hann').to(device)

    reader, writer = await asyncio.open_connection(ip, port)
    logger.info(f'[{name}] Connecting to server {ipaddr} succeed!')

    workDir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"[{name}] Trying to receive data from {ipaddr}")

    features = np.zeros((64, RECORD_SIZE), dtype=np.float32)
    recvBuf = np.zeros(RECORD_SIZE * 256 * 64, dtype=np.float32)
    idx = 0
    try:
        while True:
            errBreak = False
            recvBuf[:] = 0
            for rt in range(64 // 4):
                headerData = await reader.readexactly(3)
                magic = str(headerData, encoding="utf-8")
                logger.debug(f"magic = {magic}")
                if magic == "SHT":
                    recvBuf[RECORD_SIZE * 256 * 4 * rt:RECORD_SIZE * 256 * 4 * (rt + 1)] = np.frombuffer(
                        await reader.readexactly(2 * RECORD_SIZE * 256 * 4), dtype=np.ushort).astype(dtype=np.float32)
                    logger.info(
                        f"[{name}] Recieving data from {ipaddr} ... magic = {magic}")
                else:
                    errBreak = True
                    break
            if errBreak:
                break

            # 拼帧
            frameShts = recvBuf.reshape((-1, RECORD_SIZE))
            print(f'******** frameShts shape = {frameShts.shape}')
            # plt.plot(frameShts[0,:])
            # plt.savefig("test.png")
            # plt.close()
            # plt.plot(frameShts[:,0])
            # plt.savefig("test1.png")
            # plt.close()
            # frame = frameShts # [trigger,pt]
            # print(f'******** frame shape = {frame.shape}')
            # plt.plot(frame[0,:])
            # plt.savefig("test.png")
            # plt.close()
            # plt.plot(frame[:,0])
            # plt.savefig("test1.png")
            # plt.close()
            frame = frameShts.T  # [pt,trigger]
            print(f'******** frame shape = {frame.shape}')
            # lastFramSht=frameShts[-256:,:].copy()
            # print(f'******** lastFramSht shape = {frameShts.shape}')
            for pt in range(RECORD_SIZE):
                y = frame[pt, :]
                y_filterd = signal.filtfilt(b, a, y).copy()
                y_filterd = torch.from_numpy(y_filterd).float().to(device)
                mfcc = mfcc_layer(y_filterd).cpu().numpy()[0]
                mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
                lowest = mfcc[0, :]
                lowest = np.abs(np.diff(lowest))
                lowest = np.where((lowest < 1.7), 0, lowest)
                lowest = np.where((lowest >= 1.7), 1, lowest)
                if pt == 0:
                    print(f"lowest shape = {lowest.shape}")
                features[:, pt] = lowest

            # #debug
            # pt=214            
            # plt.plot(features[:,pt])
            # plt.savefig(f"test.{pt}.{idx}.png")
            # plt.close()
            # idx+=1
            # pass

            # 判断报警
            # ['背景噪声', '重车通过', '人工挖掘', '机械挖掘']
            outputs = np.array([[1, 0, 0, 0]] * RECORD_SIZE)
            for pt in range(RECORD_SIZE):  # 在此判断类型
                feature: np.ndarray = features[:, pt]
                ones = np.sum(feature == 1)
                # if pt==209 or pt==214:
                #     print(f"pt = {pt}, ones = {ones}")
                if ones > 11:
                    print(f"Slapping: pt = {pt}")
                    outputs[pt, :] = [0, 0, 1, 0]
            notifierQueue.put(outputs)
    except:
        traceback.print_exc()
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
        result = []
        timeStamp = datetime.now()
        if outputs is not None:
            labelMap = ['背景噪声', '重车通过', '人工挖掘', '机械挖掘']
            if isinstance(outputs, int):
                logger.debug("**************")
                result.append((outputs, 10, 1))  # 桂林周总要求 10:断纤
            else:
                pt = 0
                for pt in range(len(outputs)):
                    pt_recog = outputs[pt]
                    type = pt_recog.argmax()
                    confidence = pt_recog[type]  # /sum(pt) #sum(pt)==1
                    # print(f"sum(pt) = {sum(pt)}")
                    if type > 1 and pt_recog[type] > 0.25:  # 桂林周总：只需要人工、机械报警
                        # 桂林周总要求 7:人工，8：机械
                        result.append((pt, type + 5, confidence))

        eventToSend = []
        i = 0
        while i < len(events):
            event: Event = events[i]
            if event.howLongSinceLastUpdate(timeStamp) > EVENT_EXPIRED_TIME and (not event.isVisible()):  # 事件超时，则清理掉
                print(
                    f"**** Expired Event[{event.id}]: ptRange = {event.ptRange}, howLongSinceLastUpdate = {event.howLongSinceLastUpdate(timeStamp)}, heat = {event.heat}")
                # expired, remove event
                if event.hasBeenPublished():  # 已经作为有效报警发送到服务器的，需要发送事件结束消息
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
                            print(
                                f"**** Update Event[{event.id}]: ptRange = {event.ptRange}, type = {event.getType()}, heat = {event.heat}")
                            eventToSend.append(event)
                        result.pop(j)
                    else:
                        # event无更新，不用发送
                        j += 1

        for pt, type, confidence in result:
            j = 0
            update = False
            while j < len(events):
                event = events[j]
                if event.tryUpdate(timeStamp, pt, type):
                    # event有更新,只发送可见的事件（即level>=1的事件）
                    if event.isVisible():
                        print(
                            f"**** Update Event[{event.id}]: ptRange = {event.ptRange}, type = {event.getType()}, heat = {event.heat}")
                        eventToSend.append(event)
                        update = True
                        break
                else:
                    # event无更新，不用发送
                    j += 1
            if not update:
                # 创建新event，level<1，不用发送
                event = Event(timeStamp, pt, type)
                events.append(event)
                print(
                    f"**** New Event[{event.id}]: ptRange = {event.ptRange}, type = {event.getType()}, heat = {event.heat}")
                if type == 10:
                    logger.debug(f"firberBreak: {pt}")

        for event in events:
            # 退火
            event: Event = event
            event.annealing()
            print(f"**** Anneal Event[{event.id}]: ptRange = {event.ptRange}, heat = {event.heat}")

        for event in eventToSend:
            pkg = event.genAlarmPackage(True)
            with open('alarm.log', 'a', encoding='utf-8') as alarmFile:
                alarmFile.write(str(pkg))
                alarmFile.write('\n')
            print(
                f"**** Sending Event[{event.id}]: ptRange = {event.ptRange}, km = {event.getCenterKm()}, heat = {event.heat}")
            writer.write(pkg)  # send event-start
            await writer.drain()

        outputs = None


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
    # coros = [tcp_reconnect("PowerEngine", deviceAddr, powerEngineProc),
    #          tcp_reconnect("Notifier", serverAddr, notifierProc)]
    coros = [tcp_reconnect("Notifier", serverAddr, notifierProc),
             tcp_reconnect("MlEngine", deviceAddr, mlEngineProc)]

    await asyncio.gather(*coros)


if __name__ == "__main__":
    logger.info("DAS Machine Learning Node is running")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
    logger.info("DAS Machine Learning Node is stopped")
