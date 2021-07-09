import sys
from struct import pack, unpack
import multiprocessing
import socket
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from das_net import DasNet
import argparse
import numpy as np
cimport numpy as np
import socket
import multiprocessing
from struct import pack, unpack


cdef np.ndarray softmax(np.ndarray x):
    assert x.dtype == np.float32

    cdef float x_max = np.max(x, axis=1, keepdims=True)
    cdef float x_min = np.min(x, axis=1, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    cdef np.ndarray x_exp = np.exp(x)
    cdef float x_sum = np.sum(x_exp, axis=1, keepdims=True)
    cdef np.ndarray softmax = x_exp/x_sum
    return softmax


def socket_process(name, port, queue):
    exitFlag = False
    sck_listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sck_listen.bind(("0.0.0.0", port))
    sck_listen.listen(5)
    while not exitFlag:
        sck, addr = sck_listen.accept()
        print("%s client connected %s\n" % (name, addr))
        while True:
            dataToSend = queue.get()
            n = sck.send(pack("i", len(dataToSend)))
            if n <= 0:
                break
            n = sck.send(dataToSend)
            if n <= 0:
                break
            # Listen and loop send alert
            if dataToSend == None:
                exitFlag = True
                break
        print("%s client disconnected\n" % name)

# GPU or CPU
if __name__ == "__main__":
    # queue for alert report to frontend
    featureQueue = multiprocessing.Queue(4)
    outputQueue = multiprocessing.Queue(4)

    argsFeature = ('Feature', 2333, featureQueue)
    procFeature = multiprocessing.Process(target=socket_process, args=argsFeature)
    procFeature.start()

    argsOutput = ('Output', 2334, outputQueue)
    procOutput = multiprocessing.Process(target=socket_process, args=argsOutput)
    procOutput.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weightFile = './pretrained_dropout.pkl'
    dasNet = DasNet().to(device)
    dasNet.load_state_dict(torch.load(weightFile))
    dasNet.eval()
   
    sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sck.connect(('192.168.1.100', 2112))

    triNum = 24
    frmNum = 40
    # recordSize = 8189
    
    while True:
        headerData = bytes()
        sizeLeft = 4
        while sizeLeft > 0:
            tmp = sck.recv(sizeLeft, 0)
            sizeLeft = sizeLeft - len(tmp)
            headerData = headerData + tmp
        recordSize = int.from_bytes(headerData, 'big')
        melData = bytes()
        sizeLeft = 4 * recordSize * triNum * frmNum
        while sizeLeft > 0:
            tmp = sck.recv(sizeLeft, 0)
            sizeLeft = sizeLeft - len(tmp)
            melData = melData + tmp
        input = np.frombuffer(melData, dtype='float32')
        inputs = np.reshape(input,(-1, 1, frmNum, triNum))
        # inputs = np.divide(inputs, np.reshape(np.max(inputs, axis=-1),  (recordSize,1,frmNum,1)))
        outputs = dasNet(torch.from_numpy(inputs).to(device))
        outputs = outputs.cpu().detach().numpy()
        outputs = softmax(outputs)
        
        if(featureQueue.full()):
            featureQueue.get()
        if(outputQueue.full()):
            outputQueue.get()
        featureQueue.put(melData)
        c = 0
        i = 0
        labelMap = ['背景噪声', '重车通过', '人工挖掘', '机械挖掘']
        for pt in outputs:
            # if(np.max(pt) == pt[0]):
            #     c = c + 1
            # else:
            #     print(pt)
            i = i + 1
            if(pt.argmax() != 0 and pt[pt.argmax()] > 0.25):
                c = c + 1
                print('%.03fKM-%s'%((i * 0.02),labelMap[pt.argmax()]), end=('\n' if c % 5 == 0 else '\t'))

        # print(c)
        print('==============')
        outputQueue.put(outputs.tobytes())