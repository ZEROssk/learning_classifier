#coding: utf-8

import cv2
import os
import six
import datetime

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S

import numpy as np

# ------train model------

class bake_model(chainer.Chain):
    def __init__(self):

        super(bake_model, self).__init__(
            conv1 =  F.Convolution2D(3, 16, 5, pad=2),
            conv2 =  F.Convolution2D(16, 32, 5, pad=2),
            l3    =  F.Linear(6272, 256),
            l4    =  F.Linear(256, 2)
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, X_data, y_data, train=True):
        self.clear() #初期化
        X_data = chainer.Variable(np.asarray(X_data), volatile=not train)
        y_data = chainer.Variable(np.asarray(y_data), volatile=not train)
        h = F.max_pooling_2d(F.relu(self.conv1(X_data)), ksize = 5, stride = 2, pad =2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize = 5, stride = 2, pad =2)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)

        return F.softmax_cross_entropy(y, y_data), F.accuracy(y, y_data)

# ------DataSet------

def getDataSet():
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,2):
        path = "/Users/zero/Desktop/local/study_machine_learning/img/resize_ao/"
        imgList = os.listdir(path+str(i))

        imgNum = len(imgList)
        cutNum = imgNum - imgNum/5

        for j in range(len(imgList)):
            imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])

            if imgSrc is None:continue

            if j < cutNum:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)
        return X_train,y_train,Xtest,y_test

# ------train------

def train():
    X_train,y_train,X_test,y_test = getDataSet()
    X_train = np.array(X_train).astype(np.float32).reshape((len(X_train), 3, 256, 256)) / 255
    y_train = np.array(y_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.float32).reshape((len(X_test), 3, 256, 256)) / 255
    y_test = np.array(y_test).astype(np.int32)

    model = bake_model()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    epochNum = 5
    batchNum = 50
    epoch = 1

    # ---train & test---
    while epoch <= epochNum:
        print("epoch: {}".format(epoch))
        print(datetime.datetime.now())

        trainImgNum = len(y_train)
        testImgNum = len(y_test)

        # ---train---
        sumAcr = 0
        sumLoss = 0

        perm = np.random.permutation(trainImgNum)

        for i in six.moves.range(0, trainImgNum, batchNum):
            X_batch = X_train[perm[i:i+batchNum]]
            y_batch = y_train[perm[i:i+batchNum]]

            optimizer.zero_grads()
            loss, acc = model.forward(X_batch, y_batch)
            loss.backward()
            optimizer.update()

            sumLoss += float(loss.data) * len(y_batch)
            sumAcr += float(acc.data) * len(y_batch)
        print('train mean loss={}, accuracy={}'.format(sumLoss / trainImgNum, sumAcr / trainImgNum))

        # ---test---
        sumAcr = 0
        sumLoss = 0

        perm = np.random.permutation(testImgNum)

        for i in six.moves.range(0, testImgNum, batchNum):
            X_batch = X_test[perm[i:i+batchNum]]
            y_batch = y_test[perm[i:i+batchNum]]
            loss, acc = model.forward(X_batch, y_batch, train=False)

            sumLoss += float(loss.data) * len(y_batch)
            sumAcr += float(acc.data) * len(y_batch)
        print('test mean loss={}, accuracy={}'.format(sumLoss / testImgNum, sumAcr / testImgNum))
        epoch += 1

        # ---save model---
        S.save_hdf5('model'+str(epoch+1), model)

train()

