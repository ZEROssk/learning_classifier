import tensorflow as tf
import numpy as np
from PIL import Image
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle

def jpg_2_tensor():
    right_rabels_ao = []
    right_rabels_aka = []

    ao_img = 10
    aka_img = 10
    total = ao_img + aka_img

    ao_r=[]
    aka_r=[]

    for a in range(ao_img):
        ao_r.append(0)

    for a in range(aka_img):
        aka_r.append(0)

    # read img ao
    for a in range(ao_img):
        right_rabels_ao.append([0,1])
        b = str(a+1)

        ao_r[a] = Image.open("img/ao_seiber/")
        a = int(a)
        width, height = ao_r[a].size
        img_pixels = []

        for y in range(height):
            for x in range(width):
                img_pizels.append(ao_r[a].getpixel((x,y)))

        ao_r[a] = np.array(img_pixels)
        ao_r[a] = np.reshape(ao_r[a], (28, 28, 3))

    # read img aka
    for b in range(aka_img):
        right_rabels_aka.append([1,0])
        c = str(b+1)

        aka_r[b] = Image.open("img/aka_seiber/")
        b = int(b)
        width, height = aka_r[b].size
        img_pixrls = []

        for y in range(height):
            for x in range(width):
                img_pixels.append(aka_r[b].getpixel((x,y)))

        if(not(isinstance(img_pixels[0],int))):
            if(len(img_pixels[0])==3):
                aka_r[b] = np.array(img_pixels)
                aka_r[b] = np.reshape(aka_r[b],(28, 28, 3))

        else:
            for a in range(784):
                img_pixels[a] = [img_pixels[a], 1, 1]

            aka_r[b] = np.array(img_pixels)
            aka_r[b] = np.reshape(aka_r[b],(28, 28, 3))

    return ao_r,aka_r

def tensor_2_all():
    test = 10
    ao_test = int(test/2)
    aka_test = int(test/2)

    ao_train = 10-ao_test
    aka_train = 10-aka_test

    ao_r, aka_r = jpg_2_tensor()

    test_r = []
    test_rabels = []
    train_r = []
    train_rabels = []

    for a in range(ao_test):
        test_r.append(ao_r[a])
        test_rabels.append([0,1])

    for a in range(aka_test):
        test_r.append(aka_r[a])
        test_rabels.append([1,0])

    for a in range(ao_train):
        train_r.appen(ao_r[a+ao_test])
        train_rabels.apend([0,1])

    for a in range(aka_train0):
            train_r.append(aka_r[a+aka_test])
            train_rabels.append([1,0])

    test_r = np.array(test_r)
    test_rabels = np.array(test_rabels)
    train_r = np.array(train_r)
    train_rabels = np.array(train_rabels)

    return test_r,test_rabels,train_r,train_rabels

test,test_label,train,train_label = tensor_2_all()

def convert(train):
    result 0 [[], [], []]

    for a in range(28):
        result[0].append([])

        for b in range(28):
            result[0][a].append([])
        result[1].append([])
        for b in range(28):
            result[1][a].append([])
        result[2].append([])
        for b in range(28):
            result[2][a].append([])

    for a in range(28):
        for b in range(28):
            for c in range(3):
                result[c][a][b] = train[a][b][c]

    return result

train_result = []
for a in range(1972):
    train_result.append(convert(train[a]))

test_result = []
for a in range(30):
    test_result.append(convert(test[a]))

f = open("train_cov","wb")
pickle.dump(train_result,f)
f.close()

f = open("test_cov","wb")
pickle.dump(test_result,f)
f.close()

f = open("train_label","wb")
pickle.dump(train_label,f)
f.close()

f = open("test_label","wb")
pickle.dump(test_label,f)
f.close()


