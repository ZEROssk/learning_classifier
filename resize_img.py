# coding:utf-8

from PIL import Image
import os

imgNames = os.listdir("/Users/zero/Desktop/local/learning_classifier/img/ao_seiber")

def readImg(imgName):
    try:
        print("start")
        img_src = Image.open("/Users/zero/Desktop/local/learning_classifier/img/ao_seiber"+ imgName)
        print("done")
    except:
        print("{} is not image file".format(imgName))
        img_src = 1
    return img_src

for imgName in imgNames:
    img_src = readImg(imgName)
    if img_src == 1:continue
    else:
        resizedImg = img_src.resize((50,50))
        resizedImg.save("50_50_"+imgName)
        print(imgName+" is done")

