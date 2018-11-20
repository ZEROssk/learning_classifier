# coding:utf-8

from PIL import Image
import os

imgNames = os.listdir("/Users/zero/Desktop/local/study_machine_learning/img/ao_seiber")

def readImg(imgName):
    try:
        img_src = Image.open("/Users/zero/Desktop/local/study_machine_learning/img/ao_seiber/"+ imgName)
    except:
        print("{} is not image file".format(imgName))
        img_src = 1
    return img_src

for imgName in imgNames:
    img_src = readImg(imgName)
    if img_src == 1:continue
    else:
        resizedImg = img_src.resize((256,256))
        resizedImg.save("img/resize_ao/resize_"+imgName, quality=100)
        print(imgName+" is resize & save")

