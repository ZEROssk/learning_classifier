# coding:utf-8

from PIL import Image
import os

def readImg(imgName):
    try:
        img_src = Image.open("/Users/zero/Desktop/local/study_machine_learning/img/aka_seiber/" + imgName)
        print("done")
    except:
        print("{} is not image file".format(imgName))
        img_src = 1
    return img_src

def spinImg(imgNames):
    for imgName in imgNames:
        img_src = readImg(imgName)
        if img_src == 1:continue
        else:
            tmp = img_src.transpose(Image.FLIP_TOP_BOTTOM)
            tmp.save("data/1/flipTB_" + imgName)

            tmp = img_src.transpose(Image.ROTATE_90)
            tmp.save("data/1/spin90_" + imgName)

            tmp = img_src.transpose(Image.ROTATE_270)
            tmp.save("data/1/spin270_" + imgName)

            tmp = img_src.transpose(Image.FLIP_LEFT_RIGHT)
            tmp.save("data/1/flipLR_" + imgName)

            print("{} is done".format(imgName))

if __name__ == '__main__':
    imgNames = os.listdir("/Users/zero/Desktop/local/study_machine_learning/img/aka_seiber")
    print(imgNames)
    spinImg(imgNames)
