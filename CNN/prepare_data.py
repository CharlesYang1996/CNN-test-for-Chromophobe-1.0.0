#xu_yang 2020/5/25 cell_detection_1.0.0
#get masked cells image
import cv2 as cv
import os
import matplotlib.pyplot as plt
import time
from math_test import area_calculate_from_points
from pylab import *
from pixelbetweenpoints import pixel_between_two_points

import glob
import tkinter as tk
from tkinter import filedialog


def step1(file_path):
    cell_area_hist_list=[]
    print("###\n###\n###\n========================Step 1 Start========================\n file path: "+file_path)
    basename = os.path.basename(file_path)[:-4]
    img = cv.imread(file_path)

    print("Img size: [Width :",img.shape[0],"]","[Height :",img.shape[1],"]")


    img_masked=img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gauss = cv.GaussianBlur(gray, (5, 5), 5)

    ret, thresh = cv.threshold(gauss, 190, 255, 0)

    erode = cv.erode(thresh, None, iterations=2)

    #cv.imshow("erode",erode)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            erode[0][j]=255
    #-----find contours-----
    cnts, hierarchy = cv.findContours(erode.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)


    def cnt_area(cnt):
      area = cv.contourArea(cnt)
      return area
    counter_number=0

    for i in range(0, len(cnts)):
        x, y, w, h = cv.boundingRect(cnts[i])
        ratio=h/w
        if 600 <= cnt_area(cnts[i]) <= 2000 and  0.65<=ratio<=1.5 :
            cell_area_hist_list.append(cnt_area(cnts[i]))

            counter_number+=1

            cv.drawContours(img_masked, cnts[i], -1, (0, 0, 255), 2) #draw contours
            #CUT------------------
            x, y, w, h = cv.boundingRect(cnts[i])
            try:
                newimage = img[y :y + h , x :x + w ]  # 先用y确定高，再用x确定宽
                newimage = cv.resize(newimage, (40, 40), interpolation=cv.INTER_CUBIC)
                nrootdir = ("G:/2020summer/Project/CNN/output_single/")

                if not os.path.isdir(nrootdir):
                    os.makedirs(nrootdir)
                cv.imwrite(nrootdir +str(basename)+"_"+str(i) + ".jpg", newimage)
            except:
                pass


    print("total cells number : ", counter_number)


    #cv.imshow('img_copy', img_masked)
    print("========================Step 1 End========================")
    #cv.waitKey()



def walkFile():
    root = tk.Tk()
    root.withdraw()

    file = filedialog.askdirectory()
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            root_file=os.path.join(root, f)
            #print(root_file)


            step1(root_file)
        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))

walkFile()